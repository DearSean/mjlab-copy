"""Prepare one self-contained directory for LaFAN recovery review.

Example:
  uv run python -m mjlab.tasks.velocity.recovery_review.prepare \
    --dataset-root /path/to/lafan1
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from mjlab.tasks.velocity.recovery_data import (
  RecoverySemanticEncoder,
  build_recovery_manifest,
  load_lafan_bvh,
  write_recovery_manifest,
)
from mjlab.tasks.velocity.recovery_data.schema import SemanticMotion
from mjlab.tasks.velocity.recovery_data.semantic import CONTACT_NAMES
from mjlab.tasks.velocity.recovery_review.candidates import (
  RecoveryCandidateDetectorCfg,
  detect_recovery_candidates,
)
from mjlab.tasks.velocity.recovery_review.schema import (
  REVIEW_QUEUE_SCHEMA_VERSION,
  RecoveryReviewCandidate,
  ReviewFrame,
  ReviewQueue,
  file_sha256,
  frame_id,
  save_review_queue,
)

DEFAULT_REVIEW_DIR = Path("artifacts/recovery/review")


@dataclass(kw_only=True)
class PrepareReviewCfg:
  """Configuration for building a deterministic human-review queue."""

  dataset_root: Path
  """Directory containing the original LaFAN BVH files."""

  output_dir: Path = DEFAULT_REVIEW_DIR
  """One directory that will contain every review artifact."""

  manifest: Path | None = None
  """Optional existing manifest; otherwise it is rebuilt in output_dir."""

  validation_frames: int = 200
  """Number of validation frames sampled for threshold tuning."""

  test_frames: int = 200
  """Number of frozen test frames sampled for final label evaluation."""

  seed: int = 42
  """Deterministic audit-frame sampling seed."""


@dataclass(frozen=True)
class _AuditRecord:
  candidate: RecoveryReviewCandidate
  semantic: SemanticMotion


def prepare_review_directory(cfg: PrepareReviewCfg) -> ReviewQueue:
  """Build or copy the manifest and write its review queue into one folder."""
  if cfg.validation_frames < 0 or cfg.test_frames < 0:
    raise ValueError("Audit frame counts must be non-negative.")
  dataset_root = cfg.dataset_root.resolve()
  if not dataset_root.is_dir():
    raise FileNotFoundError(f"LaFAN dataset directory does not exist: {dataset_root}")
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  local_manifest_path = cfg.output_dir / "lafan_manifest.json"
  if cfg.manifest is None:
    manifest = build_recovery_manifest(dataset_root)
    write_recovery_manifest(manifest, local_manifest_path)
  else:
    source_manifest = cfg.manifest.resolve()
    if not source_manifest.is_file():
      raise FileNotFoundError(f"Manifest does not exist: {source_manifest}")
    if source_manifest != local_manifest_path.resolve():
      shutil.copyfile(source_manifest, local_manifest_path)

  manifest_data = _load_manifest(local_manifest_path)
  manifest_sha256 = file_sha256(local_manifest_path)
  encoder = RecoverySemanticEncoder()
  detector_cfg = RecoveryCandidateDetectorCfg()
  candidates: list[RecoveryReviewCandidate] = []
  audit_records: list[_AuditRecord] = []

  for clip_data in manifest_data["clips"]:
    if not clip_data["recovery_source"]:
      continue
    relative_path = str(clip_data["relative_path"])
    motion = load_lafan_bvh(dataset_root / relative_path)
    if file_sha256(dataset_root / relative_path) != clip_data["sha256"]:
      raise ValueError(f"Dataset file changed after manifesting: {relative_path}")
    semantic = encoder.encode(motion)
    clip_candidates = detect_recovery_candidates(
      motion,
      semantic,
      relative_path=relative_path,
      clip_sha256=str(clip_data["sha256"]),
      recording=str(clip_data["recording"]),
      subject=str(clip_data["subject"]),
      split=str(clip_data["split"]),
      cfg=detector_cfg,
    )
    candidates.extend(clip_candidates)
    audit_records.extend(
      _AuditRecord(candidate=candidate, semantic=semantic)
      for candidate in clip_candidates
    )

  candidates.sort(
    key=lambda candidate: (
      candidate.relative_path,
      candidate.auto_start_frame,
      candidate.auto_end_frame,
    )
  )
  rng = np.random.default_rng(cfg.seed)
  validation_frames = _sample_audit_frames(
    audit_records,
    split="validation",
    count=cfg.validation_frames,
    rng=rng,
  )
  test_frames = _sample_audit_frames(
    audit_records,
    split="test",
    count=cfg.test_frames,
    rng=rng,
  )
  queue = ReviewQueue(
    schema_version=REVIEW_QUEUE_SCHEMA_VERSION,
    source_manifest_sha256=manifest_sha256,
    dataset_name=str(manifest_data["dataset_name"]),
    candidates=tuple(candidates),
    validation_frames=validation_frames,
    test_frames=test_frames,
  )
  save_review_queue(queue, cfg.output_dir / "review_queue.json")
  return queue


def _sample_audit_frames(
  records: list[_AuditRecord],
  *,
  split: str,
  count: int,
  rng: np.random.Generator,
) -> tuple[ReviewFrame, ...]:
  if count == 0:
    return ()
  frame_pool: dict[tuple[str, int], ReviewFrame] = {}
  bins: dict[str, list[tuple[str, int]]] = {
    phase: [] for phase in ("fallen", "transition", "standing")
  }
  for contact in CONTACT_NAMES:
    bins[f"{contact}:contact"] = []
    bins[f"{contact}:no_contact"] = []
    bins[f"{contact}:boundary"] = []

  clearance_thresholds = np.asarray(
    RecoverySemanticEncoder().cfg.contact_clearance_height_ratios
  )
  for record in records:
    candidate = record.candidate
    if candidate.split != split:
      continue
    semantic = record.semantic
    for frame in range(candidate.auto_start_frame, candidate.auto_end_frame):
      key = (candidate.relative_path, frame)
      if key in frame_pool:
        continue
      progress = float(semantic.progress[frame])
      phase = (
        "fallen"
        if progress <= 0.35
        else "standing"
        if progress >= 0.85
        else "transition"
      )
      contacts = tuple(bool(value) for value in semantic.contacts[frame])
      frame_pool[key] = ReviewFrame(
        frame_id=frame_id(candidate.clip_sha256, frame),
        relative_path=candidate.relative_path,
        clip_sha256=candidate.clip_sha256,
        split=split,
        frame=frame,
        clip_frame_count=candidate.clip_frame_count,
        fps=candidate.fps,
        auto_phase=phase,
        auto_contacts=contacts,
      )
      bins[phase].append(key)
      clearances = semantic.features[frame, 67:75]
      for index, contact in enumerate(CONTACT_NAMES):
        label = "contact" if contacts[index] else "no_contact"
        bins[f"{contact}:{label}"].append(key)
        if abs(float(clearances[index]) - clearance_thresholds[index]) <= 0.02:
          bins[f"{contact}:boundary"].append(key)

  if len(frame_pool) < count:
    raise ValueError(
      f"Requested {count} {split} audit frames, but only {len(frame_pool)} "
      "unique candidate frames are available."
    )
  for values in bins.values():
    rng.shuffle(values)
  selected: list[tuple[str, int]] = []
  seen: set[tuple[str, int]] = set()
  active_bins = [values for values in bins.values() if values]
  positions = [0] * len(active_bins)
  while len(selected) < count and active_bins:
    advanced = False
    for index, values in enumerate(active_bins):
      while positions[index] < len(values) and values[positions[index]] in seen:
        positions[index] += 1
      if positions[index] >= len(values):
        continue
      key = values[positions[index]]
      positions[index] += 1
      selected.append(key)
      seen.add(key)
      advanced = True
      if len(selected) == count:
        break
    if not advanced:
      break
  if len(selected) < count:
    remaining = [key for key in frame_pool if key not in seen]
    rng.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)])
  return tuple(frame_pool[key] for key in selected)


def _load_manifest(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    value = json.load(stream)
  if not isinstance(value, dict) or not isinstance(value.get("clips"), list):
    raise ValueError(f"Invalid recovery manifest: {path}")
  return value


def main() -> None:
  cfg = tyro.cli(PrepareReviewCfg)
  queue = prepare_review_directory(cfg)
  output = cfg.output_dir.resolve()
  print(
    "Prepared recovery review directory: "
    f"candidates={len(queue.candidates)}, "
    f"validation_frames={len(queue.validation_frames)}, "
    f"test_frames={len(queue.test_frames)}, output={output}"
  )


if __name__ == "__main__":
  main()
