"""Compile reviewed LaFAN motion into deterministic MLD/SMP training shards."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mjlab.tasks.velocity.recovery_data.bvh import load_lafan_bvh
from mjlab.tasks.velocity.recovery_data.kinematic import (
  RecoveryKinematicEncoder,
  recovery_kinematic_feature_names,
)
from mjlab.tasks.velocity.recovery_data.resample import resample_canonical_motion
from mjlab.tasks.velocity.recovery_data.schema import (
  RECOVERY_KINEMATIC_DIM,
  RECOVERY_KINEMATIC_SCHEMA_VERSION,
)

RECOVERY_DATASET_SCHEMA_VERSION = "recovery-kinematic-dataset-v1"
_REVIEWED_MANIFEST_SCHEMA_VERSION = "recovery-reviewed-manifest-v1"
_SPLITS = ("train", "validation", "test")
_MOTION_SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
_RECOVERY_SPLIT_RATIOS = {"train": 0.7, "validation": 0.15, "test": 0.15}


@dataclass(frozen=True, kw_only=True)
class RecoveryDatasetCompilerCfg:
  """Inputs and deterministic output settings for the training dataset."""

  dataset_root: Path
  reviewed_manifest: Path = Path(
    "artifacts/recovery/review/lafan_manifest.reviewed.json"
  )
  output_dir: Path = Path("artifacts/recovery/dataset")
  target_fps: float = 20.0

  def __post_init__(self) -> None:
    if not np.isfinite(self.target_fps) or self.target_fps <= 0.0:
      raise ValueError("target_fps must be finite and positive.")


@dataclass
class _SplitBuffer:
  motion_features: list[NDArray[np.float32]] = field(default_factory=list)
  motion_offsets: list[int] = field(default_factory=lambda: [0])
  motion_clip_ids: list[str] = field(default_factory=list)
  motion_source_fps: list[float] = field(default_factory=list)
  motion_nominal_heights_m: list[float] = field(default_factory=list)
  recovery_features: list[NDArray[np.float32]] = field(default_factory=list)
  recovery_offsets: list[int] = field(default_factory=lambda: [0])
  candidate_ids: list[str] = field(default_factory=list)
  recovery_source_paths: list[str] = field(default_factory=list)
  source_start_frames: list[int] = field(default_factory=list)
  source_end_frames: list[int] = field(default_factory=list)
  support_start_frames: list[int] = field(default_factory=list)
  support_complete_frames: list[int] = field(default_factory=list)
  locomotion_takeover_frames: list[int] = field(default_factory=list)
  support_start_indices: list[int] = field(default_factory=list)
  support_complete_indices: list[int] = field(default_factory=list)
  locomotion_takeover_indices: list[int] = field(default_factory=list)
  initial_postures: list[str] = field(default_factory=list)
  terminal_modes: list[str] = field(default_factory=list)
  outcomes: list[str] = field(default_factory=list)
  complete_recovery: list[bool] = field(default_factory=list)
  recovery_nominal_heights_m: list[float] = field(default_factory=list)

  def add_motion(
    self,
    features: NDArray[np.float32],
    *,
    clip_id: str,
    source_fps: float,
    nominal_height_m: float,
  ) -> None:
    self.motion_features.append(features)
    self.motion_offsets.append(self.motion_offsets[-1] + features.shape[0])
    self.motion_clip_ids.append(clip_id)
    self.motion_source_fps.append(source_fps)
    self.motion_nominal_heights_m.append(nominal_height_m)

  def add_recovery(
    self,
    features: NDArray[np.float32],
    *,
    relative_path: str,
    source_fps: float,
    target_fps: float,
    target_start: int,
    nominal_height_m: float,
    segment: dict[str, Any],
  ) -> None:
    self.recovery_features.append(features)
    self.recovery_offsets.append(self.recovery_offsets[-1] + features.shape[0])
    self.candidate_ids.append(str(segment["candidate_id"]))
    self.recovery_source_paths.append(relative_path)
    self.source_start_frames.append(int(segment["start_frame"]))
    self.source_end_frames.append(int(segment["end_frame"]))
    self.support_start_frames.append(_optional_int(segment.get("support_start_frame")))
    self.support_complete_frames.append(
      _optional_int(segment.get("support_complete_frame"))
    )
    self.locomotion_takeover_frames.append(
      _optional_int(segment.get("locomotion_takeover_frame"))
    )
    length = features.shape[0]
    self.support_start_indices.append(
      _relative_event_index(
        segment.get("support_start_frame"),
        source_fps=source_fps,
        target_fps=target_fps,
        target_start=target_start,
        target_length=length,
      )
    )
    self.support_complete_indices.append(
      _relative_event_index(
        segment.get("support_complete_frame"),
        source_fps=source_fps,
        target_fps=target_fps,
        target_start=target_start,
        target_length=length,
      )
    )
    self.locomotion_takeover_indices.append(
      _relative_event_index(
        segment.get("locomotion_takeover_frame"),
        source_fps=source_fps,
        target_fps=target_fps,
        target_start=target_start,
        target_length=length,
      )
    )
    initial_posture = str(segment["initial_posture"])
    terminal_mode = str(segment["terminal_mode"])
    outcome = str(segment["outcome"])
    self.initial_postures.append(initial_posture)
    self.terminal_modes.append(terminal_mode)
    self.outcomes.append(outcome)
    self.complete_recovery.append(
      outcome == "success" and terminal_mode in {"stationary", "locomotion"}
    )
    self.recovery_nominal_heights_m.append(nominal_height_m)

  def arrays(self) -> dict[str, NDArray[Any]]:
    return {
      "motion_features": _concatenate_features(self.motion_features),
      "motion_offsets": np.asarray(self.motion_offsets, dtype=np.int64),
      "motion_clip_ids": _string_array(self.motion_clip_ids),
      "motion_source_fps": np.asarray(self.motion_source_fps, dtype=np.float64),
      "motion_nominal_heights_m": np.asarray(
        self.motion_nominal_heights_m, dtype=np.float32
      ),
      "recovery_features": _concatenate_features(self.recovery_features),
      "recovery_offsets": np.asarray(self.recovery_offsets, dtype=np.int64),
      "recovery_candidate_ids": _string_array(self.candidate_ids),
      "recovery_source_paths": _string_array(self.recovery_source_paths),
      "recovery_source_start_frames": np.asarray(
        self.source_start_frames, dtype=np.int64
      ),
      "recovery_source_end_frames": np.asarray(self.source_end_frames, dtype=np.int64),
      "recovery_support_start_frames": np.asarray(
        self.support_start_frames, dtype=np.int64
      ),
      "recovery_support_complete_frames": np.asarray(
        self.support_complete_frames, dtype=np.int64
      ),
      "recovery_locomotion_takeover_frames": np.asarray(
        self.locomotion_takeover_frames, dtype=np.int64
      ),
      "recovery_support_start_indices": np.asarray(
        self.support_start_indices, dtype=np.int64
      ),
      "recovery_support_complete_indices": np.asarray(
        self.support_complete_indices, dtype=np.int64
      ),
      "recovery_locomotion_takeover_indices": np.asarray(
        self.locomotion_takeover_indices, dtype=np.int64
      ),
      "recovery_initial_postures": _string_array(self.initial_postures),
      "recovery_terminal_modes": _string_array(self.terminal_modes),
      "recovery_outcomes": _string_array(self.outcomes),
      "recovery_complete": np.asarray(self.complete_recovery, dtype=np.bool_),
      "recovery_nominal_heights_m": np.asarray(
        self.recovery_nominal_heights_m, dtype=np.float32
      ),
    }


def compile_recovery_dataset(cfg: RecoveryDatasetCompilerCfg) -> dict[str, Any]:
  """Compile all LaFAN motion and accepted recovery segments into split shards."""
  dataset_root = cfg.dataset_root.resolve()
  reviewed_manifest = cfg.reviewed_manifest.resolve()
  output_dir = cfg.output_dir.resolve()
  if not dataset_root.is_dir():
    raise FileNotFoundError(f"LaFAN dataset root does not exist: {dataset_root}")
  if not reviewed_manifest.is_file():
    raise FileNotFoundError(f"Reviewed manifest does not exist: {reviewed_manifest}")

  manifest = _read_json(reviewed_manifest)
  if manifest.get("schema_version") != _REVIEWED_MANIFEST_SCHEMA_VERSION:
    raise ValueError(
      f"Expected {_REVIEWED_MANIFEST_SCHEMA_VERSION}, "
      f"got {manifest.get('schema_version')!r}."
    )
  review_summary = manifest.get("review_summary", {})
  if int(review_summary.get("pending_count", -1)) != 0:
    raise ValueError("Reviewed manifest still contains pending candidates.")
  if manifest.get("failures"):
    raise ValueError("Reviewed manifest contains source-file failures.")
  clips = manifest.get("clips")
  if not isinstance(clips, list) or not clips:
    raise ValueError("Reviewed manifest has no clips.")
  reviewed_segment_count = sum(len(clip.get("segments", [])) for clip in clips)
  if reviewed_segment_count != int(review_summary.get("accepted_count", -1)):
    raise ValueError(
      "Reviewed manifest accepted count does not match its compiled segments."
    )

  feature_names = recovery_kinematic_feature_names()
  encoder = RecoveryKinematicEncoder()
  buffers = {split: _SplitBuffer() for split in _SPLITS}
  source_hashes: dict[str, str] = {}
  recording_splits = _grouped_split_assignment(clips)

  for clip_number, clip_data in enumerate(clips, start=1):
    relative_path = str(clip_data["relative_path"])
    source_path = dataset_root / relative_path
    if not source_path.is_file():
      raise FileNotFoundError(
        f"Manifest source {relative_path!r} is absent under {dataset_root}."
      )
    actual_hash = _sha256(source_path)
    expected_hash = str(clip_data["sha256"])
    if actual_hash != expected_hash:
      raise ValueError(f"Source hash mismatch for {relative_path}.")
    source_hashes[relative_path] = actual_hash
    recording = str(clip_data["recording"])
    split = recording_splits[recording]

    source_motion = load_lafan_bvh(source_path)
    motion = resample_canonical_motion(source_motion, cfg.target_fps)
    kinematic = encoder.encode(motion)
    features = kinematic.features.astype(np.float32, copy=False)
    if tuple(kinematic.feature_names) != feature_names:
      raise RuntimeError(f"Feature schema changed while encoding {relative_path}.")
    buffers[split].add_motion(
      features,
      clip_id=relative_path,
      source_fps=source_motion.fps,
      nominal_height_m=kinematic.nominal_height_m,
    )

    segments = clip_data.get("segments", [])
    if not isinstance(segments, list):
      raise ValueError(f"segments must be a list for {relative_path}.")
    for segment in segments:
      start_frame = int(segment["start_frame"])
      end_frame = int(segment["end_frame"])
      if not 0 <= start_frame < end_frame <= source_motion.frame_count:
        raise ValueError(
          f"Invalid reviewed interval [{start_frame}, {end_frame}) for {relative_path}."
        )
      target_start, target_end = _resampled_interval(
        start_frame,
        end_frame,
        source_fps=source_motion.fps,
        target_fps=cfg.target_fps,
        target_frame_count=motion.frame_count,
      )
      recovery = features[target_start:target_end]
      if recovery.shape[0] == 0:
        raise ValueError(f"Reviewed interval became empty for {relative_path}.")
      buffers[split].add_recovery(
        recovery,
        relative_path=relative_path,
        source_fps=source_motion.fps,
        target_fps=cfg.target_fps,
        target_start=target_start,
        nominal_height_m=kinematic.nominal_height_m,
        segment=segment,
      )
    print(
      f"[{clip_number:02d}/{len(clips):02d}] {relative_path}: "
      f"{source_motion.frame_count} -> {motion.frame_count} frames, "
      f"{len(segments)} reviewed recoveries"
    )

  output_dir.mkdir(parents=True, exist_ok=True)
  split_arrays = {split: buffers[split].arrays() for split in _SPLITS}
  artifact_hashes: dict[str, str] = {}
  for split in _SPLITS:
    path = output_dir / f"{split}.npz"
    _write_npz_atomic(path, split_arrays[split])
    artifact_hashes[path.name] = _sha256(path)

  train_features = split_arrays["train"]["motion_features"]
  if train_features.shape[0] == 0:
    raise ValueError("The training split contains no motion frames.")
  normalizer = {
    "feature_names": _string_array(list(feature_names)),
    "mean": train_features.mean(axis=0, dtype=np.float64).astype(np.float32),
    "std": np.maximum(train_features.std(axis=0, dtype=np.float64), 1e-6).astype(
      np.float32
    ),
    "frame_count": np.asarray([train_features.shape[0]], dtype=np.int64),
  }
  normalizer_path = output_dir / "normalizer.npz"
  _write_npz_atomic(normalizer_path, normalizer)
  artifact_hashes[normalizer_path.name] = _sha256(normalizer_path)

  quality = _quality_report(split_arrays)
  quality_path = output_dir / "quality_report.json"
  _write_json_atomic(quality_path, quality)
  artifact_hashes[quality_path.name] = _sha256(quality_path)

  dataset_manifest = {
    "schema_version": RECOVERY_DATASET_SCHEMA_VERSION,
    "feature_schema_version": RECOVERY_KINEMATIC_SCHEMA_VERSION,
    "feature_schema_sha256": _json_sha256(list(feature_names)),
    "feature_dim": RECOVERY_KINEMATIC_DIM,
    "feature_names": list(feature_names),
    "target_fps": cfg.target_fps,
    "dataset_name": str(manifest.get("dataset_name", dataset_root.name)),
    "reviewed_manifest_sha256": _sha256(reviewed_manifest),
    "review_queue_sha256": manifest.get("review_queue_sha256"),
    "source_hashes": source_hashes,
    "recording_splits": recording_splits,
    "splits": quality["splits"],
    "artifacts": artifact_hashes,
  }
  manifest_path = output_dir / "dataset_manifest.json"
  _write_json_atomic(manifest_path, dataset_manifest)
  return dataset_manifest


def _grouped_split_assignment(clips: list[dict[str, Any]]) -> dict[str, str]:
  """Balance recovery groups without splitting synchronized recordings.

  A plain hash split is badly imbalanced for LaFAN recovery because one
  recording contains roughly half of all accepted intervals. The three largest
  recovery recordings seed train, test, and validation; remaining recovery
  recordings fill deterministic 70/15/15 deficits. Other recordings then fill
  80/10/10 motion-frame deficits for VAE pretraining.
  """
  groups: dict[str, dict[str, int]] = {}
  for clip in clips:
    recording = str(clip["recording"])
    group = groups.setdefault(recording, {"frames": 0, "recoveries": 0})
    group["frames"] += int(clip["frame_count"])
    segments = clip.get("segments", [])
    if not isinstance(segments, list):
      raise ValueError(f"segments must be a list for recording {recording}.")
    group["recoveries"] += len(segments)

  assignment: dict[str, str] = {}
  recovery_groups = sorted(
    (
      (recording, counts)
      for recording, counts in groups.items()
      if counts["recoveries"] > 0
    ),
    key=lambda item: (-item[1]["recoveries"], item[0]),
  )
  seed_splits = ("train", "test", "validation")
  for (recording, _), split in zip(recovery_groups, seed_splits, strict=False):
    assignment[recording] = split

  recovery_total = sum(counts["recoveries"] for _, counts in recovery_groups)
  recovery_counts = {
    split: sum(
      counts["recoveries"]
      for recording, counts in recovery_groups
      if assignment.get(recording) == split
    )
    for split in _SPLITS
  }
  for recording, counts in recovery_groups[len(seed_splits) :]:
    split = _largest_deficit_split(
      recovery_counts,
      total=recovery_total,
      ratios=_RECOVERY_SPLIT_RATIOS,
    )
    assignment[recording] = split
    recovery_counts[split] += counts["recoveries"]

  frame_total = sum(counts["frames"] for counts in groups.values())
  frame_counts = {
    split: sum(
      groups[recording]["frames"]
      for recording, assigned in assignment.items()
      if assigned == split
    )
    for split in _SPLITS
  }
  remaining = sorted(
    (
      (recording, counts)
      for recording, counts in groups.items()
      if recording not in assignment
    ),
    key=lambda item: (-item[1]["frames"], item[0]),
  )
  for recording, counts in remaining:
    split = _largest_deficit_split(
      frame_counts,
      total=frame_total,
      ratios=_MOTION_SPLIT_RATIOS,
    )
    assignment[recording] = split
    frame_counts[split] += counts["frames"]
  return dict(sorted(assignment.items()))


def _largest_deficit_split(
  counts: dict[str, int],
  *,
  total: int,
  ratios: dict[str, float],
) -> str:
  priority = {"train": 2, "validation": 1, "test": 0}
  return max(
    _SPLITS,
    key=lambda split: (ratios[split] * total - counts[split], priority[split]),
  )


def _resampled_interval(
  start_frame: int,
  end_frame: int,
  *,
  source_fps: float,
  target_fps: float,
  target_frame_count: int,
) -> tuple[int, int]:
  ratio = target_fps / source_fps
  target_start = int(np.ceil(start_frame * ratio - 1e-9))
  target_end = int(np.ceil(end_frame * ratio - 1e-9))
  target_start = int(np.clip(target_start, 0, target_frame_count - 1))
  target_end = int(np.clip(target_end, target_start + 1, target_frame_count))
  return target_start, target_end


def _relative_event_index(
  source_frame: Any,
  *,
  source_fps: float,
  target_fps: float,
  target_start: int,
  target_length: int,
) -> int:
  if source_frame is None:
    return -1
  target_frame = int(round(float(source_frame) * target_fps / source_fps))
  return int(np.clip(target_frame - target_start, 0, target_length - 1))


def _optional_int(value: Any) -> int:
  return -1 if value is None else int(value)


def _concatenate_features(
  arrays: list[NDArray[np.float32]],
) -> NDArray[np.float32]:
  if not arrays:
    return np.empty((0, RECOVERY_KINEMATIC_DIM), dtype=np.float32)
  return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)


def _string_array(values: list[str]) -> NDArray[np.str_]:
  width = max((len(value) for value in values), default=1)
  return np.asarray(values, dtype=f"<U{width}")


def _quality_report(split_arrays: dict[str, dict[str, NDArray[Any]]]) -> dict[str, Any]:
  splits: dict[str, Any] = {}
  for split, arrays in split_arrays.items():
    motion = arrays["motion_features"]
    recovery = arrays["recovery_features"]
    if not np.all(np.isfinite(motion)) or not np.all(np.isfinite(recovery)):
      raise ValueError(f"Compiled {split} features contain NaN or Inf.")
    complete = arrays["recovery_complete"]
    splits[split] = {
      "motion_clip_count": int(arrays["motion_clip_ids"].shape[0]),
      "motion_frame_count": int(motion.shape[0]),
      "recovery_segment_count": int(complete.shape[0]),
      "recovery_frame_count": int(recovery.shape[0]),
      "complete_recovery_count": int(np.count_nonzero(complete)),
      "partial_recovery_count": int(complete.size - np.count_nonzero(complete)),
      "finite": True,
    }
  return {
    "schema_version": RECOVERY_DATASET_SCHEMA_VERSION,
    "feature_dim": RECOVERY_KINEMATIC_DIM,
    "splits": splits,
  }


def _write_npz_atomic(path: Path, arrays: dict[str, NDArray[Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary_name = tempfile.mkstemp(
    dir=path.parent, prefix=f".{path.name}."
  )
  os.close(descriptor)
  temporary = Path(temporary_name)
  try:
    with zipfile.ZipFile(
      temporary,
      mode="w",
      compression=zipfile.ZIP_DEFLATED,
      compresslevel=6,
    ) as archive:
      for name in sorted(arrays):
        buffer = io.BytesIO()
        np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
        info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o600 << 16
        archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary_name = tempfile.mkstemp(
    dir=path.parent, prefix=f".{path.name}."
  )
  temporary = Path(temporary_name)
  try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
      json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
      stream.write("\n")
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    value = json.load(stream)
  if not isinstance(value, dict):
    raise ValueError(f"Expected a JSON object in {path}.")
  return value


def _json_sha256(value: Any) -> str:
  payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
  return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while block := stream.read(1024 * 1024):
      digest.update(block)
  return digest.hexdigest()
