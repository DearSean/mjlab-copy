"""Compile accepted human reviews into a deterministic training manifest.

Example:
  uv run python -m mjlab.tasks.velocity.recovery_review.compile \
    --review-file artifacts/recovery/review/reviewer-01.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tyro

from mjlab.tasks.velocity.recovery_review.prepare import DEFAULT_REVIEW_DIR
from mjlab.tasks.velocity.recovery_review.schema import (
  REVIEWED_MANIFEST_SCHEMA_VERSION,
  RecoveryReviewCandidate,
  file_sha256,
  load_review_queue,
  load_review_state,
  write_json_atomic,
)


@dataclass(kw_only=True)
class CompileReviewCfg:
  """Configuration for creating the reviewed training manifest."""

  review_file: Path
  review_dir: Path = DEFAULT_REVIEW_DIR
  output: Path | None = None
  require_complete: bool = True


def compile_reviewed_manifest(
  *,
  manifest_path: str | Path,
  queue_path: str | Path,
  review_path: str | Path,
  output_path: str | Path,
  require_complete: bool = True,
) -> dict[str, Any]:
  """Validate one adjudicated review and write accepted segments only."""
  manifest_path = Path(manifest_path)
  queue_path = Path(queue_path)
  queue = load_review_queue(queue_path)
  state = load_review_state(review_path, queue_path=queue_path)
  if file_sha256(manifest_path) != queue.source_manifest_sha256:
    raise ValueError("Review queue belongs to a different source manifest.")
  with manifest_path.open(encoding="utf-8") as stream:
    source_manifest = json.load(stream)
  if not isinstance(source_manifest, dict):
    raise ValueError("Source manifest must contain a JSON object.")

  pending = [
    candidate.candidate_id
    for candidate in queue.candidates
    if candidate.candidate_id not in state.segment_annotations
    or state.segment_annotations[candidate.candidate_id].decision == "pending"
  ]
  if require_complete and pending:
    raise ValueError(
      f"Cannot compile: {len(pending)} recovery candidates are still pending."
    )

  candidates_by_path: dict[str, list[dict[str, Any]]] = {}
  accepted_count = 0
  rejected_count = 0
  for candidate in queue.candidates:
    annotation = state.segment_annotations.get(candidate.candidate_id)
    if annotation is None or annotation.decision == "pending":
      continue
    annotation.validate(candidate)
    if annotation.decision == "rejected":
      rejected_count += 1
      continue
    accepted_count += 1
    candidates_by_path.setdefault(candidate.relative_path, []).append(
      _compiled_segment(candidate, asdict(annotation))
    )

  clips = source_manifest.get("clips")
  if not isinstance(clips, list):
    raise ValueError("Source manifest has no clip list.")
  for clip in clips:
    relative_path = str(clip["relative_path"])
    segments = candidates_by_path.get(relative_path, [])
    segments.sort(key=lambda segment: (segment["start_frame"], segment["end_frame"]))
    clip["segments"] = segments

  source_schema = source_manifest.get("schema_version")
  source_manifest["schema_version"] = REVIEWED_MANIFEST_SCHEMA_VERSION
  source_manifest["source_manifest_schema_version"] = source_schema
  source_manifest["source_manifest_sha256"] = queue.source_manifest_sha256
  source_manifest["review_queue_sha256"] = file_sha256(queue_path)
  source_manifest["reviewer"] = state.reviewer
  source_manifest["review_summary"] = {
    "candidate_count": len(queue.candidates),
    "accepted_count": accepted_count,
    "rejected_count": rejected_count,
    "pending_count": len(pending),
  }
  write_json_atomic(source_manifest, output_path)
  return source_manifest


def _compiled_segment(
  candidate: RecoveryReviewCandidate,
  annotation: dict[str, Any],
) -> dict[str, Any]:
  annotation.pop("decision", None)
  return {
    "candidate_id": candidate.candidate_id,
    "auto_start_frame": candidate.auto_start_frame,
    "auto_end_frame": candidate.auto_end_frame,
    "auto_support_complete_frame": candidate.auto_support_complete_frame,
    "auto_terminal_mode": candidate.auto_terminal_mode,
    **annotation,
  }


def main() -> None:
  cfg = tyro.cli(CompileReviewCfg)
  output = cfg.output or cfg.review_dir / "lafan_manifest.reviewed.json"
  result = compile_reviewed_manifest(
    manifest_path=cfg.review_dir / "lafan_manifest.json",
    queue_path=cfg.review_dir / "review_queue.json",
    review_path=cfg.review_file,
    output_path=output,
    require_complete=cfg.require_complete,
  )
  summary = result["review_summary"]
  print(
    "Compiled reviewed recovery manifest: "
    f"accepted={summary['accepted_count']}, rejected={summary['rejected_count']}, "
    f"pending={summary['pending_count']}, output={output.resolve()}"
  )


if __name__ == "__main__":
  main()
