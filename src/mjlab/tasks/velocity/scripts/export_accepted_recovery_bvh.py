"""Export accepted recovery-review segments as standalone BVH files.

Example:
  uv run python -m mjlab.tasks.velocity.scripts.export_accepted_recovery_bvh
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro

from mjlab.tasks.velocity.recovery_review.schema import (
  file_sha256,
  load_review_queue,
  load_review_state,
)


@dataclass(kw_only=True)
class ExportAcceptedRecoveryBvhCfg:
  """Configuration for exporting accepted, frame-accurate BVH segments."""

  dataset_root: Path = Path("data/lafan1")
  """Directory containing the source LaFAN BVH files."""

  review_dir: Path = Path("artifacts/recovery/review")
  """Directory containing review_queue.json and the reviewer state."""

  review_file: Path | None = None
  """Reviewer JSON; defaults to reviewer-01.json in review_dir."""

  output_dir: Path = Path("artifacts/recovery/lafan-accepted-bvh")
  """New directory for the exported BVH clips and index.json."""


def export_accepted_recovery_bvh(cfg: ExportAcceptedRecoveryBvhCfg) -> int:
  """Write every accepted review interval without modifying source BVHs."""
  dataset_root = cfg.dataset_root.resolve()
  review_dir = cfg.review_dir.resolve()
  review_file = (cfg.review_file or review_dir / "reviewer-01.json").resolve()
  queue_path = review_dir / "review_queue.json"
  output_dir = cfg.output_dir.resolve()
  if output_dir.exists():
    raise FileExistsError(
      f"Output directory already exists: {output_dir}. Choose a new output_dir."
    )

  queue = load_review_queue(queue_path)
  state = load_review_state(review_file, queue_path=queue_path)
  candidates = {candidate.candidate_id: candidate for candidate in queue.candidates}
  accepted = sorted(
    (
      (candidates[candidate_id], annotation)
      for candidate_id, annotation in state.segment_annotations.items()
      if annotation.decision == "accepted"
    ),
    key=lambda item: (
      item[0].relative_path,
      item[1].start_frame,
      item[1].end_frame,
      item[0].candidate_id,
    ),
  )
  if not accepted:
    raise ValueError(f"No accepted segments found in {review_file}.")

  output_dir.mkdir(parents=True)
  index: list[dict[str, object]] = []
  for ordinal, (candidate, annotation) in enumerate(accepted, start=1):
    annotation.validate(candidate)
    source_path = dataset_root / candidate.relative_path
    if file_sha256(source_path) != candidate.clip_sha256:
      raise ValueError(f"Source BVH changed since review: {candidate.relative_path}")
    filename = (
      f"{ordinal:03d}_{source_path.stem}_frames_"
      f"{annotation.start_frame:05d}-{annotation.end_frame:05d}.bvh"
    )
    output_path = output_dir / filename
    _write_bvh_segment(
      source_path,
      output_path,
      start_frame=annotation.start_frame,
      end_frame=annotation.end_frame,
    )
    index.append(
      {
        "file": filename,
        "source_path": candidate.relative_path,
        "source_sha256": candidate.clip_sha256,
        "candidate_id": candidate.candidate_id,
        "start_frame": annotation.start_frame,
        "end_frame": annotation.end_frame,
        "frame_count": annotation.end_frame - annotation.start_frame,
        "fps": candidate.fps,
        "annotation": asdict(annotation),
      }
    )
  (output_dir / "index.json").write_text(
    json.dumps(
      {
        "review_file": str(review_file),
        "reviewer": state.reviewer,
        "accepted_segment_count": len(index),
        "segments": index,
      },
      ensure_ascii=False,
      indent=2,
    )
    + "\n",
    encoding="utf-8",
  )
  return len(index)


def _write_bvh_segment(
  source_path: Path, output_path: Path, *, start_frame: int, end_frame: int
) -> None:
  text = source_path.read_text(encoding="utf-8")
  motion = re.search(r"(?m)^\s*MOTION\s*$", text)
  if motion is None:
    raise ValueError(f"BVH file has no MOTION section: {source_path}")
  header = text[: motion.end()]
  lines = text[motion.end() :].splitlines()
  nonempty = [line for line in lines if line.strip()]
  if len(nonempty) < 3:
    raise ValueError(f"BVH MOTION section is incomplete: {source_path}")
  frame_match = re.fullmatch(r"\s*Frames:\s*(\d+)\s*", nonempty[0])
  time_match = re.fullmatch(r"\s*Frame\s+Time:\s*([^\s]+)\s*", nonempty[1])
  if frame_match is None or time_match is None:
    raise ValueError(f"BVH MOTION header is invalid: {source_path}")
  source_frame_count = int(frame_match.group(1))
  frame_lines = nonempty[2:]
  if len(frame_lines) != source_frame_count:
    raise ValueError(
      f"Expected {source_frame_count} one-line BVH frames in {source_path}, "
      f"found {len(frame_lines)}."
    )
  if not 0 <= start_frame < end_frame <= source_frame_count:
    raise ValueError(
      f"Invalid frame interval [{start_frame}, {end_frame}) for {source_path}."
    )
  selected = frame_lines[start_frame:end_frame]
  output_path.write_text(
    "\n".join(
      (
        header.rstrip(),
        f"Frames: {len(selected)}",
        f"Frame Time: {time_match.group(1)}",
        *selected,
        "",
      )
    ),
    encoding="utf-8",
  )


def main() -> None:
  cfg = tyro.cli(ExportAcceptedRecoveryBvhCfg)
  count = export_accepted_recovery_bvh(cfg)
  print(f"Exported {count} accepted BVH segments to {cfg.output_dir.resolve()}")


if __name__ == "__main__":
  main()
