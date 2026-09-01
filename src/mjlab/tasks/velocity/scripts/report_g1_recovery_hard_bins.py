"""Resolve adaptive G1 recovery hard-bin scalars back to source clips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass(frozen=True, kw_only=True)
class G1RecoveryHardBinReportCfg:
  run_dir: Path
  dataset_dir: Path = Path("artifacts/g1_recovery")
  rank_count: int = 5


def report_g1_recovery_hard_bins(
  cfg: G1RecoveryHardBinReportCfg,
) -> list[dict[str, Any]]:
  """Read the latest stable hard-bin report and attach manifest metadata."""
  if cfg.rank_count <= 0:
    raise ValueError("rank_count must be positive.")
  manifest = json.loads((cfg.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
  train_clips = [clip for clip in manifest["clips"] if clip["split"] == "train"]
  events = EventAccumulator(str(cfg.run_dir), size_guidance={"scalars": 0})
  events.Reload()
  scalar_tags = set(events.Tags()["scalars"])
  prefix = "Curriculum/g1_recovery_assist/"
  required = f"{prefix}hard_bin_0_global_id"
  if required not in scalar_tags:
    raise ValueError(
      "The run does not contain adaptive hard-bin diagnostics. Start a new run "
      "with the adaptive sampler enabled."
    )

  report: list[dict[str, Any]] = []
  for rank in range(cfg.rank_count):
    tag_prefix = f"{prefix}hard_bin_{rank}"
    global_id = round(_latest_scalar(events, f"{tag_prefix}_global_id"))
    if global_id < 0:
      continue
    clip_index = round(_latest_scalar(events, f"{tag_prefix}_clip_index"))
    if not 0 <= clip_index < len(train_clips):
      raise ValueError(f"Logged clip index {clip_index} is outside the train split.")
    clip = train_clips[clip_index]
    report.append(
      {
        "rank": rank,
        "global_bin_id": global_id,
        "clip_index": clip_index,
        "clip_id": clip["clip_id"],
        "source_recording_id": clip["source_recording_id"],
        "temporal_bin_index": round(
          _latest_scalar(events, f"{tag_prefix}_temporal_index")
        ),
        "source_start_frame": round(
          _latest_scalar(events, f"{tag_prefix}_source_start_frame")
        ),
        "source_end_frame": round(
          _latest_scalar(events, f"{tag_prefix}_source_end_frame")
        ),
        "attempts": round(_latest_scalar(events, f"{tag_prefix}_attempts")),
        "success_rate": _latest_scalar(events, f"{tag_prefix}_success_rate"),
        "failure_ema": _latest_scalar(events, f"{tag_prefix}_failure_ema"),
      }
    )
  return report


def _latest_scalar(events: EventAccumulator, tag: str) -> float:
  values = events.Scalars(tag)
  if not values:
    raise ValueError(f"TensorBoard scalar has no values: {tag}")
  return float(values[-1].value)


def main() -> None:
  cfg = tyro.cli(G1RecoveryHardBinReportCfg)
  report = report_g1_recovery_hard_bins(cfg)
  if not report:
    print("No temporal bin has reached the minimum evidence threshold yet.")
    return
  for item in report:
    print(
      f"#{item['rank'] + 1}: {item['clip_id']} "
      f"source={item['source_recording_id']} "
      f"frames=[{item['source_start_frame']}, {item['source_end_frame']}) "
      f"success={item['success_rate']:.3f} "
      f"failure_ema={item['failure_ema']:.3f} "
      f"attempts={item['attempts']}"
    )


if __name__ == "__main__":
  main()
