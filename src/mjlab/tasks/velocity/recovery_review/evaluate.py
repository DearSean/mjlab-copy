"""Evaluate automatic recovery labels against blind human annotations.

Example:
  uv run python -m mjlab.tasks.velocity.recovery_review.evaluate \
    --review-file artifacts/recovery/review/reviewer-01.json \
    --partition validation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tyro

from mjlab.tasks.velocity.recovery_data.semantic import CONTACT_NAMES
from mjlab.tasks.velocity.recovery_review.prepare import DEFAULT_REVIEW_DIR
from mjlab.tasks.velocity.recovery_review.schema import (
  ReviewFrame,
  load_review_queue,
  load_review_state,
  write_json_atomic,
)


@dataclass(kw_only=True)
class EvaluateReviewCfg:
  """Configuration for semantic-label review metrics."""

  review_file: Path
  review_dir: Path = DEFAULT_REVIEW_DIR
  partition: Literal["validation", "test"] = "validation"
  output: Path | None = None
  require_complete: bool = True
  minimum_macro_f1: float = 0.90


def evaluate_review_labels(
  frames: tuple[ReviewFrame, ...],
  frame_annotations: dict[str, Any],
  *,
  require_complete: bool = True,
) -> dict[str, Any]:
  """Compute phase and per-contact F1 while excluding uncertain labels."""
  missing = [
    frame.frame_id
    for frame in frames
    if frame.frame_id not in frame_annotations
    or not frame_annotations[frame.frame_id].complete
  ]
  if require_complete and missing:
    raise ValueError(f"Cannot evaluate: {len(missing)} audit frames are incomplete.")

  phase_truth: list[str] = []
  phase_prediction: list[str] = []
  contact_truth: dict[str, list[bool]] = {name: [] for name in CONTACT_NAMES}
  contact_prediction: dict[str, list[bool]] = {name: [] for name in CONTACT_NAMES}
  for frame in frames:
    annotation = frame_annotations.get(frame.frame_id)
    if annotation is None:
      continue
    if annotation.phase not in {"unreviewed", "uncertain"}:
      phase_truth.append(annotation.phase)
      phase_prediction.append(frame.auto_phase)
    for index, contact in enumerate(CONTACT_NAMES):
      label = annotation.contacts[contact]
      if label in {"unreviewed", "uncertain"}:
        continue
      contact_truth[contact].append(label == "contact")
      contact_prediction[contact].append(frame.auto_contacts[index])

  phase_scores = {
    phase: _class_f1(phase_truth, phase_prediction, phase)
    for phase in ("fallen", "transition", "standing")
  }
  contact_scores = {
    contact: _binary_f1(contact_truth[contact], contact_prediction[contact])
    for contact in CONTACT_NAMES
  }
  return {
    "frame_count": len(frames),
    "complete_frame_count": len(frames) - len(missing),
    "phase": {
      "per_class_f1": phase_scores,
      "macro_f1": _finite_mean(phase_scores.values()),
      "labeled_count": len(phase_truth),
    },
    "contacts": {
      "per_site_f1": contact_scores,
      "macro_f1": _finite_mean(contact_scores.values()),
      "labeled_count": sum(len(values) for values in contact_truth.values()),
    },
  }


def _class_f1(truth: list[str], prediction: list[str], target: str) -> float | None:
  pairs = tuple(zip(truth, prediction, strict=True))
  true_positive = sum(t == target and p == target for t, p in pairs)
  false_positive = sum(t != target and p == target for t, p in pairs)
  false_negative = sum(t == target and p != target for t, p in pairs)
  denominator = 2 * true_positive + false_positive + false_negative
  return None if denominator == 0 else 2.0 * true_positive / denominator


def _binary_f1(truth: list[bool], prediction: list[bool]) -> float | None:
  if not truth:
    return None
  true_array = np.asarray(truth, dtype=np.bool_)
  prediction_array = np.asarray(prediction, dtype=np.bool_)
  true_positive = int(np.sum(true_array & prediction_array))
  false_positive = int(np.sum(~true_array & prediction_array))
  false_negative = int(np.sum(true_array & ~prediction_array))
  denominator = 2 * true_positive + false_positive + false_negative
  return None if denominator == 0 else 2.0 * true_positive / denominator


def _finite_mean(values: Any) -> float | None:
  finite = [
    float(value) for value in values if value is not None and np.isfinite(value)
  ]
  return None if not finite else float(np.mean(finite))


def main() -> None:
  cfg = tyro.cli(EvaluateReviewCfg)
  queue_path = cfg.review_dir / "review_queue.json"
  queue = load_review_queue(queue_path)
  state = load_review_state(cfg.review_file, queue_path=queue_path)
  frames = (
    queue.validation_frames if cfg.partition == "validation" else queue.test_frames
  )
  metrics = evaluate_review_labels(
    frames,
    state.frame_annotations,
    require_complete=cfg.require_complete,
  )
  metrics["partition"] = cfg.partition
  metrics["reviewer"] = state.reviewer
  output = cfg.output or cfg.review_dir / f"metrics.{cfg.partition}.json"
  write_json_atomic(metrics, output)
  phase_f1 = metrics["phase"]["macro_f1"]
  contact_f1 = metrics["contacts"]["macro_f1"]
  print(
    f"Review metrics: phase_macro_f1={phase_f1}, "
    f"contact_macro_f1={contact_f1}, output={output.resolve()}"
  )
  if cfg.require_complete and (
    phase_f1 is None
    or contact_f1 is None
    or phase_f1 < cfg.minimum_macro_f1
    or contact_f1 < cfg.minimum_macro_f1
  ):
    raise SystemExit(
      f"Review label quality did not reach macro-F1 {cfg.minimum_macro_f1:.2f}."
    )


if __name__ == "__main__":
  main()
