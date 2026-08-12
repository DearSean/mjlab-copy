"""High-recall stationary and dynamic recovery candidate detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mjlab.tasks.velocity.recovery_data.schema import (
  CanonicalMotionClip,
  SemanticMotion,
)
from mjlab.tasks.velocity.recovery_data.semantic import classify_initial_posture
from mjlab.tasks.velocity.recovery_review.schema import (
  InitialPosture,
  RecoveryReviewCandidate,
  TerminalMode,
  candidate_id,
)


@dataclass(frozen=True, kw_only=True)
class RecoveryCandidateDetectorCfg:
  """Dimensionless high-recall thresholds for a human review queue."""

  low_progress: float = 0.35
  static_progress: float = 0.85
  static_hold_s: float = 0.5
  dynamic_height_ratio: float = 0.40
  dynamic_uprightness: float = 0.55
  dynamic_horizontal_speed_height_per_s: float = 0.20
  dynamic_hold_s: float = 0.25
  support_height_ratio: float = 0.36
  support_uprightness: float = 0.45
  support_hold_s: float = 0.20
  safe_height_ratio: float = 0.30
  safe_uprightness: float = 0.30
  safe_future_s: float = 0.6
  foot_evidence_window_s: float = 0.4
  pre_roll_s: float = 0.25
  post_roll_s: float = 0.25
  min_duration_s: float = 0.75
  max_duration_s: float = 8.0

  def __post_init__(self) -> None:
    if not 0.0 <= self.low_progress < self.static_progress <= 1.0:
      raise ValueError("Recovery progress thresholds are invalid.")
    if not 0.0 < self.support_height_ratio <= self.dynamic_height_ratio:
      raise ValueError("Recovery height thresholds are invalid.")
    if not 0.0 < self.support_uprightness <= self.dynamic_uprightness <= 1.0:
      raise ValueError("Recovery uprightness thresholds are invalid.")
    durations = (
      self.static_hold_s,
      self.dynamic_hold_s,
      self.support_hold_s,
      self.safe_future_s,
      self.foot_evidence_window_s,
      self.min_duration_s,
      self.max_duration_s,
    )
    if any(duration <= 0.0 for duration in durations):
      raise ValueError("Candidate-detector durations must be positive.")
    if self.pre_roll_s < 0.0 or self.post_roll_s < 0.0:
      raise ValueError("Candidate context durations must be non-negative.")
    if self.min_duration_s > self.max_duration_s:
      raise ValueError("Candidate duration bounds are invalid.")


def detect_recovery_candidates(
  motion: CanonicalMotionClip,
  semantic: SemanticMotion,
  *,
  relative_path: str,
  clip_sha256: str,
  recording: str,
  subject: str,
  split: str,
  cfg: RecoveryCandidateDetectorCfg | None = None,
) -> tuple[RecoveryReviewCandidate, ...]:
  """Detect recovery attempts ending in standing, locomotion, or support.

  This detector deliberately favors recall over precision: a human reviewer
  decides whether each proposal is a real recovery. Future frames are available
  because this is an offline data curation step, never an online policy signal.
  """
  if semantic.features.shape[0] != motion.frame_count:
    raise ValueError("Motion and semantic frame counts do not match.")
  config = cfg or RecoveryCandidateDetectorCfg()
  fps = motion.fps
  features = semantic.features.astype(np.float64, copy=False)
  progress = semantic.progress.astype(np.float64, copy=False)
  root_height = features[:, 0]
  torso_right = features[:, 1:4]
  torso_forward = features[:, 4:7]
  torso_up = np.cross(torso_right, torso_forward)
  uprightness = np.clip(torso_up[:, 2], -1.0, 1.0)
  horizontal_speed = np.linalg.norm(features[:, 7:9], axis=-1)
  nonfoot_contact = np.any(semantic.contacts[:, :6], axis=-1)
  foot_contact = np.any(semantic.contacts[:, 6:], axis=-1)

  safe = (root_height >= config.safe_height_ratio) & (
    uprightness >= config.safe_uprightness
  )
  safe_future = _forward_all(safe, _frames(config.safe_future_s, fps))
  foot_evidence = _centered_any(
    foot_contact,
    _frames(config.foot_evidence_window_s, fps),
  )
  no_body_support = _forward_all(
    ~nonfoot_contact,
    _frames(config.support_hold_s, fps),
  )

  static_ready = _forward_all(
    progress >= config.static_progress,
    _frames(config.static_hold_s, fps),
  )
  dynamic_base = (
    (root_height >= config.dynamic_height_ratio)
    & (uprightness >= config.dynamic_uprightness)
    & (horizontal_speed >= config.dynamic_horizontal_speed_height_per_s)
    & no_body_support
    & foot_evidence
    & safe_future
  )
  dynamic_ready = _forward_all(
    dynamic_base,
    _frames(config.dynamic_hold_s, fps),
  )
  support_base = (
    (root_height >= config.support_height_ratio)
    & (uprightness >= config.support_uprightness)
    & no_body_support
    & foot_evidence
    & safe_future
  )
  support_ready = _forward_all(
    support_base,
    _frames(config.support_hold_s, fps),
  )

  low_frames = np.flatnonzero(progress <= config.low_progress)
  min_frames = _frames(config.min_duration_s, fps)
  max_frames = _frames(config.max_duration_s, fps)
  pre_roll = int(round(config.pre_roll_s * fps))
  post_roll = int(round(config.post_roll_s * fps))
  proposals: list[RecoveryReviewCandidate] = []
  cursor = 0

  while cursor < motion.frame_count:
    remaining_low = low_frames[low_frames >= cursor]
    if remaining_low.size == 0:
      break
    first_low = int(remaining_low[0])
    search_start = min(motion.frame_count - 1, first_low + min_frames)
    search_end = min(motion.frame_count, first_low + max_frames)
    terminal = _choose_terminal(
      static_ready,
      dynamic_ready,
      support_ready,
      search_start,
      search_end,
    )
    if terminal is None:
      cursor = _end_of_true_run(progress <= config.low_progress, first_low)
      continue

    terminal_frame, terminal_mode = terminal
    lowest_frame = first_low + int(np.argmin(progress[first_low : terminal_frame + 1]))
    start_frame = max(0, lowest_frame - pre_roll, terminal_frame - max_frames)
    end_frame = min(motion.frame_count, terminal_frame + post_roll + 1)
    if end_frame - start_frame < min_frames:
      cursor = max(terminal_frame + 1, first_low + 1)
      continue

    proposals.append(
      RecoveryReviewCandidate(
        candidate_id=candidate_id(clip_sha256, start_frame, end_frame),
        relative_path=relative_path,
        clip_sha256=clip_sha256,
        recording=recording,
        subject=subject,
        split=split,
        clip_frame_count=motion.frame_count,
        fps=fps,
        auto_start_frame=start_frame,
        auto_end_frame=end_frame,
        auto_support_complete_frame=terminal_frame,
        auto_terminal_mode=terminal_mode,
        auto_initial_posture=cast(
          InitialPosture, classify_initial_posture(motion, lowest_frame)
        ),
        auto_min_progress=float(progress[lowest_frame]),
        auto_terminal_progress=float(progress[terminal_frame]),
      )
    )
    cursor = end_frame

  return tuple(proposals)


def _choose_terminal(
  static_ready: NDArray[np.bool_],
  dynamic_ready: NDArray[np.bool_],
  support_ready: NDArray[np.bool_],
  start: int,
  end: int,
) -> tuple[int, TerminalMode] | None:
  complete: list[tuple[int, TerminalMode]] = []
  static_frames = np.flatnonzero(static_ready[start:end])
  if static_frames.size:
    complete.append((start + int(static_frames[0]), "stationary"))
  dynamic_frames = np.flatnonzero(dynamic_ready[start:end])
  if dynamic_frames.size:
    complete.append((start + int(dynamic_frames[0]), "locomotion"))
  if complete:
    return min(complete, key=lambda item: item[0])
  frames = np.flatnonzero(support_ready[start:end])
  if frames.size:
    return start + int(frames[0]), cast(TerminalMode, "support_only")
  return None


def _frames(duration_s: float, fps: float) -> int:
  return max(1, int(np.ceil(duration_s * fps)))


def _forward_all(mask: NDArray[np.bool_], length: int) -> NDArray[np.bool_]:
  result = np.zeros(mask.shape, dtype=np.bool_)
  if length > mask.size:
    return result
  counts = np.convolve(mask.astype(np.int32), np.ones(length, dtype=np.int32), "valid")
  result[: counts.size] = counts == length
  return result


def _centered_any(mask: NDArray[np.bool_], length: int) -> NDArray[np.bool_]:
  counts = np.convolve(mask.astype(np.int32), np.ones(length, dtype=np.int32), "same")
  return counts > 0


def _end_of_true_run(mask: NDArray[np.bool_], start: int) -> int:
  false_frames = np.flatnonzero(~mask[start:])
  if false_frames.size == 0:
    return mask.size
  return start + int(false_frames[0]) + 1
