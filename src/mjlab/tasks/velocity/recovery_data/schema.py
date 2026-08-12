"""Versioned schemas shared by recovery data producers and consumers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

RECOVERY_SEMANTIC_SCHEMA_VERSION = "recovery-semantic-v3"
RECOVERY_SEMANTIC_DIM = 93

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Skeleton:
  """A named tree of joints in a right-handed, Z-up coordinate system."""

  joint_names: tuple[str, ...]
  parents: IntArray
  offsets_m: FloatArray

  def __post_init__(self) -> None:
    joint_count = len(self.joint_names)
    if joint_count == 0:
      raise ValueError("A skeleton must contain at least one joint.")
    if self.parents.shape != (joint_count,):
      raise ValueError(
        f"parents must have shape ({joint_count},), got {self.parents.shape}."
      )
    if self.offsets_m.shape != (joint_count, 3):
      raise ValueError(
        f"offsets_m must have shape ({joint_count}, 3), got {self.offsets_m.shape}."
      )
    if int(self.parents[0]) != -1:
      raise ValueError("The root joint must have parent -1.")
    if len(set(self.joint_names)) != joint_count:
      raise ValueError("Skeleton joint names must be unique.")
    for index, parent in enumerate(self.parents[1:], start=1):
      if not 0 <= int(parent) < index:
        raise ValueError(
          f"Joint {index} has invalid parent {int(parent)}; parents must precede children."
        )
    if not np.all(np.isfinite(self.offsets_m)):
      raise ValueError("Skeleton offsets contain NaN or Inf.")

  def index(self, joint_name: str) -> int:
    """Return the index of a named joint with a useful error message."""
    try:
      return self.joint_names.index(joint_name)
    except ValueError as exc:
      raise KeyError(
        f"Joint {joint_name!r} is absent; available joints: {self.joint_names}."
      ) from exc


@dataclass(frozen=True)
class CanonicalMotionClip:
  """A motion clip converted to meters, right-handed coordinates, and Z-up."""

  skeleton: Skeleton
  fps: float
  local_positions_m: FloatArray
  local_quat_wxyz: FloatArray
  global_positions_m: FloatArray
  global_quat_wxyz: FloatArray
  source_path: Path | None = None

  def __post_init__(self) -> None:
    if not np.isfinite(self.fps) or self.fps <= 0.0:
      raise ValueError(f"fps must be finite and positive, got {self.fps}.")
    frame_count = self.local_positions_m.shape[0]
    joint_count = len(self.skeleton.joint_names)
    expected_pos_shape = (frame_count, joint_count, 3)
    expected_quat_shape = (frame_count, joint_count, 4)
    for name, array, shape in (
      ("local_positions_m", self.local_positions_m, expected_pos_shape),
      ("global_positions_m", self.global_positions_m, expected_pos_shape),
      ("local_quat_wxyz", self.local_quat_wxyz, expected_quat_shape),
      ("global_quat_wxyz", self.global_quat_wxyz, expected_quat_shape),
    ):
      if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
      if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf.")
    if frame_count == 0:
      raise ValueError("A motion clip must contain at least one frame.")

  @property
  def frame_count(self) -> int:
    return int(self.local_positions_m.shape[0])

  @property
  def duration_s(self) -> float:
    return (self.frame_count - 1) / self.fps


@dataclass(frozen=True)
class SemanticMotion:
  """A canonical clip projected to the fixed recovery semantic schema."""

  features: FloatArray
  contacts: BoolArray
  progress: FloatArray
  floor_height_m: float
  nominal_height_m: float
  feature_names: tuple[str, ...]

  def __post_init__(self) -> None:
    frame_count = self.features.shape[0]
    if self.features.shape != (frame_count, RECOVERY_SEMANTIC_DIM):
      raise ValueError(
        "features must have shape "
        f"(T, {RECOVERY_SEMANTIC_DIM}), got {self.features.shape}."
      )
    if self.contacts.shape != (frame_count, 8):
      raise ValueError(f"contacts must have shape (T, 8), got {self.contacts.shape}.")
    if self.progress.shape != (frame_count,):
      raise ValueError(f"progress must have shape (T,), got {self.progress.shape}.")
    if len(self.feature_names) != RECOVERY_SEMANTIC_DIM:
      raise ValueError(
        f"Expected {RECOVERY_SEMANTIC_DIM} feature names, "
        f"got {len(self.feature_names)}."
      )
    if not np.all(np.isfinite(self.features)):
      raise ValueError("Semantic features contain NaN or Inf.")
    if not np.all((0.0 <= self.progress) & (self.progress <= 1.0)):
      raise ValueError("Recovery progress must lie in [0, 1].")
    if not np.isfinite(self.floor_height_m):
      raise ValueError("floor_height_m must be finite.")
    if not np.isfinite(self.nominal_height_m) or self.nominal_height_m <= 0.0:
      raise ValueError("nominal_height_m must be finite and positive.")


@dataclass(frozen=True)
class RecoverySegment:
  """A fallen-to-upright interval inside one source motion clip."""

  start_frame: int
  end_frame: int
  initial_posture: str
  min_progress: float
  terminal_progress: float

  def __post_init__(self) -> None:
    if self.start_frame < 0 or self.end_frame <= self.start_frame:
      raise ValueError("A recovery segment must satisfy 0 <= start_frame < end_frame.")
    if self.initial_posture not in {
      "supine",
      "prone",
      "left_side",
      "right_side",
      "other",
    }:
      raise ValueError(f"Unknown recovery posture {self.initial_posture!r}.")
    if not 0.0 <= self.min_progress <= 1.0:
      raise ValueError("min_progress must lie in [0, 1].")
    if not 0.0 <= self.terminal_progress <= 1.0:
      raise ValueError("terminal_progress must lie in [0, 1].")
