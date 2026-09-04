"""Schemas shared by the G1 get-up data compiler and its consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

G1_RECOVERY_SCHEMA_VERSION = "g1-recovery-v1"
G1_RECOVERY_FEATURE_SCHEMA_VERSION = "g1-recovery-features-v1"
G1_PHYSICAL_INIT_SCHEMA_VERSION = "g1-recovery-physical-init-v1"
G1_NOISY_PHYSICAL_INIT_SCHEMA_VERSION = "g1-recovery-noisy-physical-init-v1"
G1_RECOVERY_TARGET_FPS = 50.0
G1_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
G1_JOINT_DIM = len(G1_JOINT_NAMES)
QuaternionOrder = Literal["xyzw", "wxyz"]


@dataclass(frozen=True)
class G1RecoveryClip:
  """One 50 Hz G1 reference clip in the simulator joint convention."""

  root_position: NDArray[np.float32]
  root_quaternion_xyzw: NDArray[np.float32]
  joint_position: NDArray[np.float32]
  root_linear_velocity: NDArray[np.float32]
  root_angular_velocity: NDArray[np.float32]
  joint_velocity: NDArray[np.float32]

  def __post_init__(self) -> None:
    frame_count = self.root_position.shape[0]
    expected = (
      ("root_position", self.root_position, (frame_count, 3)),
      ("root_quaternion_xyzw", self.root_quaternion_xyzw, (frame_count, 4)),
      ("joint_position", self.joint_position, (frame_count, G1_JOINT_DIM)),
      ("root_linear_velocity", self.root_linear_velocity, (frame_count, 3)),
      ("root_angular_velocity", self.root_angular_velocity, (frame_count, 3)),
      ("joint_velocity", self.joint_velocity, (frame_count, G1_JOINT_DIM)),
    )
    if frame_count == 0:
      raise ValueError("A G1 recovery clip must contain at least one frame.")
    for name, values, shape in expected:
      if values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {values.shape}.")
      if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf.")
    norms = np.linalg.norm(self.root_quaternion_xyzw, axis=-1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-4):
      raise ValueError("root_quaternion_xyzw must be normalized.")

  @property
  def valid_length(self) -> int:
    return int(self.root_position.shape[0])
