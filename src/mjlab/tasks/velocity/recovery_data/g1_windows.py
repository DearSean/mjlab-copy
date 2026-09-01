"""Boundary-safe reference windows for the G1 get-up task."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mjlab.tasks.velocity.recovery_data.g1_schema import G1RecoveryClip


def g1_reference_commands(clip: G1RecoveryClip) -> NDArray[np.float32]:
  """Return the RGMT 38-D command for each frame of ``clip``."""
  rotation = _quaternion_to_matrix(clip.root_quaternion_xyzw)
  inverse_rotation = np.swapaxes(rotation, -1, -2)
  gravity = np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
  body_linear = np.einsum("tij,tj->ti", inverse_rotation, clip.root_linear_velocity)
  body_angular = np.einsum("tij,tj->ti", inverse_rotation, clip.root_angular_velocity)
  projected_gravity = np.einsum("tij,j->ti", inverse_rotation, gravity)
  return np.concatenate(
    (body_linear, body_angular, projected_gravity, clip.joint_position), axis=-1
  ).astype(np.float32)


def padded_reference_window(
  commands: NDArray[np.float32], index: int, radius: int = 10
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
  """Return a padded 21-step command window and its clip-boundary mask."""
  if commands.ndim != 2 or commands.shape[1] != 38:
    raise ValueError("commands must have shape (T, 38).")
  if not 0 <= index < len(commands):
    raise IndexError("index must identify a frame in commands.")
  offsets = np.arange(-radius, radius + 1)
  indices = index + offsets
  valid = (0 <= indices) & (indices < len(commands))
  result = np.zeros((len(indices), 38), dtype=np.float32)
  result[valid] = commands[indices[valid]]
  return result, valid


def valid_smp_window_starts(
  valid_length: int, window_size: int = 10
) -> NDArray[np.int64]:
  """Return starts for windows that lie entirely within one clip."""
  if window_size <= 0:
    raise ValueError("window_size must be positive.")
  return np.arange(max(0, valid_length - window_size + 1), dtype=np.int64)


def _quaternion_to_matrix(quaternion_xyzw: NDArray[np.float32]) -> NDArray[np.float32]:
  x, y, z, w = np.moveaxis(quaternion_xyzw, -1, 0)
  matrix = np.empty(quaternion_xyzw.shape[:-1] + (3, 3), dtype=np.float32)
  matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
  matrix[..., 0, 1] = 2.0 * (x * y - z * w)
  matrix[..., 0, 2] = 2.0 * (x * z + y * w)
  matrix[..., 1, 0] = 2.0 * (x * y + z * w)
  matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
  matrix[..., 1, 2] = 2.0 * (y * z - x * w)
  matrix[..., 2, 0] = 2.0 * (x * z - y * w)
  matrix[..., 2, 1] = 2.0 * (y * z + x * w)
  matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
  return matrix
