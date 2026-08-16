"""Time-domain resampling for canonical motion clips."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mjlab.tasks.velocity.recovery_data.schema import CanonicalMotionClip


def resample_canonical_motion(
  clip: CanonicalMotionClip,
  target_fps: float,
) -> CanonicalMotionClip:
  """Resample translations linearly and rotations with quaternion SLERP."""
  if not np.isfinite(target_fps) or target_fps <= 0.0:
    raise ValueError("target_fps must be finite and positive.")
  if np.isclose(clip.fps, target_fps, rtol=0.0, atol=1e-9):
    return clip

  target_count = max(1, int(np.floor(clip.duration_s * target_fps + 1e-9)) + 1)
  target_times = np.arange(target_count, dtype=np.float64) / target_fps
  source_coordinate = np.minimum(target_times * clip.fps, clip.frame_count - 1)
  before = np.floor(source_coordinate).astype(np.int64)
  after = np.minimum(before + 1, clip.frame_count - 1)
  fraction = source_coordinate - before

  local_positions = _linear_interpolate(
    clip.local_positions_m.astype(np.float64), before, after, fraction
  )
  local_quat = _slerp(
    clip.local_quat_wxyz[before].astype(np.float64),
    clip.local_quat_wxyz[after].astype(np.float64),
    fraction,
  )
  local_rotation = _quaternion_to_matrix(local_quat)
  global_positions, global_rotation = _forward_kinematics(
    local_positions,
    local_rotation,
    clip.skeleton.parents,
  )
  global_quat = _continuous_quaternions(_matrix_to_quaternion(global_rotation))
  return CanonicalMotionClip(
    skeleton=clip.skeleton,
    fps=float(target_fps),
    local_positions_m=local_positions.astype(np.float32),
    local_quat_wxyz=local_quat.astype(np.float32),
    global_positions_m=global_positions.astype(np.float32),
    global_quat_wxyz=global_quat.astype(np.float32),
    source_path=clip.source_path,
  )


def _linear_interpolate(
  value: NDArray[np.float64],
  before: NDArray[np.int64],
  after: NDArray[np.int64],
  fraction: NDArray[np.float64],
) -> NDArray[np.float64]:
  weight = fraction.reshape((-1,) + (1,) * (value.ndim - 1))
  return value[before] * (1.0 - weight) + value[after] * weight


def _slerp(
  first: NDArray[np.float64],
  second: NDArray[np.float64],
  fraction: NDArray[np.float64],
) -> NDArray[np.float64]:
  fraction = fraction.reshape((-1,) + (1,) * (first.ndim - 1))
  first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
  second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
  dot = np.sum(first * second, axis=-1, keepdims=True)
  second = np.where(dot < 0.0, -second, second)
  dot = np.clip(np.abs(dot), 0.0, 1.0)
  angle = np.arccos(dot)
  sine = np.sin(angle)
  safe_sine = np.where(sine > 1e-8, sine, 1.0)
  first_weight = np.sin((1.0 - fraction) * angle) / safe_sine
  second_weight = np.sin(fraction * angle) / safe_sine
  spherical = first_weight * first + second_weight * second
  linear = (1.0 - fraction) * first + fraction * second
  result = np.where(sine > 1e-8, spherical, linear)
  return result / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12)


def _quaternion_to_matrix(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
  w, x, y, z = np.moveaxis(quaternion, -1, 0)
  matrix = np.empty(quaternion.shape[:-1] + (3, 3), dtype=np.float64)
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


def _forward_kinematics(
  local_positions: NDArray[np.float64],
  local_rotations: NDArray[np.float64],
  parents: NDArray[np.integer],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  global_positions = np.empty_like(local_positions)
  global_rotations = np.empty_like(local_rotations)
  global_positions[:, 0] = local_positions[:, 0]
  global_rotations[:, 0] = local_rotations[:, 0]
  for joint in range(1, len(parents)):
    parent = int(parents[joint])
    global_positions[:, joint] = global_positions[:, parent] + np.einsum(
      "tij,tj->ti", global_rotations[:, parent], local_positions[:, joint]
    )
    global_rotations[:, joint] = np.einsum(
      "tij,tjk->tik", global_rotations[:, parent], local_rotations[:, joint]
    )
  return global_positions, global_rotations


def _matrix_to_quaternion(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
  m00 = matrix[..., 0, 0]
  m11 = matrix[..., 1, 1]
  m22 = matrix[..., 2, 2]
  quaternion = np.empty(matrix.shape[:-2] + (4,), dtype=np.float64)
  quaternion[..., 0] = 0.5 * np.sqrt(np.maximum(0.0, 1.0 + m00 + m11 + m22))
  quaternion[..., 1] = 0.5 * np.copysign(
    np.sqrt(np.maximum(0.0, 1.0 + m00 - m11 - m22)),
    matrix[..., 2, 1] - matrix[..., 1, 2],
  )
  quaternion[..., 2] = 0.5 * np.copysign(
    np.sqrt(np.maximum(0.0, 1.0 - m00 + m11 - m22)),
    matrix[..., 0, 2] - matrix[..., 2, 0],
  )
  quaternion[..., 3] = 0.5 * np.copysign(
    np.sqrt(np.maximum(0.0, 1.0 - m00 - m11 + m22)),
    matrix[..., 1, 0] - matrix[..., 0, 1],
  )
  return quaternion / np.maximum(
    np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12
  )


def _continuous_quaternions(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
  result = quaternion.copy()
  for frame in range(1, result.shape[0]):
    flip = np.sum(result[frame - 1] * result[frame], axis=-1) < 0.0
    result[frame, flip] *= -1.0
  return result
