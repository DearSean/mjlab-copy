"""Minimal, dependency-free BVH reader for the LaFAN motion corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mjlab.tasks.velocity.recovery_data.schema import CanonicalMotionClip, Skeleton

# A +90 degree rotation about source X maps Y-up to Z-up while preserving
# handedness: (x, y, z) -> (x, -z, y).
_Y_UP_TO_Z_UP = np.asarray(
  [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
  dtype=np.float64,
)
_SUPPORTED_CHANNELS = {
  "Xposition",
  "Yposition",
  "Zposition",
  "Xrotation",
  "Yrotation",
  "Zrotation",
}


@dataclass(frozen=True)
class _Hierarchy:
  names: tuple[str, ...]
  parents: NDArray[np.int64]
  offsets_cm: NDArray[np.float64]
  channels: tuple[tuple[str, ...], ...]


class _TokenStream:
  def __init__(self, text: str) -> None:
    self._tokens = re.findall(r"[{}]|[^\s{}]+", text)
    self._index = 0

  def pop(self, expected: str | None = None) -> str:
    if self._index >= len(self._tokens):
      raise ValueError("Unexpected end of BVH hierarchy.")
    token = self._tokens[self._index]
    self._index += 1
    if expected is not None and token != expected:
      raise ValueError(f"Expected BVH token {expected!r}, got {token!r}.")
    return token

  def peek(self) -> str:
    if self._index >= len(self._tokens):
      raise ValueError("Unexpected end of BVH hierarchy.")
    return self._tokens[self._index]

  @property
  def exhausted(self) -> bool:
    return self._index == len(self._tokens)


def load_lafan_bvh(path: str | Path) -> CanonicalMotionClip:
  """Load one LaFAN BVH into the canonical right-handed Z-up representation.

  LaFAN stores root translations in the motion channels and repeats the first
  root translation as the hierarchy root offset. The offset is deliberately
  ignored for the root so it is not added twice.
  """
  source_path = Path(path)
  text = source_path.read_text(encoding="utf-8")
  motion_match = re.search(r"(?m)^\s*MOTION\s*$", text)
  if motion_match is None:
    raise ValueError(f"BVH file {source_path} has no MOTION section.")

  hierarchy = _parse_hierarchy(text[: motion_match.start()])
  fps, channel_data = _parse_motion(
    text[motion_match.end() :],
    sum(len(channels) for channels in hierarchy.channels),
  )
  local_positions_cm, local_rotations = _decode_channels(hierarchy, channel_data)

  conversion = _Y_UP_TO_Z_UP
  local_positions_m = np.einsum("ij,tkj->tki", conversion, local_positions_cm * 0.01)
  local_rotations = np.einsum(
    "ij,tkjl,ml->tkim", conversion, local_rotations, conversion
  )
  offsets_m = np.einsum("ij,kj->ki", conversion, hierarchy.offsets_cm * 0.01)
  offsets_m[0] = 0.0

  global_positions_m, global_rotations = _forward_kinematics(
    local_positions_m,
    local_rotations,
    hierarchy.parents,
  )
  local_quat = _continuous_quaternions(_matrix_to_quaternion(local_rotations))
  global_quat = _continuous_quaternions(_matrix_to_quaternion(global_rotations))

  skeleton = Skeleton(
    joint_names=hierarchy.names,
    parents=hierarchy.parents,
    offsets_m=offsets_m.astype(np.float32),
  )
  return CanonicalMotionClip(
    skeleton=skeleton,
    fps=fps,
    local_positions_m=local_positions_m.astype(np.float32),
    local_quat_wxyz=local_quat.astype(np.float32),
    global_positions_m=global_positions_m.astype(np.float32),
    global_quat_wxyz=global_quat.astype(np.float32),
    source_path=source_path.resolve(),
  )


def _parse_hierarchy(text: str) -> _Hierarchy:
  stream = _TokenStream(text)
  stream.pop("HIERARCHY")
  names: list[str] = []
  parents: list[int] = []
  offsets: list[tuple[float, float, float]] = []
  channel_sets: list[tuple[str, ...]] = []

  def parse_joint(parent: int, keyword: str) -> None:
    stream.pop(keyword)
    name = stream.pop()
    joint_index = len(names)
    names.append(name)
    parents.append(parent)
    stream.pop("{")
    stream.pop("OFFSET")
    offset = (float(stream.pop()), float(stream.pop()), float(stream.pop()))
    offsets.append(offset)
    stream.pop("CHANNELS")
    channel_count = int(stream.pop())
    channels = tuple(stream.pop() for _ in range(channel_count))
    unknown = set(channels) - _SUPPORTED_CHANNELS
    if unknown:
      raise ValueError(f"Joint {name!r} uses unsupported BVH channels: {unknown}.")
    channel_sets.append(channels)

    while stream.peek() != "}":
      token = stream.peek()
      if token == "JOINT":
        parse_joint(joint_index, "JOINT")
      elif token == "End":
        _skip_end_site(stream)
      else:
        raise ValueError(f"Unexpected token {token!r} in joint {name!r}.")
    stream.pop("}")

  parse_joint(-1, "ROOT")
  if not stream.exhausted:
    raise ValueError("Unexpected tokens after BVH root hierarchy.")

  offsets_array = np.asarray(offsets, dtype=np.float64)
  offsets_array[0] = 0.0
  return _Hierarchy(
    names=tuple(names),
    parents=np.asarray(parents, dtype=np.int64),
    offsets_cm=offsets_array,
    channels=tuple(channel_sets),
  )


def _skip_end_site(stream: _TokenStream) -> None:
  stream.pop("End")
  stream.pop("Site")
  stream.pop("{")
  stream.pop("OFFSET")
  for _ in range(3):
    float(stream.pop())
  stream.pop("}")


def _parse_motion(text: str, channel_count: int) -> tuple[float, NDArray[np.float64]]:
  lines = [line.strip() for line in text.splitlines() if line.strip()]
  if len(lines) < 3:
    raise ValueError("BVH MOTION section is incomplete.")
  frames_match = re.fullmatch(r"Frames:\s*(\d+)", lines[0])
  frame_time_match = re.fullmatch(r"Frame\s+Time:\s*([+\-\d.eE]+)", lines[1])
  if frames_match is None or frame_time_match is None:
    raise ValueError("BVH MOTION header must contain Frames and Frame Time.")
  frame_count = int(frames_match.group(1))
  frame_time = float(frame_time_match.group(1))
  if frame_count <= 0 or not np.isfinite(frame_time) or frame_time <= 0.0:
    raise ValueError("BVH frame count and frame time must be positive.")

  values = np.fromstring(" ".join(lines[2:]), sep=" ", dtype=np.float64)
  expected = frame_count * channel_count
  if values.size != expected:
    raise ValueError(
      f"BVH contains {values.size} motion values, expected {expected} "
      f"({frame_count} frames x {channel_count} channels)."
    )
  return 1.0 / frame_time, values.reshape(frame_count, channel_count)


def _decode_channels(
  hierarchy: _Hierarchy,
  data: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  frame_count = data.shape[0]
  joint_count = len(hierarchy.names)
  positions = np.broadcast_to(
    hierarchy.offsets_cm[None, :, :], (frame_count, joint_count, 3)
  ).copy()
  rotations = np.broadcast_to(
    np.eye(3, dtype=np.float64), (frame_count, joint_count, 3, 3)
  ).copy()

  cursor = 0
  axes = {"X": 0, "Y": 1, "Z": 2}
  for joint_index, channels in enumerate(hierarchy.channels):
    joint_data = data[:, cursor : cursor + len(channels)]
    cursor += len(channels)
    has_position_channels = any(channel.endswith("position") for channel in channels)
    if has_position_channels:
      positions[:, joint_index] = 0.0 if joint_index == 0 else positions[:, joint_index]
    for column, channel in enumerate(channels):
      axis = axes[channel[0]]
      if channel.endswith("position"):
        positions[:, joint_index, axis] += joint_data[:, column]
      else:
        axis_rotation = _axis_rotation(axis, np.deg2rad(joint_data[:, column]))
        rotations[:, joint_index] = np.einsum(
          "tij,tjk->tik", rotations[:, joint_index], axis_rotation
        )
  return positions, rotations


def _axis_rotation(axis: int, angle: NDArray[np.float64]) -> NDArray[np.float64]:
  rotation = np.zeros((angle.shape[0], 3, 3), dtype=np.float64)
  cosine = np.cos(angle)
  sine = np.sin(angle)
  rotation[:, axis, axis] = 1.0
  first = (axis + 1) % 3
  second = (axis + 2) % 3
  rotation[:, first, first] = cosine
  rotation[:, second, second] = cosine
  rotation[:, first, second] = -sine
  rotation[:, second, first] = sine
  return rotation


def _forward_kinematics(
  local_positions: NDArray[np.float64],
  local_rotations: NDArray[np.float64],
  parents: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  global_positions = np.empty_like(local_positions)
  global_rotations = np.empty_like(local_rotations)
  global_positions[:, 0] = local_positions[:, 0]
  global_rotations[:, 0] = local_rotations[:, 0]
  for joint_index in range(1, len(parents)):
    parent = int(parents[joint_index])
    global_positions[:, joint_index] = global_positions[:, parent] + np.einsum(
      "tij,tj->ti", global_rotations[:, parent], local_positions[:, joint_index]
    )
    global_rotations[:, joint_index] = np.einsum(
      "tij,tjk->tik", global_rotations[:, parent], local_rotations[:, joint_index]
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
  norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
  return quaternion / np.maximum(norm, 1e-12)


def _continuous_quaternions(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
  result = quaternion.copy()
  for frame in range(1, result.shape[0]):
    flip = np.sum(result[frame - 1] * result[frame], axis=-1) < 0.0
    result[frame, flip] *= -1.0
  return result
