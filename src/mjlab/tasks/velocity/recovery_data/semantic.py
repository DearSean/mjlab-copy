"""Projection from canonical human motion to the recovery semantic schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mjlab.tasks.velocity.recovery_data.schema import (
  RECOVERY_SEMANTIC_DIM,
  CanonicalMotionClip,
  SemanticMotion,
)

LANDMARK_NAMES = (
  "chest",
  "left_elbow",
  "right_elbow",
  "left_hand",
  "right_hand",
  "left_knee",
  "right_knee",
  "left_foot",
  "right_foot",
)
CONTACT_NAMES = (
  "pelvis",
  "chest",
  "left_hand",
  "right_hand",
  "left_knee",
  "right_knee",
  "left_foot",
  "right_foot",
)

_LAFAN_LANDMARK_JOINTS = (
  "Spine2",
  "LeftForeArm",
  "RightForeArm",
  "LeftHand",
  "RightHand",
  "LeftLeg",
  "RightLeg",
  "LeftToe",
  "RightToe",
)
_LAFAN_CONTACT_JOINTS = (
  "Hips",
  "Spine2",
  "LeftHand",
  "RightHand",
  "LeftLeg",
  "RightLeg",
  "LeftToe",
  "RightToe",
)


@dataclass(frozen=True, kw_only=True)
class RecoverySemanticEncoderCfg:
  """Dimensionless thresholds for contact and recovery progress estimation."""

  floor_quantile: float = 0.02
  contact_clearance_height_ratios: tuple[float, ...] = (
    0.12,
    0.14,
    0.06,
    0.06,
    0.08,
    0.08,
    0.035,
    0.035,
  )
  contact_exit_clearance_margin_ratio: float = 0.02
  contact_enter_max_upward_speed_height_per_s: float = 0.20
  contact_exit_max_upward_speed_height_per_s: float = 0.25
  contact_foot_enter_hold_s: float = 0.05
  contact_foot_extended_clearance_height_ratio: float = 0.06
  contact_foot_extended_max_speed_height_per_s: float = 1.0
  fallen_height_ratio: float = 0.25
  standing_height_ratio: float = 0.55
  fallen_uprightness: float = 0.20
  standing_uprightness: float = 0.85

  def __post_init__(self) -> None:
    if not 0.0 <= self.floor_quantile < 0.5:
      raise ValueError("floor_quantile must lie in [0, 0.5).")
    if len(self.contact_clearance_height_ratios) != len(CONTACT_NAMES):
      raise ValueError("A clearance ratio is required for each contact site.")
    if any(value <= 0.0 for value in self.contact_clearance_height_ratios):
      raise ValueError("Contact clearance ratios must be positive.")
    if self.contact_exit_clearance_margin_ratio <= 0.0:
      raise ValueError("Contact clearance hysteresis margin must be positive.")
    if self.contact_enter_max_upward_speed_height_per_s <= 0.0:
      raise ValueError("Contact enter upward-speed threshold must be positive.")
    if (
      self.contact_exit_max_upward_speed_height_per_s
      < self.contact_enter_max_upward_speed_height_per_s
    ):
      raise ValueError("Contact exit speed must not be below its enter speed.")
    if self.contact_foot_enter_hold_s <= 0.0:
      raise ValueError("Foot contact confirmation time must be positive.")
    if self.contact_foot_extended_clearance_height_ratio < max(
      self.contact_clearance_height_ratios[-2:]
    ):
      raise ValueError("Extended foot clearance must include the core clearance.")
    if self.contact_foot_extended_max_speed_height_per_s <= 0.0:
      raise ValueError("Extended foot contact speed threshold must be positive.")
    if not 0.0 <= self.fallen_height_ratio < self.standing_height_ratio:
      raise ValueError("Height progress thresholds are invalid.")
    if not 0.0 <= self.fallen_uprightness < self.standing_uprightness <= 1.0:
      raise ValueError("Uprightness progress thresholds are invalid.")


class RecoverySemanticEncoder:
  """Encode LaFAN clips into the 93D morphology-neutral recovery schema."""

  def __init__(self, cfg: RecoverySemanticEncoderCfg | None = None) -> None:
    self.cfg = cfg or RecoverySemanticEncoderCfg()

  def encode(self, clip: CanonicalMotionClip) -> SemanticMotion:
    indices = self._resolve_indices(clip)
    positions = clip.global_positions_m.astype(np.float64, copy=False)
    nominal_height = _nominal_height(clip)
    landmark_positions = positions[:, indices["landmarks"]]
    contact_positions = positions[:, indices["contacts"]]
    floor_height = float(
      np.quantile(np.min(contact_positions[..., 2], axis=1), self.cfg.floor_quantile)
    )

    heading = _heading_frames(positions, indices)
    torso = _torso_frames(positions, indices, heading)
    dt = 1.0 / clip.fps
    landmark_velocity = _finite_difference(landmark_positions, dt)
    contact_velocity = _finite_difference(contact_positions, dt)
    root_position = positions[:, indices["root"]]
    root_velocity = _finite_difference(root_position, dt)
    root_rotation = _quaternion_to_matrix(
      clip.global_quat_wxyz[:, indices["root"]].astype(np.float64, copy=False)
    )
    root_ang_velocity = _angular_velocity(root_rotation, dt)

    root_height = (root_position[:, 2] - floor_height) / nominal_height
    root_velocity_local = _to_local(heading, root_velocity) / nominal_height
    root_ang_velocity_local = _to_local(heading, root_ang_velocity)

    relative_landmarks = landmark_positions - root_position[:, None, :]
    relative_landmarks_local = (
      np.einsum("tji,tkj->tki", heading, relative_landmarks) / nominal_height
    )
    landmark_velocity_local = (
      np.einsum("tji,tkj->tki", heading, landmark_velocity) / nominal_height
    )

    torso_local = np.einsum("tji,tjk->tik", heading, torso)
    torso_rotation_6d = np.concatenate(
      [torso_local[..., :, 0], torso_local[..., :, 1]], axis=-1
    )

    clearance = (contact_positions[..., 2] - floor_height) / nominal_height
    contact_vertical_speed = contact_velocity[..., 2] / nominal_height
    contact_total_speed = np.linalg.norm(contact_velocity, axis=-1) / nominal_height
    clearance_threshold = np.asarray(
      self.cfg.contact_clearance_height_ratios, dtype=np.float64
    )
    extended_foot_enter = np.zeros(clearance.shape, dtype=np.bool_)
    extended_foot_enter[:, -2:] = (
      clearance[:, -2:] <= self.cfg.contact_foot_extended_clearance_height_ratio
    ) & (
      contact_total_speed[:, -2:]
      <= self.cfg.contact_foot_extended_max_speed_height_per_s
    )
    exit_clearance = clearance_threshold + self.cfg.contact_exit_clearance_margin_ratio
    exit_clearance[-2:] = (
      self.cfg.contact_foot_extended_clearance_height_ratio
      + self.cfg.contact_exit_clearance_margin_ratio
    )
    contacts = _contact_hysteresis(
      clearance,
      contact_vertical_speed,
      enter_clearance=clearance_threshold,
      additional_enter_condition=extended_foot_enter,
      exit_clearance=exit_clearance,
      enter_max_upward_speed=(self.cfg.contact_enter_max_upward_speed_height_per_s),
      exit_max_upward_speed=self.cfg.contact_exit_max_upward_speed_height_per_s,
      enter_hold_frames=np.asarray(
        [1] * (len(CONTACT_NAMES) - 2)
        + [max(1, int(np.ceil(self.cfg.contact_foot_enter_hold_s * clip.fps)))] * 2,
        dtype=np.int64,
      ),
    )

    uprightness = np.clip(torso[..., 2, 2], 0.0, 1.0)
    height_score = _smoothstep(
      root_height,
      self.cfg.fallen_height_ratio,
      self.cfg.standing_height_ratio,
    )
    upright_score = _smoothstep(
      uprightness,
      self.cfg.fallen_uprightness,
      self.cfg.standing_uprightness,
    )
    progress = np.clip(height_score * upright_score, 0.0, 1.0)

    frame_count = clip.frame_count
    features = np.concatenate(
      [
        root_height[:, None],
        torso_rotation_6d,
        root_velocity_local,
        root_ang_velocity_local,
        relative_landmarks_local.reshape(frame_count, -1),
        landmark_velocity_local.reshape(frame_count, -1),
        clearance,
        contacts.astype(np.float64),
        np.ones((frame_count, len(LANDMARK_NAMES)), dtype=np.float64),
        progress[:, None],
      ],
      axis=-1,
    )
    if features.shape[1] != RECOVERY_SEMANTIC_DIM:
      raise RuntimeError(
        f"Semantic encoder produced {features.shape[1]} features, "
        f"expected {RECOVERY_SEMANTIC_DIM}."
      )
    return SemanticMotion(
      features=features.astype(np.float32),
      contacts=contacts,
      progress=progress.astype(np.float32),
      floor_height_m=floor_height,
      nominal_height_m=nominal_height,
      feature_names=recovery_semantic_feature_names(),
    )

  @staticmethod
  def _resolve_indices(clip: CanonicalMotionClip) -> dict[str, int | NDArray[np.int64]]:
    skeleton = clip.skeleton
    return {
      "root": skeleton.index("Hips"),
      "left_hip": skeleton.index("LeftUpLeg"),
      "right_hip": skeleton.index("RightUpLeg"),
      "left_shoulder": skeleton.index("LeftShoulder"),
      "right_shoulder": skeleton.index("RightShoulder"),
      "chest": skeleton.index("Spine2"),
      "head": skeleton.index("Head"),
      "left_toe": skeleton.index("LeftToe"),
      "right_toe": skeleton.index("RightToe"),
      "landmarks": np.asarray(
        [skeleton.index(name) for name in _LAFAN_LANDMARK_JOINTS],
        dtype=np.int64,
      ),
      "contacts": np.asarray(
        [skeleton.index(name) for name in _LAFAN_CONTACT_JOINTS],
        dtype=np.int64,
      ),
    }


def recovery_semantic_feature_names() -> tuple[str, ...]:
  names: list[str] = ["root_height"]
  names.extend(f"torso_rotation_6d_{index}" for index in range(6))
  names.extend(f"root_linear_velocity_{axis}" for axis in "xyz")
  names.extend(f"root_angular_velocity_{axis}" for axis in "xyz")
  for landmark in LANDMARK_NAMES:
    names.extend(f"{landmark}_position_{axis}" for axis in "xyz")
  for landmark in LANDMARK_NAMES:
    names.extend(f"{landmark}_velocity_{axis}" for axis in "xyz")
  names.extend(f"{contact}_clearance" for contact in CONTACT_NAMES)
  names.extend(f"{contact}_contact" for contact in CONTACT_NAMES)
  names.extend(f"{landmark}_mask" for landmark in LANDMARK_NAMES)
  names.append("recovery_progress")
  if len(names) != RECOVERY_SEMANTIC_DIM:
    raise RuntimeError(
      f"Feature-name schema has {len(names)} entries, expected {RECOVERY_SEMANTIC_DIM}."
    )
  return tuple(names)


def classify_initial_posture(
  clip: CanonicalMotionClip,
  frame: int,
) -> str:
  """Classify a low posture using the body's anterior and right directions."""
  encoder = RecoverySemanticEncoder()
  indices = encoder._resolve_indices(clip)
  positions = clip.global_positions_m.astype(np.float64, copy=False)
  heading = _heading_frames(positions, indices)
  torso = _torso_frames(positions, indices, heading)
  anterior_z = float(torso[frame, 2, 1])
  right_z = float(torso[frame, 2, 0])
  if max(abs(anterior_z), abs(right_z)) < 0.5:
    return "other"
  if abs(anterior_z) >= abs(right_z):
    return "supine" if anterior_z > 0.0 else "prone"
  return "left_side" if right_z > 0.0 else "right_side"


def _nominal_height(clip: CanonicalMotionClip) -> float:
  skeleton = clip.skeleton

  def path_length(joint_name: str) -> float:
    index = skeleton.index(joint_name)
    total = 0.0
    while index > 0:
      total += float(np.linalg.norm(skeleton.offsets_m[index]))
      index = int(skeleton.parents[index])
    return total

  leg_length = 0.5 * (path_length("LeftToe") + path_length("RightToe"))
  torso_length = path_length("Head")
  height = leg_length + torso_length
  if not np.isfinite(height) or height <= 1e-6:
    raise ValueError(f"Could not derive a valid nominal height, got {height}.")
  return height


def _heading_frames(
  positions: NDArray[np.float64],
  indices: dict[str, int | NDArray[np.int64]],
) -> NDArray[np.float64]:
  left_hip = int(indices["left_hip"])
  right_hip = int(indices["right_hip"])
  left_shoulder = int(indices["left_shoulder"])
  right_shoulder = int(indices["right_shoulder"])
  right = positions[:, right_hip] - positions[:, left_hip]
  shoulder_right = positions[:, right_shoulder] - positions[:, left_shoulder]
  right[:, 2] = 0.0
  shoulder_right[:, 2] = 0.0
  right = _normalize_with_fallback(right, shoulder_right, np.asarray([1.0, 0.0, 0.0]))
  up = np.broadcast_to(np.asarray([0.0, 0.0, 1.0]), right.shape)
  forward = _normalize(np.cross(up, right))
  right = _normalize(np.cross(forward, up))
  return np.stack([right, forward, up], axis=-1)


def _torso_frames(
  positions: NDArray[np.float64],
  indices: dict[str, int | NDArray[np.int64]],
  heading: NDArray[np.float64],
) -> NDArray[np.float64]:
  root = int(indices["root"])
  chest = int(indices["chest"])
  left_shoulder = int(indices["left_shoulder"])
  right_shoulder = int(indices["right_shoulder"])
  up = _normalize_with_fallback(
    positions[:, chest] - positions[:, root],
    heading[..., 2],
    np.asarray([0.0, 0.0, 1.0]),
  )
  right_raw = positions[:, right_shoulder] - positions[:, left_shoulder]
  right_raw -= np.sum(right_raw * up, axis=-1, keepdims=True) * up
  right = _normalize_with_fallback(
    right_raw, heading[..., 0], np.asarray([1.0, 0.0, 0.0])
  )
  forward = _normalize_with_fallback(
    np.cross(up, right),
    heading[..., 1],
    np.asarray([0.0, 1.0, 0.0]),
  )
  right = _normalize(np.cross(forward, up))
  return np.stack([right, forward, up], axis=-1)


def _normalize(vector: NDArray[np.float64]) -> NDArray[np.float64]:
  return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-12)


def _normalize_with_fallback(
  vector: NDArray[np.float64],
  fallback: NDArray[np.float64],
  default: NDArray[np.float64],
) -> NDArray[np.float64]:
  result = vector.copy()
  invalid = np.linalg.norm(result, axis=-1) <= 1e-8
  if fallback.ndim == 1:
    result[invalid] = fallback
  else:
    result[invalid] = fallback[invalid]
  still_invalid = np.linalg.norm(result, axis=-1) <= 1e-8
  result[still_invalid] = default
  return _normalize(result)


def _finite_difference(value: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
  if value.shape[0] == 1:
    return np.zeros_like(value)
  return np.gradient(value, dt, axis=0, edge_order=1)


def _to_local(
  frame: NDArray[np.float64],
  vector: NDArray[np.float64],
) -> NDArray[np.float64]:
  return np.einsum("tji,tj->ti", frame, vector)


def _quaternion_to_matrix(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
  quaternion = quaternion / np.maximum(
    np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12
  )
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


def _angular_velocity(
  rotation: NDArray[np.float64],
  dt: float,
) -> NDArray[np.float64]:
  frame_count = rotation.shape[0]
  if frame_count == 1:
    return np.zeros((1, 3), dtype=np.float64)
  result = np.empty((frame_count, 3), dtype=np.float64)
  for frame in range(frame_count):
    before = max(frame - 1, 0)
    after = min(frame + 1, frame_count - 1)
    elapsed = (after - before) * dt
    delta = rotation[after] @ rotation[before].T
    trace = float(np.trace(delta))
    angle = float(np.arccos(np.clip(0.5 * (trace - 1.0), -1.0, 1.0)))
    vee = np.asarray(
      [
        delta[2, 1] - delta[1, 2],
        delta[0, 2] - delta[2, 0],
        delta[1, 0] - delta[0, 1],
      ]
    )
    coefficient = 0.5 if angle < 1e-7 else angle / (2.0 * np.sin(angle))
    result[frame] = coefficient * vee / elapsed
  return result


def _smoothstep(
  value: NDArray[np.float64], low: float, high: float
) -> NDArray[np.float64]:
  ratio = np.clip((value - low) / (high - low), 0.0, 1.0)
  return ratio * ratio * (3.0 - 2.0 * ratio)


def _contact_hysteresis(
  clearance: NDArray[np.float64],
  vertical_speed: NDArray[np.float64],
  *,
  enter_clearance: NDArray[np.float64],
  additional_enter_condition: NDArray[np.bool_],
  exit_clearance: NDArray[np.float64],
  enter_max_upward_speed: float,
  exit_max_upward_speed: float,
  enter_hold_frames: NDArray[np.int64],
) -> NDArray[np.bool_]:
  """Detect geometric contact without rejecting sliding or rolling support.

  A site enters contact when it is close to the floor and not moving upward
  faster than the enter threshold. Once active, clearance and upward-speed
  margins prevent frame-to-frame flicker. Horizontal speed never rejects a site
  inside its core clearance. It is used only in the wider foot-clearance band,
  where it separates a nearby pivoting support from a fast airborne swing.
  """
  if clearance.shape != vertical_speed.shape:
    raise ValueError("Contact clearance and vertical speed shapes must match.")
  if additional_enter_condition.shape != clearance.shape:
    raise ValueError("Additional contact-enter condition has an invalid shape.")
  if enter_hold_frames.shape != (clearance.shape[1],):
    raise ValueError("A contact confirmation length is required for every site.")
  if np.any(enter_hold_frames <= 0):
    raise ValueError("Contact confirmation lengths must be positive.")
  contacts = np.zeros(clearance.shape, dtype=np.bool_)
  active = np.zeros(clearance.shape[1], dtype=np.bool_)
  enter_count = np.zeros(clearance.shape[1], dtype=np.int64)
  for frame in range(clearance.shape[0]):
    entering = (
      (clearance[frame] <= enter_clearance) | additional_enter_condition[frame]
    ) & (vertical_speed[frame] <= enter_max_upward_speed)
    remaining = (clearance[frame] <= exit_clearance) & (
      vertical_speed[frame] <= exit_max_upward_speed
    )
    enter_count = np.where(entering, enter_count + 1, 0)
    confirmed = enter_count >= enter_hold_frames
    active = np.where(active, remaining, confirmed)
    enter_count = np.where(active, 0, enter_count)
    contacts[frame] = active
  return contacts
