"""Build a simulator-validated reset-state bank without changing SMP clips."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from mjlab.asset_zoo.robots import G1_VELOCITY_COLLISION, get_g1_robot_cfg
from mjlab.entity import Entity

from .g1_schema import G1_JOINT_NAMES, G1_PHYSICAL_INIT_SCHEMA_VERSION

_SUPPORT_PATTERN = re.compile(
  r"^(left|right)_(?:foot[1-7]|shin|hand|wrist|elbow_yaw)_collision$"
)


@dataclass(frozen=True, kw_only=True)
class G1PhysicalInitCfg:
  """Thresholds for bounded contact projection and pose-hold settling."""

  dataset_dir: Path = Path("artifacts/g1_recovery")
  output_file: Path = Path("artifacts/g1_recovery/physical_init.npz")
  report_file: Path = Path("artifacts/g1_recovery/physical_init_report.json")
  split: str = "train"
  timestep_s: float = 0.005
  settle_duration_s: float = 0.4
  support_window_s: float = 0.2
  max_contact_gap_m: float = 0.05
  contact_target_depth_m: float = 0.001
  contact_tolerance_m: float = 0.004
  max_root_projection_m: float = 0.04
  max_joint_projection_rad: float = 0.2
  projection_iterations: int = 12
  projection_damping: float = 1e-3
  max_self_penetration_m: float = 0.01
  max_ground_penetration_m: float = 0.008
  max_settle_translation_m: float = 0.12
  max_settle_rotation_rad: float = 0.45
  max_contact_force_n: float = 5000.0
  min_support_fraction: float = 0.8

  def __post_init__(self) -> None:
    positive = {
      "timestep_s": self.timestep_s,
      "settle_duration_s": self.settle_duration_s,
      "support_window_s": self.support_window_s,
      "max_contact_gap_m": self.max_contact_gap_m,
      "contact_tolerance_m": self.contact_tolerance_m,
      "max_root_projection_m": self.max_root_projection_m,
      "max_joint_projection_rad": self.max_joint_projection_rad,
      "projection_damping": self.projection_damping,
      "max_self_penetration_m": self.max_self_penetration_m,
      "max_ground_penetration_m": self.max_ground_penetration_m,
      "max_settle_translation_m": self.max_settle_translation_m,
      "max_settle_rotation_rad": self.max_settle_rotation_rad,
      "max_contact_force_n": self.max_contact_force_n,
    }
    if any(value <= 0.0 for value in positive.values()):
      raise ValueError("Physical initialization thresholds must be positive.")
    if not 0.0 < self.support_window_s <= self.settle_duration_s:
      raise ValueError("support_window_s must be inside the settling interval.")
    if not 0.0 <= self.contact_target_depth_m <= self.max_ground_penetration_m:
      raise ValueError("contact_target_depth_m must be a shallow penetration.")
    if self.projection_iterations <= 0:
      raise ValueError("projection_iterations must be positive.")
    if not 0.0 <= self.min_support_fraction <= 1.0:
      raise ValueError("min_support_fraction must be between zero and one.")


@dataclass(frozen=True)
class _FrameResult:
  accepted: bool
  reason: str
  qpos: NDArray[np.float64]
  category: str
  root_projection_m: float
  joint_projection_max_rad: float
  settle_translation_m: float
  settle_rotation_rad: float
  support_fraction: float
  max_contact_force_n: float
  max_self_penetration_m: float
  max_ground_penetration_m: float


def build_g1_physical_init(cfg: G1PhysicalInitCfg) -> dict[str, Any]:
  """Project and settle every selected frame, then save accepted reset states."""
  dataset_dir = cfg.dataset_dir.resolve()
  manifest_file = dataset_dir / "manifest.json"
  if not manifest_file.is_file():
    raise FileNotFoundError(f"G1 recovery manifest does not exist: {manifest_file}")
  manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
  clips = [clip for clip in manifest["clips"] if clip["split"] == cfg.split]
  if not clips:
    raise ValueError(f"G1 recovery manifest has no clips in split {cfg.split!r}.")

  model, ground_id = _make_model(cfg)
  data = mujoco.MjData(model)
  joint_ids = _joint_ids(model)
  joint_qpos = model.jnt_qposadr[joint_ids]
  joint_dof = model.jnt_dofadr[joint_ids]
  collision_ids = np.asarray(
    [index for index in range(model.ngeom) if _is_collision_geom(model, index)]
  )
  support_ids = np.asarray(
    [index for index in collision_ids if _is_support_geom(model, index)]
  )

  accepted: dict[str, list[Any]] = {
    "clip_id": [],
    "frame": [],
    "root_position": [],
    "root_quaternion_xyzw": [],
    "joint_position": [],
    "category": [],
    "root_projection_m": [],
    "joint_projection_max_rad": [],
    "settle_translation_m": [],
    "settle_rotation_rad": [],
    "support_fraction": [],
    "max_contact_force_n": [],
    "max_self_penetration_m": [],
    "max_ground_penetration_m": [],
  }
  reasons: Counter[str] = Counter()
  category_counts: Counter[str] = Counter()
  clip_reports: list[dict[str, Any]] = []
  for clip in clips:
    arrays = np.load(dataset_dir / clip["file"], allow_pickle=False)
    clip_accepted = 0
    clip_reasons: Counter[str] = Counter()
    for frame in range(len(arrays["joint_position"])):
      qpos = model.qpos0.copy()
      qpos[:3] = arrays["root_position"][frame]
      qpos[3:7] = arrays["root_quaternion_xyzw"][frame, (3, 0, 1, 2)]
      qpos[joint_qpos] = arrays["joint_position"][frame]
      result = _validate_frame(
        model,
        data,
        qpos,
        joint_qpos,
        joint_dof,
        collision_ids,
        support_ids,
        ground_id,
        cfg,
      )
      reasons[result.reason] += 1
      clip_reasons[result.reason] += 1
      if not result.accepted:
        continue
      clip_accepted += 1
      category_counts[result.category] += 1
      accepted["clip_id"].append(clip["clip_id"])
      accepted["frame"].append(frame)
      root_position = result.qpos[:3].copy()
      # Source clips retain their original global XY trajectory, but reset
      # states are centered at each environment origin. Preserve only the
      # simulator settling correction in XY and the absolute ground-relative Z.
      root_position[:2] -= arrays["root_position"][frame, :2]
      accepted["root_position"].append(root_position)
      accepted["root_quaternion_xyzw"].append(result.qpos[3:7][[1, 2, 3, 0]].copy())
      accepted["joint_position"].append(result.qpos[joint_qpos].copy())
      for name in (
        "category",
        "root_projection_m",
        "joint_projection_max_rad",
        "settle_translation_m",
        "settle_rotation_rad",
        "support_fraction",
        "max_contact_force_n",
        "max_self_penetration_m",
        "max_ground_penetration_m",
      ):
        accepted[name].append(getattr(result, name))
    clip_reports.append(
      {
        "clip_id": clip["clip_id"],
        "frame_count": int(clip["valid_length"]),
        "accepted_frame_count": clip_accepted,
        "reasons": dict(sorted(clip_reasons.items())),
      }
    )

  if not accepted["frame"]:
    raise RuntimeError("Physical initialization audit rejected every frame.")
  cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    cfg.output_file,
    schema_version=np.asarray(G1_PHYSICAL_INIT_SCHEMA_VERSION),
    source_manifest=np.asarray(str(manifest_file)),
    clip_id=np.asarray(accepted["clip_id"], dtype=np.str_),
    frame=np.asarray(accepted["frame"], dtype=np.int32),
    root_position=np.asarray(accepted["root_position"], dtype=np.float32),
    root_quaternion_xyzw=np.asarray(accepted["root_quaternion_xyzw"], dtype=np.float32),
    joint_position=np.asarray(accepted["joint_position"], dtype=np.float32),
    root_linear_velocity=np.zeros((len(accepted["frame"]), 3), dtype=np.float32),
    root_angular_velocity=np.zeros((len(accepted["frame"]), 3), dtype=np.float32),
    joint_velocity=np.zeros(
      (len(accepted["frame"]), len(G1_JOINT_NAMES)), dtype=np.float32
    ),
    category=np.asarray(accepted["category"], dtype=np.str_),
    root_projection_m=np.asarray(accepted["root_projection_m"], dtype=np.float32),
    joint_projection_max_rad=np.asarray(
      accepted["joint_projection_max_rad"], dtype=np.float32
    ),
    settle_translation_m=np.asarray(accepted["settle_translation_m"], dtype=np.float32),
    settle_rotation_rad=np.asarray(accepted["settle_rotation_rad"], dtype=np.float32),
    support_fraction=np.asarray(accepted["support_fraction"], dtype=np.float32),
    max_contact_force_n=np.asarray(accepted["max_contact_force_n"], dtype=np.float32),
    max_self_penetration_m=np.asarray(
      accepted["max_self_penetration_m"], dtype=np.float32
    ),
    max_ground_penetration_m=np.asarray(
      accepted["max_ground_penetration_m"], dtype=np.float32
    ),
  )
  report = {
    "schema_version": G1_PHYSICAL_INIT_SCHEMA_VERSION,
    "dataset_manifest": str(manifest_file),
    "output_file": str(cfg.output_file.resolve()),
    "split": cfg.split,
    "source_clip_count": len(clips),
    "source_frame_count": sum(int(clip["valid_length"]) for clip in clips),
    "accepted_frame_count": len(accepted["frame"]),
    "acceptance_rate": len(accepted["frame"])
    / sum(int(clip["valid_length"]) for clip in clips),
    "categories": dict(sorted(category_counts.items())),
    "reasons": dict(sorted(reasons.items())),
    "thresholds": _jsonable_cfg(cfg),
    "clips": clip_reports,
  }
  cfg.report_file.parent.mkdir(parents=True, exist_ok=True)
  cfg.report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  return report


def _make_model(cfg: G1PhysicalInitCfg) -> tuple[mujoco.MjModel, int]:
  robot_cfg = get_g1_robot_cfg()
  robot_cfg.collisions = (G1_VELOCITY_COLLISION,)
  robot = Entity(robot_cfg)
  spec = robot.spec
  spec.worldbody.add_geom(
    name="g1_physical_init_ground",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[5.0, 5.0, 0.1],
    condim=3,
    friction=[0.8, 0.005, 0.0001],
    margin=cfg.max_contact_gap_m,
  )
  spec.option.timestep = cfg.timestep_s
  model = spec.compile()
  ground_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "g1_physical_init_ground"
  )
  if ground_id < 0:
    raise RuntimeError("Failed to add the physical initialization ground plane.")
  return model, ground_id


def _validate_frame(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  original_qpos: NDArray[np.float64],
  joint_qpos: NDArray[np.int32],
  joint_dof: NDArray[np.int32],
  collision_ids: NDArray[np.int64],
  support_ids: NDArray[np.int64],
  ground_id: int,
  cfg: G1PhysicalInitCfg,
) -> _FrameResult:
  data.qpos[:] = original_qpos
  data.qvel[:] = 0.0
  data.ctrl[:] = 0.0
  mujoco.mj_forward(model, data)
  _, initial_self = _penetrations(model, data, ground_id)
  if initial_self > cfg.max_self_penetration_m:
    return _rejected("self_penetration", original_qpos, initial_self=initial_self)

  projected, root_delta, joint_delta, projection_reason = _project_contacts(
    model,
    data,
    original_qpos,
    joint_qpos,
    joint_dof,
    collision_ids,
    support_ids,
    ground_id,
    cfg,
  )
  if projection_reason is not None:
    return _rejected(projection_reason, original_qpos, initial_self=initial_self)

  model.geom_margin[ground_id] = 0.0
  try:
    result = _settle_pose(
      model,
      data,
      projected,
      joint_qpos,
      ground_id,
      cfg,
      root_delta,
      joint_delta,
    )
  finally:
    model.geom_margin[ground_id] = cfg.max_contact_gap_m
  return result


def _project_contacts(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  original_qpos: NDArray[np.float64],
  joint_qpos: NDArray[np.int32],
  joint_dof: NDArray[np.int32],
  collision_ids: NDArray[np.int64],
  support_ids: NDArray[np.int64],
  ground_id: int,
  cfg: G1PhysicalInitCfg,
) -> tuple[NDArray[np.float64], float, float, str | None]:
  qpos = original_qpos.copy()
  target_geoms = _near_ground_groups(model, data, ground_id, support_ids)
  if not target_geoms:
    target_geoms = _near_ground_groups(model, data, ground_id, collision_ids)
  if not target_geoms:
    return qpos, 0.0, 0.0, "no_near_ground_support"

  variables = np.concatenate((np.asarray([2], dtype=np.int32), joint_dof))
  original_joint = original_qpos[joint_qpos].copy()
  for _ in range(cfg.projection_iterations):
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    contacts = _ground_contact_map(model, data, ground_id)
    selected = [contacts[geom_id] for geom_id in target_geoms if geom_id in contacts]
    if len(selected) != len(target_geoms):
      return qpos, 0.0, 0.0, "projection_lost_support"
    error = np.asarray(
      [contact.dist + cfg.contact_target_depth_m for contact in selected],
      dtype=np.float64,
    )
    if float(np.max(np.abs(error))) <= cfg.contact_tolerance_m:
      break
    jacobian = np.zeros((len(selected), len(variables)), dtype=np.float64)
    full_jacobian = np.zeros((3, model.nv), dtype=np.float64)
    for row, contact in enumerate(selected):
      geom_id = contact.geom2 if contact.geom1 == ground_id else contact.geom1
      body_id = int(model.geom_bodyid[geom_id])
      mujoco.mj_jac(model, data, full_jacobian, None, contact.pos, body_id)
      jacobian[row] = full_jacobian[2, variables]
    system = jacobian @ jacobian.T
    system.flat[:: len(selected) + 1] += cfg.projection_damping
    delta = -jacobian.T @ np.linalg.solve(system, error)
    delta = np.clip(delta, -0.04, 0.04)
    qpos[2] = np.clip(
      qpos[2] + delta[0],
      original_qpos[2] - cfg.max_root_projection_m,
      original_qpos[2] + cfg.max_root_projection_m,
    )
    candidate_joint = qpos[joint_qpos] + delta[1:]
    candidate_joint = np.clip(
      candidate_joint,
      original_joint - cfg.max_joint_projection_rad,
      original_joint + cfg.max_joint_projection_rad,
    )
    joint_range = model.jnt_range[_joint_ids(model)]
    qpos[joint_qpos] = np.clip(candidate_joint, joint_range[:, 0], joint_range[:, 1])

  data.qpos[:] = qpos
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)
  contacts = _ground_contact_map(model, data, ground_id)
  if any(geom_id not in contacts for geom_id in target_geoms):
    return qpos, 0.0, 0.0, "projection_lost_support"
  max_error = max(
    abs(float(contacts[geom_id].dist) + cfg.contact_target_depth_m)
    for geom_id in target_geoms
  )
  ground_penetration, self_penetration = _penetrations(model, data, ground_id)
  if max_error > cfg.contact_tolerance_m:
    return qpos, 0.0, 0.0, "projection_not_converged"
  if self_penetration > cfg.max_self_penetration_m:
    return qpos, 0.0, 0.0, "projection_self_penetration"
  if ground_penetration > cfg.max_ground_penetration_m:
    return qpos, 0.0, 0.0, "projection_ground_penetration"
  root_delta = abs(float(qpos[2] - original_qpos[2]))
  joint_delta = float(np.max(np.abs(qpos[joint_qpos] - original_joint)))
  return qpos, root_delta, joint_delta, None


def _settle_pose(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  projected_qpos: NDArray[np.float64],
  joint_qpos: NDArray[np.int32],
  ground_id: int,
  cfg: G1PhysicalInitCfg,
  root_projection_m: float,
  joint_projection_max_rad: float,
) -> _FrameResult:
  data.qpos[:] = projected_qpos
  data.qvel[:] = 0.0
  for actuator_id in range(model.nu):
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    data.ctrl[actuator_id] = projected_qpos[int(model.jnt_qposadr[joint_id])]
  mujoco.mj_forward(model, data)
  start_position = projected_qpos[:3].copy()
  start_quaternion = projected_qpos[3:7].copy()
  steps = max(1, round(cfg.settle_duration_s / cfg.timestep_s))
  support_steps = max(1, round(cfg.support_window_s / cfg.timestep_s))
  support_count = 0
  max_force = 0.0
  max_self = 0.0
  max_ground = 0.0
  for step in range(steps):
    mujoco.mj_step(model, data)
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
      return _rejected("nonfinite_settle", projected_qpos)
    ground_penetration, self_penetration = _penetrations(model, data, ground_id)
    max_ground = max(max_ground, ground_penetration)
    max_self = max(max_self, self_penetration)
    max_force = max(max_force, _max_contact_force(model, data))
    if step >= steps - support_steps and _has_ground_support(model, data, ground_id):
      support_count += 1
  support_fraction = support_count / support_steps
  translation = float(np.linalg.norm(data.qpos[:3] - start_position))
  rotation = _quaternion_distance(start_quaternion, data.qpos[3:7])
  reason = "accepted"
  if max_self > cfg.max_self_penetration_m:
    reason = "settle_self_penetration"
  elif max_ground > cfg.max_ground_penetration_m:
    reason = "settle_ground_penetration"
  elif max_force > cfg.max_contact_force_n:
    reason = "settle_contact_force"
  elif translation > cfg.max_settle_translation_m:
    reason = "settle_translation"
  elif rotation > cfg.max_settle_rotation_rad:
    reason = "settle_rotation"
  elif support_fraction < cfg.min_support_fraction:
    reason = "settle_lost_support"
  if reason != "accepted":
    return _rejected(
      reason,
      projected_qpos,
      initial_self=max_self,
      ground=max_ground,
      force=max_force,
      translation=translation,
      rotation=rotation,
      support=support_fraction,
    )

  settled_qpos = data.qpos.copy()
  joint_range = model.jnt_range[_joint_ids(model)]
  settled_qpos[joint_qpos] = np.clip(
    settled_qpos[joint_qpos], joint_range[:, 0], joint_range[:, 1]
  )
  mujoco.mju_normalize4(settled_qpos[3:7])
  data.qpos[:] = settled_qpos
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)
  ground_penetration, self_penetration = _penetrations(model, data, ground_id)
  if self_penetration > cfg.max_self_penetration_m:
    return _rejected(
      "clamped_self_penetration", settled_qpos, initial_self=self_penetration
    )
  if ground_penetration > cfg.max_ground_penetration_m:
    return _rejected(
      "clamped_ground_penetration", settled_qpos, ground=ground_penetration
    )
  if not _has_ground_support(model, data, ground_id):
    return _rejected("clamped_lost_support", settled_qpos)
  category = (
    "raw_stable"
    if root_projection_m < 1e-4 and joint_projection_max_rad < 1e-4
    else "projected_stable"
  )
  return _FrameResult(
    accepted=True,
    reason="accepted",
    qpos=settled_qpos,
    category=category,
    root_projection_m=root_projection_m,
    joint_projection_max_rad=joint_projection_max_rad,
    settle_translation_m=translation,
    settle_rotation_rad=rotation,
    support_fraction=support_fraction,
    max_contact_force_n=max_force,
    max_self_penetration_m=max_self,
    max_ground_penetration_m=max_ground,
  )


def _near_ground_groups(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  ground_id: int,
  allowed_geoms: NDArray[np.int64],
) -> list[int]:
  allowed = set(int(value) for value in allowed_geoms)
  grouped: dict[str, tuple[float, int]] = {}
  for geom_id, contact in _ground_contact_map(model, data, ground_id).items():
    if geom_id not in allowed:
      continue
    group = _support_group(model.geom(geom_id).name)
    current = grouped.get(group)
    if current is None or contact.dist < current[0]:
      grouped[group] = (float(contact.dist), geom_id)
  return [value[1] for value in grouped.values()]


def _ground_contact_map(
  model: mujoco.MjModel, data: mujoco.MjData, ground_id: int
) -> dict[int, mujoco.MjContact]:
  result: dict[int, mujoco.MjContact] = {}
  for contact in data.contact[: data.ncon]:
    if contact.geom1 != ground_id and contact.geom2 != ground_id:
      continue
    geom_id = int(contact.geom2 if contact.geom1 == ground_id else contact.geom1)
    previous = result.get(geom_id)
    if previous is None or contact.dist < previous.dist:
      result[geom_id] = contact
  return result


def _penetrations(
  model: mujoco.MjModel, data: mujoco.MjData, ground_id: int
) -> tuple[float, float]:
  ground = 0.0
  self_penetration = 0.0
  for contact in data.contact[: data.ncon]:
    penetration = max(0.0, -float(contact.dist))
    if contact.geom1 == ground_id or contact.geom2 == ground_id:
      ground = max(ground, penetration)
      continue
    body1 = int(model.geom_bodyid[contact.geom1])
    body2 = int(model.geom_bodyid[contact.geom2])
    if body1 != 0 and body2 != 0:
      self_penetration = max(self_penetration, penetration)
  return ground, self_penetration


def _has_ground_support(
  model: mujoco.MjModel, data: mujoco.MjData, ground_id: int
) -> bool:
  return any(
    _is_collision_geom(model, geom_id)
    for geom_id in _ground_contact_map(model, data, ground_id)
  )


def _max_contact_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
  force = np.zeros(6, dtype=np.float64)
  maximum = 0.0
  for index in range(data.ncon):
    mujoco.mj_contactForce(model, data, index, force)
    maximum = max(maximum, float(np.linalg.norm(force[:3])))
  return maximum


def _quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
  dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
  return float(2.0 * np.arccos(dot))


def _support_group(name: str) -> str:
  for suffix in ("foot", "shin", "hand", "wrist", "elbow"):
    if suffix in name:
      side = "left" if name.startswith("left_") else "right"
      return f"{side}_{suffix}"
  return name


def _is_collision_geom(model: mujoco.MjModel, geom_id: int) -> bool:
  name = model.geom(geom_id).name
  return name is not None and name.endswith("_collision")


def _is_support_geom(model: mujoco.MjModel, geom_id: int) -> bool:
  name = model.geom(geom_id).name
  return name is not None and _SUPPORT_PATTERN.fullmatch(name) is not None


def _joint_ids(model: mujoco.MjModel) -> NDArray[np.int32]:
  return np.asarray(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      for name in G1_JOINT_NAMES
    ],
    dtype=np.int32,
  )


def _rejected(
  reason: str,
  qpos: NDArray[np.float64],
  *,
  initial_self: float = 0.0,
  ground: float = 0.0,
  force: float = 0.0,
  translation: float = 0.0,
  rotation: float = 0.0,
  support: float = 0.0,
) -> _FrameResult:
  return _FrameResult(
    accepted=False,
    reason=reason,
    qpos=qpos.copy(),
    category="rejected",
    root_projection_m=0.0,
    joint_projection_max_rad=0.0,
    settle_translation_m=translation,
    settle_rotation_rad=rotation,
    support_fraction=support,
    max_contact_force_n=force,
    max_self_penetration_m=initial_self,
    max_ground_penetration_m=ground,
  )


def _jsonable_cfg(cfg: G1PhysicalInitCfg) -> dict[str, Any]:
  return {
    name: str(value) if isinstance(value, Path) else value
    for name, value in vars(cfg).items()
  }
