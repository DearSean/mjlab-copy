"""RL Boy velocity environment configurations."""

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.actuator import IdealPdActuator
from mjlab.asset_zoo.robots import (
  RL_BOY_ACTION_SCALE,
  get_rlboy_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.rlboy.recovery_assist import (
  RECOVERY_ASSIST_EVENT_NAME,
  RlBoyRecoveryAssist,
  actuator_torque_limit_excess_penalty,
  normal_group_payload,
  normal_randomization_curriculum,
  prepare_recovery_group,
  push_normal_group,
  recovery_assist_curriculum,
  recovery_assist_reward_weight_curriculum,
  recovery_failure_penalty,
  recovery_mask,
  recovery_succeeded,
  recovery_success_bonus,
  recovery_timed_out,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.string import resolve_expr

_FALLEN_POSES = [
  {
    "pos": (0.0, 0.0, 0.09),
    "quat": (0.70710678, 0.0, -0.70710678, 0.0),
  },
  {
    "pos": (0.0, 0.0, 0.09),
    "quat": (0.70710678, 0.0, 0.70710678, 0.0),
  },
  {
    "pos": (0.0, 0.0, 0.09),
    "quat": (0.70710678, 0.70710678, 0.0, 0.0),
  },
  {
    "pos": (0.0, 0.0, 0.09),
    "quat": (0.70710678, -0.70710678, 0.0, 0.0),
  },
]
_RECOVERY_FRAME_DIR = (
  Path(__file__).resolve().parents[6] / "motions72" / "motions" / "getup_frame_data"
)
_RECOVERY_FRAME_FILES = (
  "getup*.csv",
  "fall*.csv",
)
_RECOVERY_CSV_JOINT_NAMES = (
  "left_hip_yaw_joint",
  "left_hip_roll_joint",
  "left_hip_pitch_joint",
  "left_knee_pitch_joint",
  "left_ankle_pitch_joint",
  "right_hip_yaw_joint",
  "right_hip_roll_joint",
  "right_hip_pitch_joint",
  "right_knee_pitch_joint",
  "right_ankle_pitch_joint",
  "waist_yaw_joint",
  "head_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_pitch_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_pitch_joint",
)

_CONTINUOUS_TORQUE_LIMIT_BY_ACTUATOR = {
  r".*_(shoulder|elbow).*": 0.8,
  r".*_(hip|knee).*": 8.0,
  r"(waist_yaw_joint|head_yaw_joint|.*_ankle_pitch_joint)": 4.0,
}
_PEAK_TORQUE_LIMIT_BY_ACTUATOR = {
  r".*_(shoulder|elbow).*": 3.0,
  r".*_(hip|knee).*": 20.0,
  r"(waist_yaw_joint|head_yaw_joint|.*_ankle_pitch_joint)": 11.0,
}
_TORQUE_FEEDBACK_HISTORY_LENGTH = 4

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_POSTURE_PHASE_CFG = mdp.PosturePhaseEstimatorCfg()
_POSTURE_PHASE_ESTIMATOR = mdp.PosturePhaseEstimator(_POSTURE_PHASE_CFG)

_KNOCKDOWN_STAGES = [
  {
    "step": 0,
    "velocity_range": {
      "x": (-1.5, 1.5),
      "y": (-1.5, 1.5),
      "roll": (-2.5, 2.5),
      "pitch": (-2.5, 2.5),
      "yaw": (-1.0, 1.0),
    },
  }
]

_NORMAL_RANDOMIZATION_STAGES = [
  {
    "step": 0,
    "payload_range": (0.0, 0.0),
    "velocity_range": {},
    "push_interval_s": (8.0, 12.0),
  },
  {
    "step": 1200 * 24,
    "payload_range": (0.0, 0.125),
    "velocity_range": {},
    "push_interval_s": (8.0, 12.0),
  },
  {
    "step": 2000 * 24,
    "payload_range": (0.0, 0.25),
    "velocity_range": {
      "x": (-0.06, 0.06),
      "y": (-0.06, 0.06),
      "yaw": (-0.06, 0.06),
    },
    "push_interval_s": (8.0, 12.0),
  },
  {
    "step": 2800 * 24,
    "payload_range": (0.0, 0.5),
    "velocity_range": {
      "x": (-0.16, 0.16),
      "y": (-0.16, 0.16),
      "roll": (-0.06, 0.06),
      "pitch": (-0.06, 0.06),
      "yaw": (-0.13, 0.13),
    },
    "push_interval_s": (6.0, 10.0),
  },
  {
    "step": 3400 * 24,
    "payload_range": (0.0, 1.0),
    "velocity_range": {
      "x": (-0.32, 0.32),
      "y": (-0.32, 0.32),
      "z": (-0.26, 0.26),
      "roll": (-0.33, 0.33),
      "pitch": (-0.33, 0.33),
      "yaw": (-0.5, 0.5),
    },
    "push_interval_s": (1.0, 3.0),
  },
]

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def _selected_names(names: tuple[str, ...], ids: list[int] | slice) -> tuple[str, ...]:
  if isinstance(ids, slice):
    return names[ids]
  return tuple(names[index] for index in ids)


class _ActuatorLimitTerm:
  def __init__(self, cfg, env: "ManagerBasedRlEnv"):
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._asset: "Entity" = env.scene[self._asset_cfg.name]
    actuator_names = _selected_names(
      self._asset.actuator_names, self._asset_cfg.actuator_ids
    )
    limits = resolve_expr(cfg.params["limit_by_actuator"], actuator_names)
    if any(limit is None for limit in limits):
      missing = [
        name
        for name, limit in zip(actuator_names, limits, strict=True)
        if limit is None
      ]
      raise ValueError(f"Missing torque limit for actuator(s): {missing}")
    self._limits = torch.tensor(limits, device=env.device, dtype=torch.float32)
    if bool((self._limits <= 0.0).any()):
      raise ValueError("Torque limits must be positive.")

  def _applied_ratio(self) -> torch.Tensor:
    torques = self._asset.data.actuator_force[:, self._asset_cfg.actuator_ids]
    return torques / self._limits


class applied_actuator_torque_ratio(_ActuatorLimitTerm):
  """Applied actuator force normalized by a configured per-actuator limit."""

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    limit_by_actuator: dict[str, float],
  ) -> torch.Tensor:
    del env, asset_cfg, limit_by_actuator
    return self._applied_ratio()


class max_abs_applied_actuator_torque_ratio(_ActuatorLimitTerm):
  """Maximum absolute applied torque ratio for each environment."""

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    limit_by_actuator: dict[str, float],
  ) -> torch.Tensor:
    del env, asset_cfg, limit_by_actuator
    return torch.max(torch.abs(self._applied_ratio()), dim=1).values


class requested_actuator_torque_ratio(_ActuatorLimitTerm):
  """Unclipped explicit-actuator effort request normalized by actuator limit."""

  def __init__(self, cfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    all_ctrl_ids = list(range(self._asset.num_actuators))
    if isinstance(self._asset_cfg.actuator_ids, slice):
      selected_ctrl_ids = all_ctrl_ids[self._asset_cfg.actuator_ids]
    else:
      selected_ctrl_ids = self._asset_cfg.actuator_ids
    selected_ctrl_id_set = set(selected_ctrl_ids)
    covered_ctrl_ids: set[int] = set()
    self._explicit_actuators: list[IdealPdActuator] = []
    for actuator in self._asset.actuators:
      actuator_ctrl_ids = set(actuator.ctrl_ids.tolist())
      selected_for_actuator = selected_ctrl_id_set & actuator_ctrl_ids
      if not selected_for_actuator:
        continue
      if not isinstance(actuator, IdealPdActuator):
        names = [
          self._asset.actuator_names[index] for index in sorted(selected_for_actuator)
        ]
        raise ValueError(
          "requested_actuator_torque_ratio requires explicit PD/DC actuators; "
          f"unsupported actuator(s): {names}"
        )
      self._explicit_actuators.append(actuator)
      covered_ctrl_ids.update(selected_for_actuator)
    if covered_ctrl_ids != selected_ctrl_id_set:
      missing = [
        self._asset.actuator_names[index]
        for index in sorted(selected_ctrl_id_set - covered_ctrl_ids)
      ]
      raise ValueError(
        f"Missing explicit actuator effort feedback for actuator(s): {missing}"
      )

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    limit_by_actuator: dict[str, float],
  ) -> torch.Tensor:
    del env, asset_cfg, limit_by_actuator
    requested = torch.zeros_like(self._asset.data.actuator_force)
    for actuator in self._explicit_actuators:
      assert actuator.computed_effort is not None
      requested[:, actuator.ctrl_ids] = actuator.computed_effort
    return requested[:, self._asset_cfg.actuator_ids] / self._limits


def _set_feedback_observation_history(cfg: ManagerBasedRlEnvCfg) -> None:
  history_terms = (
    "joint_pos",
    "joint_vel",
    "actions",
    "applied_torque_continuous_ratio",
    "applied_torque_peak_ratio",
    "requested_torque_peak_ratio",
  )
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    for term_name in history_terms:
      if term_name in terms:
        terms[term_name].history_length = _TORQUE_FEEDBACK_HISTORY_LENGTH
        terms[term_name].flatten_history_dim = True


def _add_torque_feedback_observations(cfg: ManagerBasedRlEnvCfg) -> None:
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    terms["applied_torque_continuous_ratio"] = ObservationTermCfg(
      func=applied_actuator_torque_ratio,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "limit_by_actuator": _CONTINUOUS_TORQUE_LIMIT_BY_ACTUATOR,
      },
      clip=(-2.0, 2.0),
      history_length=_TORQUE_FEEDBACK_HISTORY_LENGTH,
    )
    terms["applied_torque_peak_ratio"] = ObservationTermCfg(
      func=applied_actuator_torque_ratio,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "limit_by_actuator": _PEAK_TORQUE_LIMIT_BY_ACTUATOR,
      },
      clip=(-1.0, 1.0),
      history_length=_TORQUE_FEEDBACK_HISTORY_LENGTH,
    )
    terms["requested_torque_peak_ratio"] = ObservationTermCfg(
      func=requested_actuator_torque_ratio,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "limit_by_actuator": _PEAK_TORQUE_LIMIT_BY_ACTUATOR,
      },
      clip=(-2.0, 2.0),
      history_length=_TORQUE_FEEDBACK_HISTORY_LENGTH,
    )


def base_height_penalty_recovery(
  env: "ManagerBasedRlEnv",
  min_height: float = 0.38,
  recover_height: float = 0.24,
  scale_near: float = 10.0,
  scale_far: float = 3.0,
  max_penalty: float = 4.0,
  recovery_event_name: str | None = None,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """按 base height 惩罚，鼓励策略在跌倒前主动恢复。

  惩罚曲线（base_height 越低惩罚越大）：
  - height >= min_height: 无惩罚
  - recover_height <= height < min_height: 温和的指数惩罚（给策略恢复梯度）
  - height < recover_height: 陡峭的线性惩罚，平滑饱和到 max_penalty

  返回 POSITIVE cost，RewardTermCfg 中需配 NEGATIVE weight 才会变成惩罚。
  """
  if asset_cfg is None:
    asset_cfg = SceneEntityCfg("robot")
  asset: "Entity" = env.scene[asset_cfg.name]
  base_height = asset.data.root_link_pos_w[:, 2]

  # 高度介于 [recover_height, min_height] 之间的小幅下沉（指数区）
  near_error = torch.clamp(
    min_height - torch.clamp(base_height, min=recover_height),
    min=0.0,
  )
  # 跌至 recover_height 之下的严重下沉（线性区）
  far_error = torch.clamp(recover_height - base_height, min=0.0)

  near_penalty = torch.exp(scale_near * near_error) - 1.0
  far_penalty = scale_far * far_error
  raw_penalty = near_penalty + far_penalty

  # 平滑饱和：raw -> 0 时 penalty -> 0；raw -> inf 时 penalty -> max_penalty
  penalty = max_penalty * raw_penalty / (raw_penalty + max_penalty)
  if recovery_event_name is not None:
    penalty *= ~recovery_mask(env, recovery_event_name)
  return penalty


def recovery_state_potential(
  env: "ManagerBasedRlEnv",
  height_weight: float,
  upright_weight: float,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Return a bounded recovery potential from independent height and tilt scores."""
  if asset_cfg is None:
    asset_cfg = SceneEntityCfg("robot")
  if height_weight < 0.0 or upright_weight < 0.0:
    raise ValueError("Recovery potential weights cannot be negative.")
  if not math.isclose(height_weight + upright_weight, 1.0):
    raise ValueError("Recovery potential weights must sum to one.")

  state = _POSTURE_PHASE_ESTIMATOR.estimate(env, asset_cfg)
  # Height remains an independently weighted signal, but only inside the
  # upright hemisphere. This makes the complete potential exactly zero at a
  # 90-degree-or-worse roll/pitch without introducing an angle threshold.
  height_score = state.height * state.uprightness
  return height_weight * height_score + upright_weight * state.uprightness


def recovery_walk_gate(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Measure continuous confidence that the robot is ready for walking."""
  if asset_cfg is None:
    asset_cfg = SceneEntityCfg("robot")
  return _POSTURE_PHASE_ESTIMATOR.estimate(env, asset_cfg).walk_gate


def _reward_gate_scale(
  env: "ManagerBasedRlEnv",
  min_scale: float,
) -> torch.Tensor:
  walk_gate = recovery_walk_gate(env)
  return min_scale + (1.0 - min_scale) * walk_gate


def gated_track_linear_velocity(
  env: "ManagerBasedRlEnv",
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Fade linear-velocity tracking in as height and orientation recover."""
  if asset_cfg is None:
    asset_cfg = SceneEntityCfg("robot")
  gate = recovery_walk_gate(env, asset_cfg)
  return gate * mdp.track_linear_velocity(env, std, command_name, asset_cfg)


def gated_track_angular_velocity(
  env: "ManagerBasedRlEnv",
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Fade angular-velocity tracking in as height and orientation recover."""
  if asset_cfg is None:
    asset_cfg = SceneEntityCfg("robot")
  gate = recovery_walk_gate(env, asset_cfg)
  return gate * mdp.track_angular_velocity(env, std, command_name, asset_cfg)


class gated_upright(mdp.upright):
  """Keep an upright-state reward floor while smoothly entering walking."""

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
    gate_min_scale: float = 0.5,
  ) -> torch.Tensor:
    reward = super().__call__(env, std, asset_cfg, terrain_sensor_names)
    scale = _reward_gate_scale(
      env,
      gate_min_scale,
    )
    return reward * scale


class gated_variable_posture(mdp.variable_posture):
  """Relax the default-pose objective during fallen recovery."""

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    std_standing: object,
    std_walking: object,
    std_running: object,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
    gate_min_scale: float = 0.1,
  ) -> torch.Tensor:
    reward = super().__call__(
      env,
      std_standing,
      std_walking,
      std_running,
      asset_cfg,
      command_name,
      walking_threshold,
      running_threshold,
    )
    scale = _reward_gate_scale(
      env,
      gate_min_scale,
    )
    return reward * scale


def gated_body_angular_velocity_penalty(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
  gate_min_scale: float,
) -> torch.Tensor:
  penalty = mdp.body_angular_velocity_penalty(env, asset_cfg)
  scale = _reward_gate_scale(
    env,
    gate_min_scale,
  )
  return penalty * scale


def gated_angular_momentum_penalty(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  gate_min_scale: float,
) -> torch.Tensor:
  penalty = mdp.angular_momentum_penalty(env, sensor_name)
  scale = _reward_gate_scale(
    env,
    gate_min_scale,
  )
  return penalty * scale


def gated_action_rate_l2(
  env: "ManagerBasedRlEnv",
  gate_min_scale: float,
) -> torch.Tensor:
  penalty = envs_mdp.action_rate_l2(env)
  scale = _reward_gate_scale(
    env,
    gate_min_scale,
  )
  return penalty * scale


def gated_feet_air_time(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  threshold_min: float,
  threshold_max: float,
  command_name: str,
  command_threshold: float,
  gate_min_scale: float,
) -> torch.Tensor:
  reward = mdp.feet_air_time(
    env,
    sensor_name,
    threshold_min,
    threshold_max,
    command_name,
    command_threshold,
  )
  scale = _reward_gate_scale(
    env,
    gate_min_scale,
  )
  return reward * scale


def gated_feet_clearance(
  env: "ManagerBasedRlEnv",
  target_height: float,
  height_sensor_name: str,
  command_name: str,
  command_threshold: float,
  asset_cfg: SceneEntityCfg,
  gate_min_scale: float,
) -> torch.Tensor:
  penalty = mdp.feet_clearance(
    env,
    target_height,
    height_sensor_name,
    command_name,
    command_threshold,
    asset_cfg,
  )
  scale = _reward_gate_scale(
    env,
    gate_min_scale,
  )
  return penalty * scale


class gated_feet_swing_height(mdp.feet_swing_height):
  """Disable the swing-foot objective during fallen recovery."""

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    gate_min_scale: float = 0.0,
  ) -> torch.Tensor:
    penalty = super().__call__(
      env,
      sensor_name,
      height_sensor_name,
      target_height,
      command_name,
      command_threshold,
    )
    scale = _reward_gate_scale(
      env,
      gate_min_scale,
    )
    return penalty * scale


def gated_feet_slip(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  command_name: str,
  command_threshold: float,
  asset_cfg: SceneEntityCfg,
  gate_min_scale: float,
) -> torch.Tensor:
  penalty = mdp.feet_slip(env, sensor_name, command_name, command_threshold, asset_cfg)
  scale = _reward_gate_scale(
    env,
    gate_min_scale,
  )
  return penalty * scale


class recovery_potential_progress:
  """Reward changes in a bounded recovery potential while assistance is active."""

  def __init__(
    self,
    cfg: RewardTermCfg,
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg | None = None,
    **_: object,
  ):
    del cfg
    self.asset_cfg = asset_cfg or SceneEntityCfg("robot")
    self.previous_potential = torch.zeros(env.num_envs, device=env.device)
    self.was_active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.was_active[env_ids] = False

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    event_name: str,
    height_weight: float,
    upright_weight: float,
    asset_cfg: SceneEntityCfg | None = None,
  ) -> torch.Tensor:
    del asset_cfg  # The resolved asset config from __init__ is used.
    potential = recovery_state_potential(
      env,
      height_weight,
      upright_weight,
      self.asset_cfg,
    )
    active = recovery_mask(env, event_name)
    continuing = active & self.was_active
    delta = torch.where(
      continuing,
      potential - self.previous_potential,
      torch.zeros_like(potential),
    )
    self.previous_potential.copy_(potential)
    self.was_active.copy_(active)
    return delta / env.step_dt


def recovery_time_penalty(
  env: "ManagerBasedRlEnv",
  event_name: str,
) -> torch.Tensor:
  """Return a constant per-second cost while a recovery attempt is active."""
  return recovery_mask(env, event_name).float()


def rlboy_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create RL Boy rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()
  assert cfg is not None

  # 仿真参数
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70
  from mjlab.utils.nan_guard import NanGuardCfg

  cfg.sim.nan_guard = NanGuardCfg(
    enabled=True,
    buffer_size=100,  # 保留 NaN 前多少步
    output_dir="/tmp/mjlab/nan_dumps",
    max_envs_to_dump=5,  # 最多导出的 env 数量
  )

  # 替换机器人实体
  cfg.scene.entities = {"robot": get_rlboy_robot_cfg()}
  _add_torque_feedback_observations(cfg)
  _set_feedback_observation_history(cfg)
  cfg.metrics["applied_torque_continuous_ratio"] = MetricsTermCfg(
    func=max_abs_applied_actuator_torque_ratio,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "limit_by_actuator": _CONTINUOUS_TORQUE_LIMIT_BY_ACTUATOR,
    },
  )

  # 设置地形射线扫描传感器帧为 RL_BOY 的 base_link
  # 注意: RL_BOY 没有 pelvis，使用 base_link 作为主躯干
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"

  # 足端高度扫描绑定到左右脚踝 site
  # 注意: RL_BOY 只有 ankle_pitch，没有 ankle_roll
  # 需要在 XML 中添加 foot site，或者使用现有的 ankle link
  # RL_BOY 机器人脚部 site 名称（已在 XML 的 ankle body 下添加 left_foot / right_foot）
  site_names = ("left_foot", "right_foot")
  foot_body_names = ("left_ankle_pitch_link", "right_ankle_pitch_link")
  foot_geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="body", name=s, entity="robot") for s in foot_body_names
      )
      # 使用更小的 ring radius 因为 RL_BOY 的脚较小
      sensor.pattern = RingPatternCfg.single_ring(radius=0.02, num_samples=6)

  # 脚部接触传感器
  # RL_BOY 使用 foot geom 作为脚部
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_pitch_link|right_ankle_pitch_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  # 自碰撞检测传感器
  # RL_BOY 使用 base_link 作为主躯干
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # 设置动作缩放
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = RL_BOY_ACTION_SCALE

  # 设置 Viewer 视角主体
  # RL_BOY 有 base_link 作为视觉参考点
  cfg.viewer.body_name = "base_link"

  # 命令可视化偏移
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # TODO: 根据实际机器人高度调整
  twist_cmd.viz.z_offset = 0.5  # RL_BOY 站立高度约 0.45m

  # 事件配置 - 摩擦随机化
  # RL_BOY 脚部 geom 名称
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events.pop("reset_robot_joints", None)

  # ===== 新增域随机化 (DR) =====
  # 1) PD 增益随机化 ±10%：让策略适应电机响应的实物差异
  #    asset_cfg 不指定 actuator_names 时默认 actuator_ids=slice(None)，覆盖所有 20 个执行器
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
    },
  )
  # 2) 全身 link 质量随机化 ±5%（pseudo_inertia 同步缩放 mass+inertia，物理上一致）
  #    alpha = log(density_ratio), ±5% 质量变化 ≈ alpha_range = (-0.05, 0.05)
  cfg.events["link_mass"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=r"^(?!left_wrist_link$|right_wrist_link$).+",
      ),
      "alpha_range": (-0.05, 0.05),
    },
  )
  # 3) 负载随机化：课程按阶段扩大范围，每个 episode 重新采样
  #    注：body_mass 单独使用时不修改 inertia，docstring 明确指出仅适用于「在 COM 添加点质量」的场景
  cfg.events["base_payload"] = EventTermCfg(
    mode="reset",
    func=envs_mdp.dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
      "operation": "add",
      "ranges": (0.0, 0.25),
    },
  )

  # ===== 新增域随机化结束 =====

  # 姿态奖励标准差配置
  # RL_BOY 关节结构:
  # - 腿部: hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch (无 ankle_roll)
  # - 腰部: waist_yaw
  # - 手臂: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch (无 _joint 后缀)
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # 腿部关节
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee_pitch.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    # 注意: RL_BOY 没有 ankle_roll 关节
    # 腰部关节
    r".*waist_yaw.*": 0.2,
    # 头部关节（锁定，不主动运动）
    r".*head_yaw.*": 0.05,
    # 手臂关节
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow_pitch.*": 0.15,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # 腿部关节
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee_pitch.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    # 腰部关节
    r".*waist_yaw.*": 0.3,
    # 头部关节（锁定，不主动运动）
    r".*head_yaw.*": 0.05,
    # 手臂关节
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow_pitch.*": 0.35,
  }

  # 躯干直立奖励主体
  # RL_BOY 使用 base_link 作为主躯干
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  # 足部清洁和滑动奖励
  # 注意: RL_BOY 没有 ankle_roll，使用 ankle_pitch 作为脚部参考
  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # 奖励权重调整
  cfg.rewards["pose"].weight = 0.5
  cfg.rewards["action_rate_l2"].weight = -0.03
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.2

  for reward_name, reward_func in (
    ("track_linear_velocity", gated_track_linear_velocity),
    ("track_angular_velocity", gated_track_angular_velocity),
  ):
    cfg.rewards[reward_name].func = reward_func

  gated_reward_cfgs = {
    "upright": (gated_upright, 0.5),
    "pose": (gated_variable_posture, 0.1),
    "body_ang_vel": (gated_body_angular_velocity_penalty, 0.2),
    "angular_momentum": (gated_angular_momentum_penalty, 0.2),
    "action_rate_l2": (gated_action_rate_l2, 0.3),
    "air_time": (gated_feet_air_time, 0.0),
    "foot_clearance": (gated_feet_clearance, 0.0),
    "foot_swing_height": (gated_feet_swing_height, 0.0),
    "foot_slip": (gated_feet_slip, 0.1),
  }
  for reward_name, (reward_func, min_scale) in gated_reward_cfgs.items():
    cfg.rewards[reward_name].func = reward_func
    cfg.rewards[reward_name].params.update(
      {
        "gate_min_scale": min_scale,
      }
    )

  # base height 恢复奖励
  # 在策略接近跌倒时提供早期梯度,鼓励主动恢复姿态
  cfg.rewards["base_height_recovery"] = RewardTermCfg(
    func=base_height_penalty_recovery,
    weight=-0.5,
    params={
      # RL_BOY 站立时 base_link 高度约 0.45m
      "min_height": 0.38,
      # 跌至 0.24m 以下视为不可恢复(严重跌倒)
      "recover_height": 0.24,
      "scale_near": 10.0,
      "scale_far": 3.0,
      "max_penalty": 4.0,
    },
  )

  # 自碰撞惩罚
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.3,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Rough terrain keeps the ordinary deterministic fall termination.
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=envs_mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  # Play 模式覆盖
  if play:
    # 无限episode长度
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def rlboy_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create RL Boy flat terrain velocity configuration."""
  cfg = rlboy_rough_env_cfg(play=play)

  cfg.sim.njmax = 500
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # 切换到平地地形
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # 移除地形扫描传感器 (无地形可扫描)
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # 禁用地形课程
  cfg.curriculum.pop("terrain_levels", None)

  if not play:
    # The reset population only selects initial poses. Assistance activates
    # dynamically for any robot that remains fallen, while curriculum outcomes
    # continue to use the controlled initial-pose population.
    cfg.events[RECOVERY_ASSIST_EVENT_NAME] = EventTermCfg(
      func=RlBoyRecoveryAssist,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("waist_yaw_link",)),
        "poses": _FALLEN_POSES,
        "recovery_stage_probabilities": (0.6, 0.5, 0.4),
        "post_stage_recovery_probability": 0.35,
        "low_force_recovery_probability": 0.3,
        "recovery_probability_limits": (0.25, 0.65),
        "recovery_probability_feedback_gain": 0.5,
        "recovery_probability_smoothing": 0.1,
        "recovery_probability_min_attempts": 50,
        "angle_noise_ramp_attempts": 300,
        "frame_dir": str(_RECOVERY_FRAME_DIR),
        "frame_files": _RECOVERY_FRAME_FILES,
        "source_names": ("getup", "fall", "canonical"),
        "csv_joint_names": _RECOVERY_CSV_JOINT_NAMES,
        "pose_stage_source_weights": (
          (1.0, 0.0, 0.0),
          (1.0, 0.0, 0.0),
          (0.15, 0.25, 0.6),
        ),
        "force_ranges": (
          (50.0, 50.0),
          (40.0, 45.0),
          (32.0, 38.0),
          (25.0, 30.0),
          (20.0, 24.0),
          (12.0, 19.0),
          (6.0, 11.0),
          (0.0, 5.0),
          (0.0, 0.0),
        ),
        "posture_cfg": _POSTURE_PHASE_CFG,
        "fall_confirm_s": 0.12,
        "upright_hold_s": 0.5,
        "force_ramp_up_s": 0.3,
        "force_ramp_down_s": 0.5,
        "recovery_timeout_s": 5.0,
        "root_height_range": (0.1, 0.13),
        "root_lin_vel_range": (-0.1, 0.1),
        "root_ang_vel_range": (-0.2, 0.2),
        "joint_position_ranges": {
          r".*_hip_pitch_joint": (-0.25, 0.25),
          r".*_hip_roll_joint": (-0.15, 0.15),
          r".*_hip_yaw_joint": (-0.12, 0.12),
          r".*_knee_pitch_joint": (-0.3, 0.3),
          r".*_ankle_pitch_joint": (-0.18, 0.18),
          r".*_shoulder_pitch_joint": (-0.4, 0.4),
          r".*_shoulder_roll_joint": (-0.25, 0.25),
          r".*_shoulder_yaw_joint": (-0.2, 0.2),
          r".*_elbow_pitch_joint": (-0.35, 0.35),
          "waist_yaw_joint": (-0.15, 0.15),
          "head_yaw_joint": (0.0, 0.0),
        },
        "joint_velocity_ranges": {
          r".*(hip|knee|ankle).*": (-0.25, 0.25),
          r".*(shoulder|elbow).*": (-0.4, 0.4),
          "waist_yaw_joint": (-0.15, 0.15),
          "head_yaw_joint": (0.0, 0.0),
        },
      },
    )
    cfg.events["push_robot"] = EventTermCfg(
      func=push_normal_group,
      mode="interval",
      interval_range_s=(8.0, 12.0),
      params={
        "event_name": RECOVERY_ASSIST_EVENT_NAME,
        "stages": _NORMAL_RANDOMIZATION_STAGES,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    )
    cfg.events["knockdown_robot"] = EventTermCfg(
      func=push_normal_group,
      mode="interval",
      interval_range_s=(13.0, 15.0),
      params={
        "event_name": RECOVERY_ASSIST_EVENT_NAME,
        "stages": _KNOCKDOWN_STAGES,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    )
    cfg.events["base_payload"] = EventTermCfg(
      func=normal_group_payload,
      mode="reset",
      params={
        "event_name": RECOVERY_ASSIST_EVENT_NAME,
        "stages": _NORMAL_RANDOMIZATION_STAGES,
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      },
    )
    cfg.events = {
      "prepare_recovery_group": EventTermCfg(
        func=prepare_recovery_group,
        mode="reset",
        params={"event_name": RECOVERY_ASSIST_EVENT_NAME},
      ),
      **cfg.events,
    }
    cfg.rewards["base_height_recovery"].params["recovery_event_name"] = (
      RECOVERY_ASSIST_EVENT_NAME
    )
    cfg.rewards["recovery_potential_progress"] = RewardTermCfg(
      func=recovery_potential_progress,
      weight=2.0,
      params={
        "event_name": RECOVERY_ASSIST_EVENT_NAME,
        "height_weight": 0.6,
        "upright_weight": 0.4,
      },
    )
    cfg.rewards["recovery_time"] = RewardTermCfg(
      func=recovery_time_penalty,
      weight=-0.05,
      params={"event_name": RECOVERY_ASSIST_EVENT_NAME},
    )
    cfg.rewards["recovery_success"] = RewardTermCfg(
      func=recovery_success_bonus,
      weight=1.0,
      params={"event_name": RECOVERY_ASSIST_EVENT_NAME},
    )
    cfg.rewards["recovery_failure"] = RewardTermCfg(
      func=recovery_failure_penalty,
      weight=-2.0,
      params={"event_name": RECOVERY_ASSIST_EVENT_NAME},
    )
    cfg.rewards["continuous_torque_excess"] = RewardTermCfg(
      func=actuator_torque_limit_excess_penalty,
      weight=0.0,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "limit_by_actuator": _CONTINUOUS_TORQUE_LIMIT_BY_ACTUATOR,
        "threshold_ratio": 1.0,
        "log_prefix": "continuous_torque_excess",
      },
    )
    cfg.rewards["peak_torque_usage"] = RewardTermCfg(
      func=actuator_torque_limit_excess_penalty,
      weight=0.0,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "limit_by_actuator": _PEAK_TORQUE_LIMIT_BY_ACTUATOR,
        "threshold_ratio": 0.85,
        "log_prefix": "peak_torque_usage",
      },
    )
    cfg.terminations["recovery_succeeded"] = TerminationTermCfg(
      func=recovery_succeeded,
      time_out=True,
      params={"event_name": RECOVERY_ASSIST_EVENT_NAME},
    )
    cfg.terminations["recovery_timed_out"] = TerminationTermCfg(
      func=recovery_timed_out,
      params={"event_name": RECOVERY_ASSIST_EVENT_NAME},
    )
    cfg.curriculum["recovery_assist"] = CurriculumTermCfg(
      func=recovery_assist_curriculum,
      params={
        "event_name": RECOVERY_ASSIST_EVENT_NAME,
        "window_size": 500,
        "success_threshold": 0.9,
      },
    )
    cfg.curriculum["torque_penalties"] = CurriculumTermCfg(
      func=recovery_assist_reward_weight_curriculum,
      params={
        "event_name": RECOVERY_ASSIST_EVENT_NAME,
        "assist_level": 6,
        "assist_weights": {
          "continuous_torque_excess": -0.02,
          "peak_torque_usage": -0.01,
        },
        "complete_weights": {
          "continuous_torque_excess": -0.05,
          "peak_torque_usage": -0.02,
        },
      },
    )
    cfg.curriculum["normal_randomization"] = CurriculumTermCfg(
      func=normal_randomization_curriculum,
      params={
        "push_event_name": "push_robot",
        "stages": _NORMAL_RANDOMIZATION_STAGES,
      },
    )

  if not play:
    # 定制速度指令课程学习。Play 模式保留下方手工设置的指令范围。
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
      func=mdp.commands_vel,
      log=False,
      params={
        "command_name": "twist",
        "payload_event_name": None,
        "velocity_stages": [
          # 阶段 0: 起步 —— 小范围、低速
          {
            "step": 0,
            "lin_vel_x": (-0.6, 0.8),
            "lin_vel_y": (-0.3, 0.3),
            "ang_vel_z": (-0.4, 0.4),
            "payload_range": (0.0, 0.25),
          },
          # 阶段 1: 提升 x 方向速度上限
          {
            "step": 800 * 24,
            "lin_vel_x": (-1.0, 1.2),
            "lin_vel_y": (-0.5, 0.5),
            "ang_vel_z": (-0.6, 0.6),
            "payload_range": (0.0, 0.5),
          },
          # 阶段 2: 进一步提速并扩大侧向与偏航
          {
            "step": 1600 * 24,
            "lin_vel_x": (-1.5, 1.8),
            "lin_vel_y": (-0.7, 0.7),
            "ang_vel_z": (-0.8, 0.8),
            "payload_range": (0.0, 1.0),
          },
          # 阶段 3: 接近最终能力上限
          {
            "step": 3200 * 24,
            "lin_vel_x": (-2.0, 2.5),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": (-1.0, 1.0),
            "payload_range": (0.0, 2.0),
          },
        ],
      },
    )

  # Falling is never terminal in the flat task. Dedicated recovery episodes still
  # use their own success and timeout conditions during training.
  cfg.terminations.pop("fell_over", None)

  for reward_name in (
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
    "self_collisions",
  ):
    cfg.rewards[reward_name].log = False
  cfg.metrics.pop("mean_action_acc", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    # TODO: 根据 RL_BOY 的实际能力调整速度范围
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def rlboy_flat_gru_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the flat RL Boy environment for an actor-GRU policy.

  The recurrent actor receives only signals that are intended to be available
  on the robot. The MLP critic keeps the complete privileged observation set.
  Both groups use current-frame observations; temporal state is represented by
  the actor GRU instead of explicit observation stacking.
  """
  cfg = rlboy_flat_env_cfg(play=play)

  for group_cfg in cfg.observations.values():
    group_cfg.history_length = None
    for term_cfg in group_cfg.terms.values():
      term_cfg.history_length = 0

  actor_terms = cfg.observations["actor"].terms
  del actor_terms["base_lin_vel"]
  del actor_terms["applied_torque_peak_ratio"]

  return cfg
