"""Fallen-recovery assistance curriculum for the RL_BOY velocity task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


RECOVERY_ASSIST_EVENT_NAME = "recovery_assist"


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
  """Multiply quaternions in ``(w, x, y, z)`` order."""
  w1, x1, y1, z1 = lhs.unbind(-1)
  w2, x2, y2, z2 = rhs.unbind(-1)
  return torch.stack(
    (
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ),
    dim=-1,
  )


class RlBoyRecoveryAssist:
  """Manage recovery-group resets, upward assistance, and recovery outcomes."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self._env = env
    self._asset: Entity = env.scene[params["asset_cfg"].name]
    self._body_ids = params["asset_cfg"].body_ids
    self._command_name: str = params["command_name"]
    self._poses: list[dict[str, tuple[float, ...]]] = params["poses"]
    self._recovery_probability: float = params["recovery_probability"]
    self._force_levels = torch.tensor(
      params["force_levels"], device=env.device, dtype=torch.float32
    )
    self._upright_height: float = params["upright_height"]
    self._upright_angle: float = params["upright_angle"]
    self._ramp_duration_s: float = params["ramp_duration_s"]
    self._independent_hold_s: float = params["independent_hold_s"]
    self._recovery_timeout_s: float = params["recovery_timeout_s"]

    self.level = 0
    self.is_recovery = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.succeeded = torch.zeros_like(self.is_recovery)
    self.elapsed_s = torch.zeros(env.num_envs, device=env.device)
    self.hold_s = torch.zeros_like(self.elapsed_s)
    self.applied_force = torch.zeros_like(self.elapsed_s)
    self.pose_index = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )

    num_poses = len(self._poses)
    self.attempts = torch.zeros(num_poses, device=env.device, dtype=torch.long)
    self.successes = torch.zeros_like(self.attempts)

  @property
  def target_force(self) -> float:
    return float(self._force_levels[self.level].item())

  @property
  def recovery_timeout_s(self) -> float:
    return self._recovery_timeout_s

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Apply fallen poses after all reset-mode randomization has completed."""
    if env_ids is None:
      env_ids = torch.arange(
        self._env.num_envs, device=self._env.device, dtype=torch.long
      )

    recovery_ids = env_ids[self.is_recovery[env_ids]]
    if len(recovery_ids) > 0:
      pose_indices = self.pose_index[recovery_ids]
      root_pos = torch.tensor(
        [pose["pos"] for pose in self._poses],
        device=self._env.device,
        dtype=torch.float32,
      )[pose_indices]
      root_quat = torch.tensor(
        [pose["quat"] for pose in self._poses],
        device=self._env.device,
        dtype=torch.float32,
      )[pose_indices]
      root_pos += self._env.scene.env_origins[recovery_ids]

      yaw = torch.rand(len(recovery_ids), device=self._env.device) * 2.0 * math.pi
      half_yaw = 0.5 * yaw
      yaw_quat = torch.stack(
        (
          torch.cos(half_yaw),
          torch.zeros_like(yaw),
          torch.zeros_like(yaw),
          torch.sin(half_yaw),
        ),
        dim=-1,
      )
      root_quat = _quat_mul(yaw_quat, root_quat)
      root_state = torch.zeros(
        len(recovery_ids), 13, device=self._env.device, dtype=torch.float32
      )
      root_state[:, :3] = root_pos
      root_state[:, 3:7] = root_quat
      self._asset.write_root_state_to_sim(root_state, env_ids=recovery_ids)
      self.applied_force[recovery_ids] = self.target_force

    self._zero_recovery_commands()
    self._write_force(env_ids)

  def prepare_group(self, env_ids: torch.Tensor) -> None:
    """Choose episode groups before reset-mode randomization terms run."""
    recovery_mask = (
      torch.rand(len(env_ids), device=self._env.device)
      < self._recovery_probability
    )
    recovery_ids = env_ids[recovery_mask]

    self.is_recovery[env_ids] = False
    self.is_recovery[recovery_ids] = True
    self.succeeded[env_ids] = False
    self.elapsed_s[env_ids] = 0.0
    self.hold_s[env_ids] = 0.0
    self.applied_force[env_ids] = 0.0
    self.pose_index[env_ids] = -1
    if len(recovery_ids) > 0:
      self.pose_index[recovery_ids] = torch.randint(
        len(self._poses), (len(recovery_ids),), device=self._env.device
      )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    poses: list[dict[str, tuple[float, ...]]],
    recovery_probability: float,
    force_levels: tuple[float, ...],
    upright_height: float,
    upright_angle: float,
    ramp_duration_s: float,
    independent_hold_s: float,
    recovery_timeout_s: float,
  ) -> None:
    dt = env.step_dt
    del (
      env_ids,
      asset_cfg,
      command_name,
      poses,
      recovery_probability,
      force_levels,
      upright_height,
      upright_angle,
      ramp_duration_s,
      independent_hold_s,
      recovery_timeout_s,
    )
    active = self.is_recovery & ~self.succeeded
    self.elapsed_s[active] += dt

    height = self._asset.data.root_link_pos_w[:, 2]
    gravity_z = self._asset.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0)
    angle = torch.acos(-gravity_z).abs()
    upright = (height >= self._upright_height) & (angle <= self._upright_angle)

    ramp_rate = self.target_force / max(self._ramp_duration_s, dt)
    ramp_down = active & upright
    ramp_up = active & ~upright
    self.applied_force[ramp_down] = torch.clamp(
      self.applied_force[ramp_down] - ramp_rate * dt, min=0.0
    )
    self.applied_force[ramp_up] = torch.clamp(
      self.applied_force[ramp_up] + ramp_rate * dt, max=self.target_force
    )

    independent = active & upright & (self.applied_force <= 1e-4)
    self.hold_s[independent] += dt
    self.hold_s[active & ~independent] = 0.0
    newly_succeeded = active & (self.hold_s >= self._independent_hold_s)
    self.succeeded[newly_succeeded] = True
    self.applied_force[newly_succeeded] = 0.0

    self._zero_recovery_commands()
    self._write_force()

  def _zero_recovery_commands(self) -> None:
    command = self._env.command_manager.get_command(self._command_name)
    assert isinstance(command, torch.Tensor)
    command[self.is_recovery] = 0.0
    term = self._env.command_manager.get_term(self._command_name)
    for attr in ("vel_command_w", "heading_target"):
      value = getattr(term, attr, None)
      if value is not None:
        value[self.is_recovery] = 0.0
    for attr in (
      "is_heading_env",
      "is_world_env",
      "is_forward_env",
      "is_standing_env",
    ):
      value = getattr(term, attr, None)
      if value is not None:
        value[self.is_recovery] = False

  def _write_force(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(
        self._env.num_envs, device=self._env.device, dtype=torch.long
      )
    forces = torch.zeros(
      len(env_ids), 1, 3, device=self._env.device, dtype=torch.float32
    )
    forces[:, 0, 2] = self.applied_force[env_ids]
    torques = torch.zeros_like(forces)
    self._asset.write_external_wrench_to_sim(
      forces, torques, env_ids=env_ids, body_ids=self._body_ids
    )

  def record_outcomes(self, env_ids: torch.Tensor) -> None:
    """Accumulate completed recovery attempts by initial fallen direction."""
    recovery_ids = env_ids[self.is_recovery[env_ids]]
    if len(recovery_ids) == 0:
      return
    pose_indices = self.pose_index[recovery_ids]
    self.attempts += torch.bincount(pose_indices, minlength=len(self._poses)).to(
      self.attempts
    )
    successful_poses = pose_indices[self.succeeded[recovery_ids]]
    self.successes += torch.bincount(successful_poses, minlength=len(self._poses)).to(
      self.successes
    )

  def update_level(
    self,
    window_size: int,
    success_threshold: float,
    direction_threshold: float,
    min_direction_attempts: int,
    failure_threshold: float,
  ) -> None:
    attempts = int(self.attempts.sum().item())
    if attempts < window_size:
      return
    overall_rate = float(self.successes.sum().item()) / max(attempts, 1)
    direction_ready = bool((self.attempts >= min_direction_attempts).all())
    direction_rates = self.successes.float() / self.attempts.clamp_min(1)

    if (
      self.level < len(self._force_levels) - 1
      and overall_rate >= success_threshold
      and direction_ready
      and bool((direction_rates >= direction_threshold).all())
    ):
      self.level += 1
    elif self.level > 0 and overall_rate < failure_threshold:
      self.level -= 1
    self.attempts.zero_()
    self.successes.zero_()

  def curriculum_state(self) -> dict[str, torch.Tensor]:
    attempts = self.attempts.sum()
    rate = self.successes.sum().float() / attempts.clamp_min(1)
    return {
      "level": torch.tensor(self.level, device=self._env.device),
      "force_n": self._force_levels[self.level],
      "attempts": attempts,
      "success_rate": rate,
    }


def _get_assist(env: ManagerBasedRlEnv, event_name: str) -> RlBoyRecoveryAssist:
  term = env.event_manager.get_term_cfg(event_name).func
  if not isinstance(term, RlBoyRecoveryAssist):
    raise TypeError(f"Event '{event_name}' is not an RlBoyRecoveryAssist.")
  return term


def recovery_assist_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  event_name: str,
  window_size: int,
  success_threshold: float,
  direction_threshold: float,
  min_direction_attempts: int,
  failure_threshold: float,
) -> dict[str, torch.Tensor]:
  """Update assistance using outcomes from fallen-recovery environments only."""
  if isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  assist = _get_assist(env, event_name)
  assist.record_outcomes(env_ids)
  assist.update_level(
    window_size,
    success_threshold,
    direction_threshold,
    min_direction_attempts,
    failure_threshold,
  )
  return assist.curriculum_state()


def prepare_recovery_group(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  event_name: str,
) -> None:
  """Select recovery episodes before other reset-mode events execute."""
  if isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  _get_assist(env, event_name).prepare_group(env_ids)


def _active_stage(
  step_counter: int,
  stages: list[dict[str, Any]],
) -> dict[str, Any]:
  stage = stages[0]
  for candidate in stages:
    if step_counter >= candidate["step"]:
      stage = candidate
  return stage


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def normal_group_payload(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  event_name: str,
  stages: list[dict[str, Any]],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Randomize normal-group payload and clear it for recovery episodes."""
  if isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  assist = _get_assist(env, event_name)
  recovery_ids = env_ids[assist.is_recovery[env_ids]]
  normal_ids = env_ids[~assist.is_recovery[env_ids]]
  stage = _active_stage(env.common_step_counter, stages)

  if len(recovery_ids) > 0:
    envs_mdp.dr.body_mass(
      env,
      recovery_ids,
      ranges=(0.0, 0.0),
      operation="add",
      asset_cfg=asset_cfg,
    )
  if len(normal_ids) > 0:
    envs_mdp.dr.body_mass(
      env,
      normal_ids,
      ranges=stage["payload_range"],
      operation="add",
      asset_cfg=asset_cfg,
    )


def push_normal_group(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  stages: list[dict[str, Any]],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Apply the active push stage to normal environments only."""
  assist = _get_assist(env, event_name)
  normal_ids = env_ids[~assist.is_recovery[env_ids]]
  if len(normal_ids) == 0:
    return
  velocity_range = _active_stage(env.common_step_counter, stages)["velocity_range"]
  if not velocity_range:
    return
  envs_mdp.push_by_setting_velocity(
    env,
    normal_ids,
    velocity_range=velocity_range,
    asset_cfg=asset_cfg,
  )


def normal_randomization_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  push_event_name: str,
  stages: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
  """Update push timing and report the normal-group randomization stage."""
  del env_ids
  stage_index = 0
  for index, candidate in enumerate(stages):
    if env.common_step_counter >= candidate["step"]:
      stage_index = index
  stage = stages[stage_index]
  env.event_manager.get_term_cfg(
    push_event_name
  ).interval_range_s = stage["push_interval_s"]
  max_push = max(
    (max(abs(low), abs(high)) for low, high in stage["velocity_range"].values()),
    default=0.0,
  )
  return {
    "stage": torch.tensor(stage_index, device=env.device),
    "payload_max": torch.tensor(stage["payload_range"][1], device=env.device),
    "push_max": torch.tensor(max_push, device=env.device),
  }


def recovery_bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  event_name: str,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Ignore bad orientation only for active recovery-group episodes."""
  if asset_cfg is None:
    asset_cfg = SceneEntityCfg("robot")
  asset: Entity = env.scene[asset_cfg.name]
  angle = torch.acos(-asset.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0)).abs()
  fell = angle > limit_angle
  return fell & ~_get_assist(env, event_name).is_recovery


def recovery_succeeded(
  env: ManagerBasedRlEnv,
  event_name: str,
) -> torch.Tensor:
  """End a recovery episode after unassisted stable standing."""
  return (
    _get_assist(env, event_name).is_recovery & _get_assist(env, event_name).succeeded
  )


def recovery_timed_out(
  env: ManagerBasedRlEnv,
  event_name: str,
) -> torch.Tensor:
  """Terminate recovery episodes that do not stand before their deadline."""
  assist = _get_assist(env, event_name)
  return (
    assist.is_recovery
    & ~assist.succeeded
    & (assist.elapsed_s >= assist.recovery_timeout_s)
  )


def recovery_mask(env: ManagerBasedRlEnv, event_name: str) -> torch.Tensor:
  """Return the recovery-group mask for reward terms."""
  return _get_assist(env, event_name).is_recovery
