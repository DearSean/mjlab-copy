"""Reward gates for transitioning from fallen recovery to velocity tracking."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .recovery import recovery_walk_gate
from .rewards import (
  angular_momentum_penalty,
  body_angular_velocity_penalty,
  feet_air_time,
  feet_clearance,
  feet_slip,
  feet_swing_height,
  track_angular_velocity,
  track_linear_velocity,
  upright,
  variable_posture,
)

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


def _reward_gate_scale(
  env: ManagerBasedRlEnv,
  min_scale: float,
) -> torch.Tensor:
  walk_gate = recovery_walk_gate(env)
  return min_scale + (1.0 - min_scale) * walk_gate


def gated_track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Fade linear-velocity tracking in as height and orientation recover."""
  asset_cfg = asset_cfg or SceneEntityCfg("robot")
  gate = recovery_walk_gate(env, asset_cfg)
  return gate * track_linear_velocity(env, std, command_name, asset_cfg)


def gated_track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Fade angular-velocity tracking in as height and orientation recover."""
  asset_cfg = asset_cfg or SceneEntityCfg("robot")
  gate = recovery_walk_gate(env, asset_cfg)
  return gate * track_angular_velocity(env, std, command_name, asset_cfg)


class gated_upright(upright):
  """Keep an upright-state reward floor while smoothly entering walking."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
    gate_min_scale: float = 0.5,
  ) -> torch.Tensor:
    reward = super().__call__(env, std, asset_cfg, terrain_sensor_names)
    return reward * _reward_gate_scale(env, gate_min_scale)


class gated_variable_posture(variable_posture):
  """Relax the default-pose objective during fallen recovery."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
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
    return reward * _reward_gate_scale(env, gate_min_scale)


def gated_body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  gate_min_scale: float,
) -> torch.Tensor:
  """Scale body angular-velocity cost by walking readiness."""
  penalty = body_angular_velocity_penalty(env, asset_cfg)
  return penalty * _reward_gate_scale(env, gate_min_scale)


def gated_angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  gate_min_scale: float,
) -> torch.Tensor:
  """Scale angular-momentum cost by walking readiness."""
  penalty = angular_momentum_penalty(env, sensor_name)
  return penalty * _reward_gate_scale(env, gate_min_scale)


def gated_action_rate_l2(
  env: ManagerBasedRlEnv,
  gate_min_scale: float,
) -> torch.Tensor:
  """Scale action-rate cost by walking readiness."""
  penalty = envs_mdp.action_rate_l2(env)
  return penalty * _reward_gate_scale(env, gate_min_scale)


def gated_feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float,
  threshold_max: float,
  command_name: str,
  command_threshold: float,
  gate_min_scale: float,
) -> torch.Tensor:
  """Scale feet-air-time reward by walking readiness."""
  reward = feet_air_time(
    env,
    sensor_name,
    threshold_min,
    threshold_max,
    command_name,
    command_threshold,
  )
  return reward * _reward_gate_scale(env, gate_min_scale)


def gated_feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str,
  command_threshold: float,
  asset_cfg: SceneEntityCfg,
  gate_min_scale: float,
) -> torch.Tensor:
  """Scale feet-clearance cost by walking readiness."""
  penalty = feet_clearance(
    env,
    target_height,
    height_sensor_name,
    command_name,
    command_threshold,
    asset_cfg,
  )
  return penalty * _reward_gate_scale(env, gate_min_scale)


class gated_feet_swing_height(feet_swing_height):
  """Disable the swing-foot objective during fallen recovery."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
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
    return penalty * _reward_gate_scale(env, gate_min_scale)


def gated_feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float,
  asset_cfg: SceneEntityCfg,
  gate_min_scale: float,
) -> torch.Tensor:
  """Scale feet-slip cost by walking readiness."""
  penalty = feet_slip(
    env,
    sensor_name,
    command_name,
    command_threshold,
    asset_cfg,
  )
  return penalty * _reward_gate_scale(env, gate_min_scale)
