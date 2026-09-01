"""Task, tracking, termination, and logging terms for G1 recovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from .recovery_events import (
  FALLEN_MODE,
  REFERENCE_MODE,
  STAND_MODE,
  get_g1_recovery_state,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def recovery_progress_reward(
  env: ManagerBasedRlEnv,
  event_name: str,
  max_progress_rate: float,
  max_regression_rate: float,
) -> torch.Tensor:
  """Reward progress per second so RewardManager's dt scaling cancels cleanly."""
  state = get_g1_recovery_state(env, event_name)
  rate = (state.current_progress - state.previous_progress) / env.step_dt
  return rate.clamp(-max_regression_rate, max_progress_rate)


def recovery_gate(progress: torch.Tensor, low: float, high: float) -> torch.Tensor:
  """Return a smooth transition from recovery to standing objectives."""
  if high <= low:
    raise ValueError(f"recovery gate requires high > low, got {low=} and {high=}")
  ratio = ((progress - low) / (high - low)).clamp(0.0, 1.0)
  return ratio.square() * (3.0 - 2.0 * ratio)


def _gated_scale(
  env: ManagerBasedRlEnv,
  event_name: str,
  minimum: float,
  gate_low: float,
  gate_high: float,
) -> torch.Tensor:
  progress = get_g1_recovery_state(env, event_name).current_progress
  gate = recovery_gate(progress, gate_low, gate_high)
  return minimum + (1.0 - minimum) * gate


class GatedUpright:
  """Retain an upright attractor without dominating early recovery."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._reward = mdp.upright(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
    terrain_sensor_names: tuple[str, ...] | None = None,
    event_name: str = "g1_recovery_reset",
    gate_minimum: float = 0.5,
    gate_low: float = 0.55,
    gate_high: float = 0.85,
  ) -> torch.Tensor:
    reward = self._reward(env, std, asset_cfg, terrain_sensor_names)
    return reward * _gated_scale(env, event_name, gate_minimum, gate_low, gate_high)


class GatedVariablePosture:
  """Relax the default standing pose while large recovery motions are needed."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._reward = mdp.variable_posture(cfg, env)

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
    event_name: str = "g1_recovery_reset",
    gate_minimum: float = 0.1,
    gate_low: float = 0.55,
    gate_high: float = 0.85,
  ) -> torch.Tensor:
    reward = self._reward(
      env,
      std_standing,
      std_walking,
      std_running,
      asset_cfg,
      command_name,
      walking_threshold,
      running_threshold,
    )
    return reward * _gated_scale(env, event_name, gate_minimum, gate_low, gate_high)


class BroadRecoveryPosture:
  """Reward the default pose without saturating throughout recovery."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    asset = env.scene[cfg.params["asset_cfg"].name]
    limits = asset.data.default_joint_pos_limits[0]
    self._default_joint_pos = asset.data.default_joint_pos[0].clone()
    self._pose_scale = (0.5 * (limits[:, 1] - limits[:, 0])).clamp_min(0.1)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    event_name: str,
    gain: float,
    gate_minimum: float,
    gate_low: float,
    gate_high: float,
  ) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    error = (asset.data.joint_pos - self._default_joint_pos) / self._pose_scale
    reward = torch.exp(-gain * torch.mean(error.square(), dim=1))
    return reward * _gated_scale(env, event_name, gate_minimum, gate_low, gate_high)


class RecoveryCompositeProgress:
  """Reward independent height and upright progress without a reset spike."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self._previous = torch.zeros(env.num_envs, device=env.device)
    self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._initialized[ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    height_weight: float,
    upright_weight: float,
    max_progress_rate: float,
    max_regression_rate: float,
  ) -> torch.Tensor:
    if abs(height_weight + upright_weight - 1.0) > 1e-6:
      raise ValueError("Composite recovery progress weights must sum to one.")
    asset = env.scene[asset_cfg.name]
    root_height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    nominal_height = asset.data.default_root_state[:, 2].clamp_min(1e-6)
    height = (root_height / nominal_height).clamp(0.0, 1.0)
    upright = (-asset.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
    potential = height_weight * height + upright_weight * upright
    delta = torch.where(
      self._initialized,
      potential - self._previous,
      torch.zeros_like(potential),
    )
    self._previous.copy_(potential)
    self._initialized.fill_(True)
    rate = delta / env.step_dt
    return rate.clamp(-max_regression_rate, max_progress_rate)


class RecoveryMilestoneLift:
  """Reward each new height-uprightness milestone at most once per episode."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self._best_progress = torch.zeros(env.num_envs, device=env.device)
    self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._initialized[ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    event_name: str,
    max_progress_rate: float,
  ) -> torch.Tensor:
    if max_progress_rate <= 0.0:
      raise ValueError("max_progress_rate must be positive.")
    state = get_g1_recovery_state(env, event_name)
    progress = state.current_progress.clamp(0.0, 1.0)
    improvement = torch.where(
      self._initialized,
      (progress - self._best_progress).clamp_min(0.0),
      torch.zeros_like(progress),
    )
    self._best_progress.copy_(torch.maximum(self._best_progress, progress))
    self._best_progress.copy_(
      torch.where(self._initialized, self._best_progress, progress)
    )
    self._initialized.fill_(True)
    active = (state.mode != STAND_MODE) & ~state.succeeded
    return (improvement / env.step_dt).clamp_max(max_progress_rate) * active.float()


def recovery_hold_reward(
  env: ManagerBasedRlEnv,
  event_name: str,
  gate_low: float,
  gate_high: float,
) -> torch.Tensor:
  """Densely reward approaching and holding the active success region."""
  state = get_g1_recovery_state(env, event_name)
  return recovery_gate(state.current_progress, gate_low, gate_high)


def gated_action_rate_l2(
  env: ManagerBasedRlEnv,
  event_name: str,
  gate_minimum: float,
  gate_low: float,
  gate_high: float,
) -> torch.Tensor:
  penalty = envs_mdp.action_rate_l2(env)
  return penalty * _gated_scale(env, event_name, gate_minimum, gate_low, gate_high)


def gated_self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float,
  event_name: str,
  gate_minimum: float,
  gate_low: float,
  gate_high: float,
) -> torch.Tensor:
  cost = mdp.self_collision_cost(env, sensor_name, force_threshold)
  return cost * _gated_scale(env, event_name, gate_minimum, gate_low, gate_high)


def gated_body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  event_name: str,
  gate_minimum: float,
  gate_low: float,
  gate_high: float,
) -> torch.Tensor:
  penalty = mdp.body_angular_velocity_penalty(env, asset_cfg)
  return penalty * _gated_scale(env, event_name, gate_minimum, gate_low, gate_high)


def gated_angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  event_name: str,
  gate_minimum: float,
  gate_low: float,
  gate_high: float,
) -> torch.Tensor:
  penalty = mdp.angular_momentum_penalty(env, sensor_name)
  return penalty * _gated_scale(env, event_name, gate_minimum, gate_low, gate_high)


def recovery_standing_potential(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  event_name: str,
  upright_floor: float,
  gate_low: float,
  gate_high: float,
) -> torch.Tensor:
  """Shape height even while sideways, then hand over to standing rewards."""
  asset = env.scene[asset_cfg.name]
  root_height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  nominal_height = asset.data.default_root_state[:, 2].clamp_min(1e-6)
  height = (root_height / nominal_height).clamp(0.0, 1.0)
  upright = (-asset.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
  shaped_upright = upright_floor + (1.0 - upright_floor) * upright
  progress = get_g1_recovery_state(env, event_name).current_progress
  return (1.0 - recovery_gate(progress, gate_low, gate_high)) * (
    height * shaped_upright
  )


class FallenDurationPenalty:
  """Increase urgency smoothly while a recovery state remains below success."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self.elapsed = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.elapsed[slice(None) if env_ids is None else env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    event_name: str,
    threshold: float,
    tau_s: float,
    max_penalty: float,
  ) -> torch.Tensor:
    state = get_g1_recovery_state(env, event_name)
    active = (state.mode != STAND_MODE) & (state.current_progress < threshold)
    self.elapsed.copy_(torch.where(active, self.elapsed + env.step_dt, 0.0))
    raw = torch.exp(self.elapsed / tau_s) - 1.0
    penalty = max_penalty * raw / (raw + max_penalty)
    return penalty * active.float()


def recovery_success_bonus(env: ManagerBasedRlEnv, event_name: str) -> torch.Tensor:
  """Return a true one-shot bonus after RewardManager applies dt scaling."""
  succeeded = get_g1_recovery_state(env, event_name).just_succeeded
  return succeeded.float() / env.step_dt


def recovery_failure_penalty(
  env: ManagerBasedRlEnv, event_name: str, timeout_name: str
) -> torch.Tensor:
  """Return a dt-corrected one-shot penalty for unsuccessful episode timeout."""
  state = get_g1_recovery_state(env, event_name)
  timed_out = env.termination_manager.get_term(timeout_name) & ~state.succeeded
  return timed_out.float() / env.step_dt


def recovery_succeeded(
  env: ManagerBasedRlEnv,
  event_name: str,
  threshold: float,
  hold_steps: int,
) -> torch.Tensor:
  """Terminate after height*uprightness holds for the configured duration."""
  state = get_g1_recovery_state(env, event_name)
  state.just_succeeded.zero_()
  ready = state.current_progress >= threshold
  state.hold_count.copy_(
    torch.where(
      ready,
      state.hold_count + 1,
      torch.zeros_like(state.hold_count),
    )
  )
  newly_succeeded = ~state.succeeded & (state.hold_count >= hold_steps)
  state.succeeded |= newly_succeeded
  state.just_succeeded |= newly_succeeded
  return newly_succeeded


def recovery_mode_metric(
  env: ManagerBasedRlEnv, event_name: str, mode: str
) -> torch.Tensor:
  state = get_g1_recovery_state(env, event_name)
  mode_id = {
    "reference": REFERENCE_MODE,
    "fallen": FALLEN_MODE,
    "stand": STAND_MODE,
  }[mode]
  return (state.mode == mode_id).float()


def recovery_mode_success_metric(
  env: ManagerBasedRlEnv, event_name: str, mode: str
) -> torch.Tensor:
  """Return the terminal success numerator for one reset mode."""
  state = get_g1_recovery_state(env, event_name)
  mode_id = {
    "reference": REFERENCE_MODE,
    "fallen": FALLEN_MODE,
    "stand": STAND_MODE,
  }[mode]
  return ((state.mode == mode_id) & state.succeeded).float()


def recovery_progress_bin_metric(
  env: ManagerBasedRlEnv,
  event_name: str,
  lower: float,
  upper: float,
) -> torch.Tensor:
  """Return the terminal denominator for one initial-progress bin."""
  if not 0.0 <= lower < upper <= 1.0:
    raise ValueError("Recovery progress bins must satisfy 0 <= lower < upper <= 1.")
  state = get_g1_recovery_state(env, event_name)
  return (
    (state.mode != STAND_MODE)
    & (state.reset_progress >= lower)
    & (state.reset_progress < upper)
  ).float()


def recovery_progress_bin_success_metric(
  env: ManagerBasedRlEnv,
  event_name: str,
  lower: float,
  upper: float,
) -> torch.Tensor:
  """Return the terminal success numerator for one initial-progress bin."""
  state = get_g1_recovery_state(env, event_name)
  in_bin = recovery_progress_bin_metric(env, event_name, lower, upper).bool()
  return (in_bin & state.succeeded).float()


def recovery_progress_metric(env: ManagerBasedRlEnv, event_name: str) -> torch.Tensor:
  return get_g1_recovery_state(env, event_name).current_progress


def recovery_assistance_metric(env: ManagerBasedRlEnv, event_name: str) -> torch.Tensor:
  return get_g1_recovery_state(env, event_name).applied_force
