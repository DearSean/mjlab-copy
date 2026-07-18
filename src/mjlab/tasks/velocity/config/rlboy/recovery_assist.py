"""Fallen-recovery assistance curriculum for the RL_BOY velocity task."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.posture_phase import (
  PosturePhaseEstimator,
  PosturePhaseEstimatorCfg,
)
from mjlab.utils.string import resolve_expr

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


RECOVERY_ASSIST_EVENT_NAME = "recovery_assist"


def _selected_names(names: tuple[str, ...], ids: list[int] | slice) -> tuple[str, ...]:
  if isinstance(ids, slice):
    return names[ids]
  return tuple(names[index] for index in ids)


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


class actuator_torque_limit_excess_penalty:
  """Penalize actuator torques only after they exceed configured limits."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._asset: Entity = env.scene[self._asset_cfg.name]
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
    self._threshold_ratio: float = cfg.params.get("threshold_ratio", 1.0)
    if self._threshold_ratio < 0.0:
      raise ValueError("threshold_ratio cannot be negative.")
    self._log_prefix: str | None = cfg.params.get("log_prefix")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    limit_by_actuator: dict[str, float],
    threshold_ratio: float = 1.0,
    log_prefix: str | None = None,
  ) -> torch.Tensor:
    del asset_cfg, limit_by_actuator, threshold_ratio, log_prefix
    torques = self._asset.data.actuator_force[:, self._asset_cfg.actuator_ids]
    ratio = torch.abs(torques) / self._limits
    excess = torch.clamp(ratio - self._threshold_ratio, min=0.0)
    if self._log_prefix is not None:
      env.extras["log"][f"Metrics/{self._log_prefix}/max_ratio"] = ratio.max()
      env.extras["log"][f"Metrics/{self._log_prefix}/active_fraction"] = (
        (excess > 0.0).float().mean()
      )
    return torch.sum(torch.square(excess), dim=1)


class RlBoyRecoveryAssist:
  """Manage recovery-group resets, upward assistance, and recovery outcomes."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self._env = env
    self._asset: Entity = env.scene[params["asset_cfg"].name]
    self._body_ids = params["asset_cfg"].body_ids
    self._poses: list[dict[str, tuple[float, ...]]] = params["poses"]
    self._recovery_stage_probabilities: tuple[float, ...] = params[
      "recovery_stage_probabilities"
    ]
    self._post_stage_recovery_probability: float = params[
      "post_stage_recovery_probability"
    ]
    self._low_force_recovery_probability: float = params[
      "low_force_recovery_probability"
    ]
    self._recovery_probability_limits: tuple[float, float] = params[
      "recovery_probability_limits"
    ]
    self._recovery_probability_feedback_gain: float = params[
      "recovery_probability_feedback_gain"
    ]
    self._recovery_probability_smoothing: float = params[
      "recovery_probability_smoothing"
    ]
    self._recovery_probability_min_attempts: int = params[
      "recovery_probability_min_attempts"
    ]
    self._angle_noise_ramp_attempts: int = params["angle_noise_ramp_attempts"]
    self._pose_stage_source_weights = torch.tensor(
      params["pose_stage_source_weights"], device=env.device, dtype=torch.float32
    )
    self._force_ranges = torch.tensor(
      params["force_ranges"], device=env.device, dtype=torch.float32
    )
    self._posture_estimator = PosturePhaseEstimator(params["posture_cfg"])
    self._fall_confirm_s: float = params["fall_confirm_s"]
    self._upright_hold_s: float = params["upright_hold_s"]
    self._force_ramp_up_s: float = params["force_ramp_up_s"]
    self._force_ramp_down_s: float = params["force_ramp_down_s"]
    self._recovery_timeout_s: float = params["recovery_timeout_s"]
    self._root_height_range: tuple[float, float] = params["root_height_range"]
    self._root_lin_vel_range: tuple[float, float] = params["root_lin_vel_range"]
    self._root_ang_vel_range: tuple[float, float] = params["root_ang_vel_range"]
    joint_position_ranges = resolve_expr(
      params["joint_position_ranges"], self._asset.joint_names, (0.0, 0.0)
    )
    joint_velocity_ranges = resolve_expr(
      params["joint_velocity_ranges"], self._asset.joint_names, (0.0, 0.0)
    )
    self._joint_position_ranges = torch.tensor(
      joint_position_ranges, device=env.device, dtype=torch.float32
    )
    self._joint_velocity_ranges = torch.tensor(
      joint_velocity_ranges, device=env.device, dtype=torch.float32
    )
    csv_joint_names: tuple[str, ...] = params["csv_joint_names"]
    if len(csv_joint_names) != 20:
      raise ValueError("csv_joint_names must contain exactly 20 joint names.")
    missing_joint_names = set(csv_joint_names) - set(self._asset.joint_names)
    if missing_joint_names:
      raise ValueError(f"CSV joints not found in robot: {sorted(missing_joint_names)}")
    self._csv_joint_ids = torch.tensor(
      [self._asset.joint_names.index(name) for name in csv_joint_names],
      device=env.device,
      dtype=torch.long,
    )
    frame_dir = Path(params["frame_dir"])
    self._frame_files: tuple[str, ...] = params["frame_files"]
    self._csv_frames = tuple(
      self._load_frames(frame_dir, frame_file) for frame_file in self._frame_files
    )
    self._canonical_source = len(self._csv_frames)
    self._source_names: tuple[str, ...] = params.get(
      "source_names", (*self._frame_files, "canonical")
    )

    self.level = 0
    self.pose_stage = 0
    self._recovery_probability = self._recovery_stage_probabilities[0]
    self._target_recovery_probability = self._recovery_probability
    self.starts_fallen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.assist_active = torch.zeros_like(self.starts_fallen)
    self.fallen_detected = torch.zeros_like(self.starts_fallen)
    self.succeeded = torch.zeros_like(self.starts_fallen)
    self.just_succeeded = torch.zeros_like(self.starts_fallen)
    self.elapsed_s = torch.zeros(env.num_envs, device=env.device)
    self.fall_confirm_s = torch.zeros_like(self.elapsed_s)
    self.upright_hold_s = torch.zeros_like(self.elapsed_s)
    self.sampled_force = torch.zeros_like(self.elapsed_s)
    self.applied_force = torch.zeros_like(self.elapsed_s)
    self.sample_source = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )

    if len(self._poses) == 0:
      raise ValueError("At least one canonical fallen pose is required.")
    if self._angle_noise_ramp_attempts <= 0:
      raise ValueError("angle_noise_ramp_attempts must be positive.")
    if len(self._recovery_stage_probabilities) != len(self._pose_stage_source_weights):
      raise ValueError(
        "recovery_stage_probabilities must match pose_stage_source_weights."
      )
    probability_values = (
      *self._recovery_stage_probabilities,
      self._post_stage_recovery_probability,
      self._low_force_recovery_probability,
      *self._recovery_probability_limits,
    )
    if any(not 0.0 <= probability <= 1.0 for probability in probability_values):
      raise ValueError("Recovery probabilities must be between zero and one.")
    if self._recovery_probability_limits[0] > self._recovery_probability_limits[1]:
      raise ValueError("recovery_probability_limits must be ordered.")
    if not 0.0 < self._recovery_probability_smoothing <= 1.0:
      raise ValueError("recovery_probability_smoothing must be in (0, 1].")
    if self._recovery_probability_feedback_gain < 0.0:
      raise ValueError("recovery_probability_feedback_gain cannot be negative.")
    if self._recovery_probability_min_attempts <= 0:
      raise ValueError("recovery_probability_min_attempts must be positive.")
    expected_source_count = len(self._csv_frames) + 1
    if len(self._source_names) != expected_source_count:
      raise ValueError("source_names must match CSV sources plus canonical poses.")
    if self._pose_stage_source_weights.shape != (
      len(self._recovery_stage_probabilities),
      expected_source_count,
    ):
      raise ValueError(
        "pose_stage_source_weights must have one row per pose stage and "
        "one column per CSV source plus canonical poses."
      )
    if bool((self._pose_stage_source_weights < 0.0).any()):
      raise ValueError("pose_stage_source_weights cannot contain negative values.")
    if not torch.allclose(
      self._pose_stage_source_weights.sum(dim=1),
      torch.ones(len(self._pose_stage_source_weights), device=env.device),
    ):
      raise ValueError("Each pose_stage_source_weights row must sum to one.")
    self.attempts = torch.zeros((), device=env.device, dtype=torch.long)
    self.successes = torch.zeros_like(self.attempts)
    self.source_samples = torch.zeros(
      expected_source_count, device=env.device, dtype=torch.long
    )

  def _load_frames(self, frame_dir: Path, pattern: str) -> torch.Tensor:
    """Load and validate root pose plus joint position CSV frames once."""
    rows: list[list[float]] = []
    paths = sorted(frame_dir.glob(pattern))
    if not paths:
      raise ValueError(f"No recovery frames match {frame_dir / pattern}.")
    for path in paths:
      with path.open(newline="", encoding="utf-8") as csv_file:
        for line_number, row in enumerate(csv.reader(csv_file), start=1):
          if len(row) != 27:
            raise ValueError(
              f"{path}:{line_number} has {len(row)} columns; expected 27."
            )
          rows.append([float(value) for value in row])
    return torch.tensor(rows, device=self._env.device, dtype=torch.float32)

  @property
  def target_force(self) -> float:
    """Return the midpoint of the active force range for display purposes."""
    return float(self._force_ranges[self.level].mean().item())

  @property
  def force_range(self) -> tuple[float, float]:
    force_range = self._force_ranges[self.level]
    return float(force_range[0].item()), float(force_range[1].item())

  @property
  def recovery_timeout_s(self) -> float:
    return self._recovery_timeout_s

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Apply fallen poses after all reset-mode randomization has completed."""
    if env_ids is None:
      env_ids = torch.arange(
        self._env.num_envs, device=self._env.device, dtype=torch.long
      )
    self.just_succeeded[env_ids] = False

    recovery_ids = env_ids[self.starts_fallen[env_ids]]
    if len(recovery_ids) > 0:
      root_pos, root_quat, csv_joint_pos = self._sample_initial_states(recovery_ids)
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
      root_state[:, 7:10].uniform_(*self._root_lin_vel_range)
      root_state[:, 10:13].uniform_(*self._root_ang_vel_range)
      self._asset.write_root_state_to_sim(root_state, env_ids=recovery_ids)

      default_joint_pos = self._asset.data.default_joint_pos
      default_joint_vel = self._asset.data.default_joint_vel
      soft_joint_pos_limits = self._asset.data.soft_joint_pos_limits
      assert default_joint_pos is not None
      assert default_joint_vel is not None
      assert soft_joint_pos_limits is not None
      joint_pos = default_joint_pos[recovery_ids].clone()
      csv_mask = self.sample_source[recovery_ids] != self._canonical_source
      if bool(csv_mask.any()):
        csv_rows = csv_mask.nonzero(as_tuple=False).squeeze(-1)
        joint_pos[csv_rows[:, None], self._csv_joint_ids[None, :]] = csv_joint_pos[
          csv_rows
        ]
      joint_vel = default_joint_vel[recovery_ids].clone()
      position_noise = self._sample_joint_position_noise(joint_pos)
      velocity_noise = torch.rand_like(joint_vel)
      velocity_noise = self._joint_velocity_ranges[:, 0] + velocity_noise * (
        self._joint_velocity_ranges[:, 1] - self._joint_velocity_ranges[:, 0]
      )
      joint_pos += position_noise
      limits = soft_joint_pos_limits[recovery_ids]
      joint_pos.clamp_(limits[..., 0], limits[..., 1])
      joint_vel += velocity_noise
      self._asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=recovery_ids)
      force_min, force_max = self.force_range
      self.sampled_force[recovery_ids] = (
        torch.rand(len(recovery_ids), device=self._env.device) * (force_max - force_min)
        + force_min
      )

    self._write_force(env_ids)

  def _sample_initial_states(
    self, recovery_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample curriculum-dependent root and joint states."""
    count = len(recovery_ids)
    sources = self.sample_source[recovery_ids]
    root_pos = torch.zeros(count, 3, device=self._env.device)
    root_quat = torch.zeros(count, 4, device=self._env.device)
    csv_joint_pos = torch.zeros(count, 20, device=self._env.device)

    for source, frames in enumerate(self._csv_frames):
      mask = sources == source
      sample_count = int(mask.sum().item())
      if sample_count == 0:
        continue
      frame_ids = torch.randint(len(frames), (sample_count,), device=self._env.device)
      sampled = frames[frame_ids]
      root_pos[mask, 2] = sampled[:, 2]
      # CSV stores (qx, qy, qz, qw); simulation expects (qw, qx, qy, qz).
      quat_xyzw = sampled[:, 3:7]
      root_quat[mask] = quat_xyzw[:, (3, 0, 1, 2)]
      csv_joint_pos[mask] = sampled[:, 7:27]

    canonical_mask = sources == self._canonical_source
    canonical_count = int(canonical_mask.sum().item())
    if canonical_count > 0:
      pose_ids = torch.randint(
        len(self._poses), (canonical_count,), device=self._env.device
      )
      canonical_pos = torch.tensor(
        [pose["pos"] for pose in self._poses],
        device=self._env.device,
        dtype=torch.float32,
      )
      canonical_quat = torch.tensor(
        [pose["quat"] for pose in self._poses],
        device=self._env.device,
        dtype=torch.float32,
      )
      root_pos[canonical_mask] = canonical_pos[pose_ids]
      root_pos[canonical_mask, 2].uniform_(*self._root_height_range)
      root_quat[canonical_mask] = canonical_quat[pose_ids]

    root_quat /= root_quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return root_pos, root_quat, csv_joint_pos

  def _sample_joint_position_noise(self, joint_pos: torch.Tensor) -> torch.Tensor:
    """Apply no angle noise in stage one and smoothly enable it in stage two."""
    if self.pose_stage == 0:
      return torch.zeros_like(joint_pos)
    noise = torch.rand_like(joint_pos)
    noise = self._joint_position_ranges[:, 0] + noise * (
      self._joint_position_ranges[:, 1] - self._joint_position_ranges[:, 0]
    )
    if self.pose_stage == 1:
      noise *= self._angle_noise_scale()
    return noise

  def _angle_noise_scale(self) -> float:
    if self.pose_stage == 0:
      return 0.0
    if self.pose_stage >= 2:
      return 1.0
    success_rate = float((self.successes.float() / self.attempts.clamp_min(1)).item())
    performance_scale = min(max((success_rate - 0.5) / 0.4, 0.0), 1.0)
    evidence_scale = min(
      float(self.attempts.item()) / self._angle_noise_ramp_attempts, 1.0
    )
    return performance_scale * evidence_scale

  def prepare_group(self, env_ids: torch.Tensor) -> None:
    """Choose episode groups before reset-mode randomization terms run."""
    recovery_mask = (
      torch.rand(len(env_ids), device=self._env.device) < self._recovery_probability
    )
    recovery_ids = env_ids[recovery_mask]

    self.starts_fallen[env_ids] = False
    self.starts_fallen[recovery_ids] = True
    self.assist_active[env_ids] = False
    self.assist_active[recovery_ids] = True
    self.fallen_detected[env_ids] = False
    self.fallen_detected[recovery_ids] = True
    self.succeeded[env_ids] = False
    self.just_succeeded[env_ids] = False
    self.elapsed_s[env_ids] = 0.0
    self.fall_confirm_s[env_ids] = 0.0
    self.upright_hold_s[env_ids] = 0.0
    self.sampled_force[env_ids] = 0.0
    self.applied_force[env_ids] = 0.0
    self.sample_source[env_ids] = -1
    if len(recovery_ids) > 0:
      self.sample_source[recovery_ids] = torch.multinomial(
        self._pose_stage_source_weights[self.pose_stage],
        len(recovery_ids),
        replacement=True,
      )
      self.source_samples += torch.bincount(
        self.sample_source[recovery_ids], minlength=len(self.source_samples)
      ).to(self.source_samples)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    poses: list[dict[str, tuple[float, ...]]],
    recovery_stage_probabilities: tuple[float, ...],
    post_stage_recovery_probability: float,
    low_force_recovery_probability: float,
    recovery_probability_limits: tuple[float, float],
    recovery_probability_feedback_gain: float,
    recovery_probability_smoothing: float,
    recovery_probability_min_attempts: int,
    angle_noise_ramp_attempts: int,
    frame_dir: str,
    frame_files: tuple[str, ...],
    source_names: tuple[str, ...],
    csv_joint_names: tuple[str, ...],
    pose_stage_source_weights: tuple[tuple[float, ...], ...],
    force_ranges: tuple[tuple[float, float], ...],
    posture_cfg: PosturePhaseEstimatorCfg,
    fall_confirm_s: float,
    upright_hold_s: float,
    force_ramp_up_s: float,
    force_ramp_down_s: float,
    recovery_timeout_s: float,
    root_height_range: tuple[float, float],
    root_lin_vel_range: tuple[float, float],
    root_ang_vel_range: tuple[float, float],
    joint_position_ranges: dict[str, tuple[float, float]],
    joint_velocity_ranges: dict[str, tuple[float, float]],
  ) -> None:
    dt = env.step_dt
    self.just_succeeded.zero_()
    del (
      env_ids,
      asset_cfg,
      poses,
      recovery_stage_probabilities,
      post_stage_recovery_probability,
      low_force_recovery_probability,
      recovery_probability_limits,
      recovery_probability_feedback_gain,
      recovery_probability_smoothing,
      recovery_probability_min_attempts,
      angle_noise_ramp_attempts,
      frame_dir,
      frame_files,
      source_names,
      csv_joint_names,
      pose_stage_source_weights,
      force_ranges,
      posture_cfg,
      fall_confirm_s,
      upright_hold_s,
      force_ramp_up_s,
      force_ramp_down_s,
      recovery_timeout_s,
      root_height_range,
      root_lin_vel_range,
      root_ang_vel_range,
      joint_position_ranges,
      joint_velocity_ranges,
    )

    posture = self._posture_estimator.estimate(self._env)
    fallen = posture.needs_recovery
    self.fallen_detected.copy_(fallen)
    confirming = fallen & ~self.assist_active
    self.fall_confirm_s = torch.where(
      confirming, self.fall_confirm_s + dt, torch.zeros_like(self.fall_confirm_s)
    )
    newly_active = confirming & (self.fall_confirm_s >= self._fall_confirm_s)
    force_min, force_max = self.force_range
    sampled_force = (
      torch.rand(env.num_envs, device=env.device) * (force_max - force_min) + force_min
    )
    self.sampled_force[newly_active] = sampled_force[newly_active]
    self.assist_active[newly_active] = True
    self.succeeded[newly_active] = False
    self.elapsed_s[newly_active] = 0.0
    self.upright_hold_s[newly_active] = 0.0

    active = self.assist_active & ~self.succeeded
    self.elapsed_s[active] += dt
    upright = posture.is_ready

    desired_force = torch.where(
      active & ~upright, self.sampled_force, torch.zeros_like(self.sampled_force)
    )
    ramp_reference = self.sampled_force.clamp_min(1.0)
    ramp_up_step = ramp_reference * dt / max(self._force_ramp_up_s, dt)
    ramp_down_step = ramp_reference * dt / max(self._force_ramp_down_s, dt)
    force_error = desired_force - self.applied_force
    force_step = torch.where(
      force_error >= 0.0,
      torch.minimum(force_error, ramp_up_step),
      torch.maximum(force_error, -ramp_down_step),
    )
    self.applied_force += force_step

    independently_upright = active & upright & (self.applied_force <= 1e-3)
    self.upright_hold_s = torch.where(
      independently_upright,
      self.upright_hold_s + dt,
      torch.zeros_like(self.upright_hold_s),
    )
    newly_succeeded = (
      active
      & (self.upright_hold_s >= self._upright_hold_s)
      & (self.applied_force <= 1e-3)
    )
    self.succeeded[newly_succeeded] = True
    self.just_succeeded[newly_succeeded] = True
    self.assist_active[newly_succeeded] = False
    self.applied_force[newly_succeeded] = 0.0

    self._write_force()

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
    """Accumulate completed controlled recovery attempts globally."""
    recovery_ids = env_ids[self.starts_fallen[env_ids]]
    if len(recovery_ids) == 0:
      return
    self.attempts += len(recovery_ids)
    self.successes += self.succeeded[recovery_ids].sum()

  def update_level(
    self,
    window_size: int,
    success_threshold: float,
  ) -> None:
    attempts = int(self.attempts.item())
    if attempts < window_size:
      return
    if int(self.successes.item()) >= success_threshold * attempts:
      if self.pose_stage < len(self._pose_stage_source_weights) - 1:
        self.pose_stage += 1
      elif self.level < len(self._force_ranges) - 1:
        self.level += 1
    self.attempts.zero_()
    self.successes.zero_()

  def update_recovery_probability(self, success_target: float) -> None:
    """Smoothly adapt the next-reset recovery population around its stage base."""
    base_probability = self._base_recovery_probability()
    target_probability = base_probability
    attempts = int(self.attempts.item())
    if attempts >= self._recovery_probability_min_attempts:
      success_rate = float((self.successes.float() / self.attempts).item())
      feedback = self._recovery_probability_feedback_gain * (
        success_target - success_rate
      )
      target_probability += min(max(feedback, -0.1), 0.1)
    target_probability = min(
      max(target_probability, self._recovery_probability_limits[0]),
      self._recovery_probability_limits[1],
    )
    self._target_recovery_probability = target_probability
    alpha = self._recovery_probability_smoothing
    self._recovery_probability = (
      1.0 - alpha
    ) * self._recovery_probability + alpha * target_probability

  def _base_recovery_probability(self) -> float:
    if self.pose_stage < len(self._recovery_stage_probabilities) - 1:
      return self._recovery_stage_probabilities[self.pose_stage]
    if self.level == 0:
      return self._recovery_stage_probabilities[-1]
    if self.level >= len(self._force_ranges) - 2:
      return self._low_force_recovery_probability
    return self._post_stage_recovery_probability

  def curriculum_state(self) -> dict[str, torch.Tensor]:
    attempts = self.attempts
    rate = self.successes.sum().float() / attempts.clamp_min(1)
    force_range = self._force_ranges[self.level]
    recovery_count = self.assist_active.sum().clamp_min(1)
    actual_mean = (
      self.applied_force * self.assist_active.float()
    ).sum() / recovery_count
    state = {
      "level": torch.tensor(self.level, device=self._env.device),
      "pose_stage": torch.tensor(self.pose_stage, device=self._env.device),
      "force_n": force_range.mean(),
      "force_min_n": force_range[0],
      "force_max_n": force_range[1],
      "actual_force_mean_n": actual_mean,
      "attempts": attempts,
      "success_rate": rate,
      "recovery_probability": torch.tensor(
        self._recovery_probability, device=self._env.device
      ),
      "target_recovery_probability": torch.tensor(
        self._target_recovery_probability, device=self._env.device
      ),
      "angle_noise_scale": torch.tensor(
        self._angle_noise_scale(), device=self._env.device
      ),
    }
    for index, source_name in enumerate(self._source_names):
      state[f"{source_name}_samples"] = self.source_samples[index]
    return state


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
) -> dict[str, torch.Tensor]:
  """Update assistance using outcomes from fallen-recovery environments only."""
  if isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  assist = _get_assist(env, event_name)
  assist.record_outcomes(env_ids)
  assist.update_recovery_probability(success_threshold)
  assist.update_level(window_size, success_threshold)
  return assist.curriculum_state()


def recovery_assist_reward_weight_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  event_name: str,
  assist_level: int,
  assist_weights: dict[str, float],
  complete_weights: dict[str, float],
) -> dict[str, torch.Tensor]:
  """Stage reward weights based on recovery assistance progress."""
  del env_ids
  assist = _get_assist(env, event_name)
  force_range = assist._force_ranges[assist.level]
  complete = (
    assist.pose_stage >= 2
    and assist.level >= len(assist._force_ranges) - 1
    and bool(torch.all(force_range == 0.0).item())
  )
  active_weights = complete_weights if complete else {}
  if assist.pose_stage >= 2 and assist.level >= assist_level and not complete:
    active_weights = assist_weights
  applied_weights: dict[str, torch.Tensor] = {
    "active": torch.tensor(float(bool(active_weights)), device=env.device),
    "complete": torch.tensor(float(complete), device=env.device),
  }
  reward_names = set(assist_weights) | set(complete_weights)
  for reward_name in reward_names:
    weight = active_weights.get(reward_name, 0.0)
    env.reward_manager.get_term_cfg(reward_name).weight = weight
    applied_weights[f"{reward_name}_weight"] = torch.tensor(weight, device=env.device)
  return applied_weights


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
  recovery_ids = env_ids[assist.starts_fallen[env_ids]]
  normal_ids = env_ids[~assist.starts_fallen[env_ids]]
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
  normal_ids = env_ids[
    ~assist.starts_fallen[env_ids]
    & ~assist.fallen_detected[env_ids]
    & ~assist.assist_active[env_ids]
  ]
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
  env.event_manager.get_term_cfg(push_event_name).interval_range_s = stage[
    "push_interval_s"
  ]
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
  return fell & ~_get_assist(env, event_name).assist_active


def recovery_succeeded(
  env: ManagerBasedRlEnv,
  event_name: str,
) -> torch.Tensor:
  """End a recovery episode once the configured root height is reached."""
  return (
    _get_assist(env, event_name).starts_fallen & _get_assist(env, event_name).succeeded
  )


def recovery_timed_out(
  env: ManagerBasedRlEnv,
  event_name: str,
) -> torch.Tensor:
  """Terminate recovery episodes that do not stand before their deadline."""
  assist = _get_assist(env, event_name)
  return (
    assist.assist_active
    & ~assist.succeeded
    & (assist.elapsed_s >= assist.recovery_timeout_s)
  )


def recovery_failure_penalty(
  env: ManagerBasedRlEnv,
  event_name: str,
) -> torch.Tensor:
  """Return a one-shot failure cost corrected for reward-rate dt scaling."""
  return recovery_timed_out(env, event_name).float() / env.step_dt


def recovery_success_bonus(
  env: ManagerBasedRlEnv,
  event_name: str,
) -> torch.Tensor:
  """Return a one-shot success reward corrected for reward-rate dt scaling."""
  return _get_assist(env, event_name).just_succeeded.float() / env.step_dt


def recovery_mask(env: ManagerBasedRlEnv, event_name: str) -> torch.Tensor:
  """Return the dynamic assistance mask."""
  return _get_assist(env, event_name).assist_active
