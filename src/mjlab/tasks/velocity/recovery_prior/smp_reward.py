"""Frozen online SMP reward for G1 recovery PPO."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.recovery_data.g1_schema import G1_JOINT_NAMES
from mjlab.tasks.velocity.recovery_prior.g1_smp_data import (
  G1_SMP_FEATURE_DIM,
  G1_SMP_FEATURE_SCHEMA_VERSION,
)
from mjlab.tasks.velocity.recovery_prior.smp_model import SmpDenoiser, SmpDenoiserCfg
from mjlab.utils.buffers import CircularBuffer
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


REFERENCE_MODE = 0
FALLEN_MODE = 1


def normalized_esm_reward(
  errors: torch.Tensor, calibration_means: torch.Tensor, scale: float
) -> torch.Tensor:
  """Convert per-timestep ESM errors into a fixed-scale reward."""
  return torch.exp(-normalized_esm_energy(errors, calibration_means, scale))


def normalized_esm_energy(
  errors: torch.Tensor, calibration_means: torch.Tensor, scale: float
) -> torch.Tensor:
  """Convert per-timestep ESM errors into calibrated diffusion energy."""
  normalized = errors / calibration_means[:, None].clamp_min(1e-6)
  return scale * normalized.mean(dim=0)


def ood_score_gate(
  score: torch.Tensor, full_below: float, zero_above: float
) -> torch.Tensor:
  """Gate return guidance on out-of-distribution SMP scores."""
  if not 0.0 <= full_below < zero_above <= 1.0:
    raise ValueError("OOD score bounds must satisfy 0 <= full < zero <= 1.")
  ratio = ((zero_above - score) / (zero_above - full_below)).clamp(0.0, 1.0)
  return ratio.square() * (3.0 - 2.0 * ratio)


def energy_descent_signal(
  previous: torch.Tensor,
  current: torch.Tensor,
  step_dt: float,
  max_rate: float,
) -> torch.Tensor:
  """Return a bounded signed signal for motion toward lower SMP energy."""
  if step_dt <= 0.0:
    raise ValueError("step_dt must be positive.")
  if max_rate <= 0.0:
    raise ValueError("max_rate must be positive.")
  rate = (previous - current) / step_dt
  return (rate / max_rate).clamp(-1.0, 1.0)


def directional_progress_signal(
  feature_delta: torch.Tensor,
  previous_direction: torch.Tensor,
  minimum_motion_rms: float,
) -> torch.Tensor:
  """Measure motion alignment with the previous denoising direction."""
  if minimum_motion_rms <= 0.0:
    raise ValueError("minimum_motion_rms must be positive.")
  delta_norm = torch.linalg.vector_norm(feature_delta, dim=-1)
  direction_norm = torch.linalg.vector_norm(previous_direction, dim=-1)
  cosine = torch.sum(feature_delta * previous_direction, dim=-1) / (
    delta_norm * direction_norm
  ).clamp_min(1e-6)
  motion_rms = torch.mean(feature_delta.square(), dim=-1).sqrt()
  motion_gate = (motion_rms / minimum_motion_rms).clamp(0.0, 1.0)
  valid_direction = direction_norm > 1e-6
  return torch.where(valid_direction, cosine.clamp(-1.0, 1.0), 0.0) * motion_gate


def recovery_guidance_gate(
  progress: torch.Tensor, low: float, high: float
) -> torch.Tensor:
  """Fade directional SMP guidance before terminal standing posture."""
  if not 0.0 <= low < high <= 1.0:
    raise ValueError("SMP handoff bounds must satisfy 0 <= low < high <= 1.")
  ratio = ((progress - low) / (high - low)).clamp(0.0, 1.0)
  smooth = ratio.square() * (3.0 - 2.0 * ratio)
  return 1.0 - smooth


def progress_handoff_weight(
  progress: torch.Tensor,
  recovery_weight: float,
  terminal_weight: float,
  low: float,
  high: float,
) -> torch.Tensor:
  """Smoothly hand reward weight from recovery motion to terminal posture."""
  if not 0.0 <= terminal_weight <= recovery_weight:
    raise ValueError("SMP weights must satisfy 0 <= terminal <= recovery.")
  if not 0.0 <= low < high <= 1.0:
    raise ValueError("SMP handoff bounds must satisfy 0 <= low < high <= 1.")
  ratio = ((progress - low) / (high - low)).clamp(0.0, 1.0)
  gate = ratio.square() * (3.0 - 2.0 * ratio)
  return recovery_weight + (terminal_weight - recovery_weight) * gate


def update_landing_contact_hold(
  awaiting_landing: torch.Tensor,
  contact_steps: torch.Tensor,
  ground_contact: torch.Tensor,
  required_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Debounce whole-body ground contact before starting an SMP history."""
  if required_steps <= 0:
    raise ValueError("required_steps must be positive.")
  held = torch.where(
    awaiting_landing & ground_contact,
    contact_steps + 1,
    torch.zeros_like(contact_steps),
  )
  landed = awaiting_landing & (held >= required_steps)
  return held, landed


class G1SmpReward:
  """Score simulated 10-step motion windows with a frozen EMA denoiser."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    params = cfg.params
    self._env = env
    self._asset: Entity = env.scene[params["asset_cfg"].name]
    if tuple(self._asset.joint_names) != G1_JOINT_NAMES:
      raise ValueError("G1 SMP joint order does not match the simulator entity.")

    checkpoint_path = Path(params["checkpoint"])
    checkpoint = torch.load(
      checkpoint_path, map_location=env.device, weights_only=False
    )
    if checkpoint.get("schema_version") != params["checkpoint_schema_version"]:
      raise ValueError(f"Unsupported SMP checkpoint schema in {checkpoint_path}.")
    model_cfg = SmpDenoiserCfg(**checkpoint["model_cfg"])
    if model_cfg.feature_dim != G1_SMP_FEATURE_DIM or model_cfg.window_size != 10:
      raise ValueError("SMP checkpoint must use the G1 10x51 feature schema.")
    self._window_size = model_cfg.window_size
    self._model = SmpDenoiser(model_cfg).to(env.device)
    self._model.load_state_dict(checkpoint["ema_state_dict"])
    self._model.eval()
    for parameter in self._model.parameters():
      parameter.requires_grad_(False)

    normalizer = np.load(Path(params["normalizer_file"]))
    schema = str(normalizer["feature_schema_version"])
    if schema != G1_SMP_FEATURE_SCHEMA_VERSION:
      raise ValueError(f"Unsupported SMP feature normalizer schema {schema!r}.")
    self._mean = torch.as_tensor(normalizer["mean"], device=env.device)
    self._std = torch.as_tensor(normalizer["std"], device=env.device)
    if self._mean.shape != (G1_SMP_FEATURE_DIM,) or self._std.shape != (
      G1_SMP_FEATURE_DIM,
    ):
      raise ValueError("SMP normalizer must contain 51-dimensional statistics.")

    self._event_name: str = params["event_name"]
    landing_sensor = env.scene[params["landing_sensor_name"]]
    if not isinstance(landing_sensor, ContactSensor):
      raise TypeError("SMP landing_sensor_name must select a ContactSensor.")
    if landing_sensor.data.found is None:
      raise ValueError("SMP landing sensor must provide the 'found' field.")
    self._landing_sensor = landing_sensor
    self._landing_contact_hold_steps = int(params["landing_contact_hold_steps"])
    update_landing_contact_hold(
      torch.zeros(1, dtype=torch.bool, device=env.device),
      torch.zeros(1, dtype=torch.long, device=env.device),
      torch.zeros(1, dtype=torch.bool, device=env.device),
      self._landing_contact_hold_steps,
    )
    self._timesteps = torch.tensor(
      params["esm_timesteps"], device=env.device, dtype=torch.long
    )
    self._calibration_means = torch.tensor(
      params["esm_error_means"], device=env.device, dtype=torch.float32
    )
    if self._calibration_means.shape != self._timesteps.shape:
      raise ValueError("esm_error_means must match esm_timesteps.")
    if bool((self._calibration_means <= 0.0).any()):
      raise ValueError("esm_error_means must be positive.")
    if bool((self._timesteps < 0).any()) or bool(
      (self._timesteps >= model_cfg.diffusion_steps).any()
    ):
      raise ValueError("esm_timesteps are outside the diffusion schedule.")
    beta = torch.linspace(1e-4, 0.02, model_cfg.diffusion_steps, device=env.device)
    self._alpha_bar = torch.cumprod(1.0 - beta, dim=0)
    self._scale = float(params["smp_scale"])
    self._reward_weight = float(params["reward_weight"])
    self._terminal_reward_weight = float(params["terminal_reward_weight"])
    self._handoff_low, self._handoff_high = (
      float(value) for value in params["handoff_progress"]
    )
    progress_handoff_weight(
      torch.zeros(1, device=env.device),
      self._reward_weight,
      self._terminal_reward_weight,
      self._handoff_low,
      self._handoff_high,
    )
    self._nominal_height = float(params["nominal_height_m"])
    self._energy_descent_weight = float(params["energy_descent_weight"])
    self._energy_descent_max_rate = float(params["energy_descent_max_rate"])
    self._direction_weight = float(params["direction_weight"])
    self._direction_minimum_motion_rms = float(params["direction_minimum_motion_rms"])
    self._ood_score_full_below, self._ood_score_zero_above = (
      float(value) for value in params["ood_score_range"]
    )
    if self._energy_descent_weight < 0.0 or self._direction_weight < 0.0:
      raise ValueError("SMP guidance weights must be non-negative.")
    energy_descent_signal(
      torch.zeros(1, device=env.device),
      torch.zeros(1, device=env.device),
      env.step_dt,
      self._energy_descent_max_rate,
    )
    directional_progress_signal(
      torch.zeros(1, G1_SMP_FEATURE_DIM, device=env.device),
      torch.zeros(1, G1_SMP_FEATURE_DIM, device=env.device),
      self._direction_minimum_motion_rms,
    )
    ood_score_gate(
      torch.zeros(1, device=env.device),
      self._ood_score_full_below,
      self._ood_score_zero_above,
    )

    body_ids, body_names = self._asset.find_bodies(
      ("left_wrist_yaw_link", "right_wrist_yaw_link"), preserve_order=True
    )
    site_ids, site_names = self._asset.find_sites(
      ("left_foot", "right_foot"), preserve_order=True
    )
    if len(body_ids) != 2 or len(site_ids) != 2:
      raise ValueError(
        "G1 SMP requires both wrist bodies and both foot sites; got "
        f"{body_names} and {site_names}."
      )
    self._endpoint_body_ids = body_ids
    self._endpoint_site_ids = site_ids
    limits = self._asset.data.default_joint_pos_limits[0]
    self._pose_scale = (0.5 * (limits[:, 1] - limits[:, 0])).clamp_min(0.1)
    self._q0 = self._asset.data.default_joint_pos[0].clone()
    self._history = CircularBuffer(self._window_size, env.num_envs, env.device)
    self._fixed_noise = torch.randn(
      len(self._timesteps),
      env.num_envs,
      model_cfg.window_size,
      model_cfg.feature_dim,
      device=env.device,
    )
    self._previous_energy = torch.zeros(env.num_envs, device=env.device)
    self._previous_feature = torch.zeros(
      env.num_envs, model_cfg.feature_dim, device=env.device
    )
    self._previous_direction = torch.zeros_like(self._previous_feature)
    self._guidance_ready = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    self._awaiting_landing = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    self._full_history_required = torch.zeros_like(self._awaiting_landing)
    self._landing_contact_steps = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )
    self.raw_reward = torch.zeros(env.num_envs, device=env.device)
    self.capped_reward = torch.zeros_like(self.raw_reward)
    self.absolute_reward = torch.zeros_like(self.raw_reward)
    self.energy = torch.zeros_like(self.raw_reward)
    self.energy_descent = torch.zeros_like(self.raw_reward)
    self.direction_alignment = torch.zeros_like(self.raw_reward)
    self.ood_gate = torch.zeros_like(self.raw_reward)
    self.guidance_gate = torch.zeros_like(self.raw_reward)
    self.energy_descent_reward = torch.zeros_like(self.raw_reward)
    self.direction_reward = torch.zeros_like(self.raw_reward)
    self.weighted_reward = torch.zeros_like(self.raw_reward)
    self.task_cap = torch.zeros_like(self.raw_reward)
    self.prior_weight = torch.zeros_like(self.raw_reward)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    tensor_ids = (
      torch.arange(self._env.num_envs, device=self._env.device, dtype=torch.long)
      if env_ids is None
      else _slice_ids(self._env, env_ids)
      if isinstance(env_ids, slice)
      else env_ids
    )
    self._history.reset(tensor_ids)
    state = _get_recovery_state(self._env, self._event_name)
    self._awaiting_landing[tensor_ids] = state.reset_noisy[tensor_ids]
    self._full_history_required[tensor_ids] = state.reset_noisy[tensor_ids]
    self._landing_contact_steps[tensor_ids] = 0
    self._fixed_noise[:, tensor_ids] = torch.randn(
      len(self._timesteps),
      len(tensor_ids),
      self._window_size,
      G1_SMP_FEATURE_DIM,
      device=self._env.device,
    )
    self._guidance_ready[tensor_ids] = False
    self._previous_energy[tensor_ids] = 0.0
    self._previous_feature[tensor_ids] = 0.0
    self._previous_direction[tensor_ids] = 0.0
    self.raw_reward[tensor_ids] = 0.0
    self.capped_reward[tensor_ids] = 0.0
    self.absolute_reward[tensor_ids] = 0.0
    self.energy[tensor_ids] = 0.0
    self.energy_descent[tensor_ids] = 0.0
    self.direction_alignment[tensor_ids] = 0.0
    self.ood_gate[tensor_ids] = 0.0
    self.guidance_gate[tensor_ids] = 0.0
    self.energy_descent_reward[tensor_ids] = 0.0
    self.direction_reward[tensor_ids] = 0.0
    self.weighted_reward[tensor_ids] = 0.0
    self.task_cap[tensor_ids] = 0.0
    self.prior_weight[tensor_ids] = 0.0

  @torch.no_grad()
  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    checkpoint: str,
    checkpoint_schema_version: str,
    normalizer_file: str,
    event_name: str,
    landing_sensor_name: str,
    landing_contact_hold_steps: int,
    esm_timesteps: tuple[int, ...],
    esm_error_means: tuple[float, ...],
    smp_scale: float,
    reward_weight: float,
    terminal_reward_weight: float,
    handoff_progress: tuple[float, float],
    nominal_height_m: float,
    energy_descent_weight: float,
    energy_descent_max_rate: float,
    direction_weight: float,
    direction_minimum_motion_rms: float,
    ood_score_range: tuple[float, float],
  ) -> torch.Tensor:
    del (
      env,
      asset_cfg,
      checkpoint,
      checkpoint_schema_version,
      normalizer_file,
      event_name,
      landing_sensor_name,
      landing_contact_hold_steps,
      esm_timesteps,
      esm_error_means,
      smp_scale,
      reward_weight,
      terminal_reward_weight,
      handoff_progress,
      nominal_height_m,
      energy_descent_weight,
      energy_descent_max_rate,
      direction_weight,
      direction_minimum_motion_rms,
      ood_score_range,
    )
    feature = (self._encode_feature() - self._mean) / self._std
    state = _get_recovery_state(self._env, self._event_name)
    awaiting_before_update = self._awaiting_landing.clone()
    assert self._landing_sensor.data.found is not None
    ground_contact = torch.any(self._landing_sensor.data.found > 0, dim=-1)
    contact_steps, landed = update_landing_contact_hold(
      self._awaiting_landing,
      self._landing_contact_steps,
      ground_contact,
      self._landing_contact_hold_steps,
    )
    self._landing_contact_steps.copy_(contact_steps)
    self._awaiting_landing[landed] = False
    self._landing_contact_steps[landed] = 0
    self._history.append(feature)
    # The common circular buffer appends every environment at once. Clear rows
    # which were still landing at the start of this control step, including the
    # step that completed the contact debounce. Their first retained sample is
    # therefore the next, fully post-landing state.
    self._history.reset(awaiting_before_update)
    self.raw_reward.zero_()
    self.capped_reward.zero_()
    self.absolute_reward.zero_()
    self.energy.zero_()
    self.energy_descent.zero_()
    self.direction_alignment.zero_()
    self.ood_gate.zero_()
    self.guidance_gate.zero_()
    self.energy_descent_reward.zero_()
    self.direction_reward.zero_()
    self.weighted_reward.zero_()
    self.prior_weight.copy_(
      progress_handoff_weight(
        state.current_progress,
        self._reward_weight,
        self._terminal_reward_weight,
        self._handoff_low,
        self._handoff_high,
      )
    )
    self.task_cap.copy_(0.3 + 0.7 * state.current_progress.clamp(0.0, 1.0))
    mode_gate = (state.mode == REFERENCE_MODE) | (state.mode == FALLEN_MODE)
    # Clean states preserve the original immediate score. Lifted noisy states
    # must first land and then contribute ten genuine post-landing frames; the
    # CircularBuffer's first-frame backfill must not be scored for those rows.
    history_ready = (~self._full_history_required) | (
      self._history.current_length >= self._window_size
    )
    valid = history_ready & ~self._awaiting_landing & mode_gate
    if not bool(valid.any()):
      return self.weighted_reward

    windows = self._history.buffer[valid]
    batch_size = len(windows)
    ensemble_size = len(self._timesteps)
    expanded = windows.repeat(ensemble_size, 1, 1)
    timesteps = self._timesteps.repeat_interleave(batch_size)
    noise = self._fixed_noise[:, valid].reshape_as(expanded)
    alpha_bar = self._alpha_bar[timesteps, None, None]
    noised = alpha_bar.sqrt() * expanded + (1.0 - alpha_bar).sqrt() * noise
    predicted = self._model(noised, timesteps)
    errors = torch.mean(torch.square(predicted - noise), dim=(1, 2)).reshape(
      ensemble_size, batch_size
    )
    energy = normalized_esm_energy(errors, self._calibration_means, self._scale)
    score = torch.exp(-energy)
    denoised = (
      noised - (1.0 - alpha_bar).sqrt() * predicted
    ) / alpha_bar.sqrt().clamp_min(1e-6)
    direction = (
      denoised.reshape(
        ensemble_size, batch_size, self._window_size, G1_SMP_FEATURE_DIM
      ).mean(dim=0)[:, -1]
      - windows[:, -1]
    )
    ready = self._guidance_ready[valid]
    descent = torch.zeros_like(energy)
    alignment = torch.zeros_like(energy)
    if bool(ready.any()):
      descent[ready] = energy_descent_signal(
        self._previous_energy[valid][ready],
        energy[ready],
        self._env.step_dt,
        self._energy_descent_max_rate,
      )
      alignment[ready] = directional_progress_signal(
        feature[valid][ready] - self._previous_feature[valid][ready],
        self._previous_direction[valid][ready],
        self._direction_minimum_motion_rms,
      )
    distribution_gate = ood_score_gate(
      score, self._ood_score_full_below, self._ood_score_zero_above
    )
    motion_gate = recovery_guidance_gate(
      state.current_progress[valid], self._handoff_low, self._handoff_high
    )
    guidance_gate = distribution_gate * motion_gate
    capped = torch.minimum(score, self.task_cap[valid])
    absolute = self.prior_weight[valid] * capped
    descent_reward = self._energy_descent_weight * guidance_gate * descent
    direction_reward = self._direction_weight * guidance_gate * alignment
    self.raw_reward[valid] = score
    self.capped_reward[valid] = capped
    self.absolute_reward[valid] = absolute
    self.energy[valid] = energy
    self.energy_descent[valid] = descent
    self.direction_alignment[valid] = alignment
    self.ood_gate[valid] = distribution_gate
    self.guidance_gate[valid] = guidance_gate
    self.energy_descent_reward[valid] = descent_reward
    self.direction_reward[valid] = direction_reward
    self.weighted_reward[valid] = absolute + descent_reward + direction_reward
    self._previous_energy[valid] = energy
    self._previous_feature[valid] = feature[valid]
    self._previous_direction[valid] = direction
    self._guidance_ready[valid] = True
    return self.weighted_reward

  def _encode_feature(self) -> torch.Tensor:
    data = self._asset.data
    pelvis_position = data.root_link_pos_w
    pelvis_quaternion = data.root_link_quat_w
    body_endpoints = data.body_link_pos_w[:, self._endpoint_body_ids]
    site_endpoints = data.site_pos_w[:, self._endpoint_site_ids]
    endpoints = torch.cat((body_endpoints, site_endpoints), dim=1)
    endpoint_quaternion = pelvis_quaternion[:, None, :].expand(-1, 4, -1)
    endpoint_local = quat_apply_inverse(
      endpoint_quaternion, endpoints - pelvis_position[:, None, :]
    )
    height = (
      pelvis_position[:, 2] - self._env.scene.env_origins[:, 2]
    ) / self._nominal_height
    return torch.cat(
      (
        height[:, None],
        data.projected_gravity_b,
        data.root_link_lin_vel_b,
        data.root_link_ang_vel_b,
        (data.joint_pos - self._q0) / self._pose_scale,
        endpoint_local.flatten(1),
      ),
      dim=1,
    )


def _slice_ids(env: ManagerBasedRlEnv, env_ids: slice) -> torch.Tensor:
  return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]


def _get_recovery_state(env: ManagerBasedRlEnv, event_name: str) -> Any:
  state = env.event_manager.get_term_cfg(event_name).func
  required = ("mode", "current_progress", "reset_noisy")
  if any(not hasattr(state, name) for name in required):
    raise TypeError(f"Event '{event_name}' is not a G1 recovery state.")
  return cast(Any, state)


def get_g1_smp_reward(env: ManagerBasedRlEnv, reward_name: str) -> G1SmpReward:
  term = env.reward_manager.get_term_cfg(reward_name).func
  if not isinstance(term, G1SmpReward):
    raise TypeError(f"Reward '{reward_name}' is not a G1SmpReward.")
  return term


def g1_smp_metric(env: ManagerBasedRlEnv, reward_name: str, field: str) -> torch.Tensor:
  reward = get_g1_smp_reward(env, reward_name)
  value = getattr(reward, field)
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"SMP reward field {field!r} is not a tensor.")
  return value
