"""Frozen online SMP reward for G1 recovery PPO."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
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
  normalized = errors / calibration_means[:, None].clamp_min(1e-6)
  return torch.exp(-scale * normalized.mean(dim=0))


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
    self._history = CircularBuffer(10, env.num_envs, env.device)
    self.raw_reward = torch.zeros(env.num_envs, device=env.device)
    self.capped_reward = torch.zeros_like(self.raw_reward)
    self.weighted_reward = torch.zeros_like(self.raw_reward)
    self.task_cap = torch.zeros_like(self.raw_reward)
    self.prior_weight = torch.zeros_like(self.raw_reward)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    tensor_ids: torch.Tensor | slice = slice(None) if env_ids is None else env_ids
    if env_ids is None or isinstance(env_ids, slice):
      self._history.reset(None if env_ids is None else _slice_ids(self._env, env_ids))
    else:
      self._history.reset(env_ids)
    self.raw_reward[tensor_ids] = 0.0
    self.capped_reward[tensor_ids] = 0.0
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
    esm_timesteps: tuple[int, ...],
    esm_error_means: tuple[float, ...],
    smp_scale: float,
    reward_weight: float,
    terminal_reward_weight: float,
    handoff_progress: tuple[float, float],
    nominal_height_m: float,
  ) -> torch.Tensor:
    del (
      env,
      asset_cfg,
      checkpoint,
      checkpoint_schema_version,
      normalizer_file,
      event_name,
      esm_timesteps,
      esm_error_means,
      smp_scale,
      reward_weight,
      terminal_reward_weight,
      handoff_progress,
      nominal_height_m,
    )
    feature = (self._encode_feature() - self._mean) / self._std
    self._history.append(feature)
    self.raw_reward.zero_()
    self.capped_reward.zero_()
    self.weighted_reward.zero_()
    state = _get_recovery_state(self._env, self._event_name)
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
    valid = (self._history.current_length >= 10) & mode_gate
    if not bool(valid.any()):
      return self.weighted_reward

    windows = self._history.buffer[valid]
    batch_size = len(windows)
    ensemble_size = len(self._timesteps)
    expanded = windows.repeat(ensemble_size, 1, 1)
    timesteps = self._timesteps.repeat_interleave(batch_size)
    noise = torch.randn_like(expanded)
    alpha_bar = self._alpha_bar[timesteps, None, None]
    noised = alpha_bar.sqrt() * expanded + (1.0 - alpha_bar).sqrt() * noise
    predicted = self._model(noised, timesteps)
    errors = torch.mean(torch.square(predicted - noise), dim=(1, 2)).reshape(
      ensemble_size, batch_size
    )
    score = normalized_esm_reward(errors, self._calibration_means, self._scale)
    capped = torch.minimum(score, self.task_cap[valid])
    self.raw_reward[valid] = score
    self.capped_reward[valid] = capped
    self.weighted_reward[valid] = self.prior_weight[valid] * capped
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
  required = ("mode", "current_progress")
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
