"""Conditional flow policy used by the G1 recovery task."""

from __future__ import annotations

import copy
from typing import Literal, cast

import torch
import torch.nn as nn
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from tensordict import TensorDict


class ConditionalFlowPolicy(nn.Module):
  """A single conditional flow over continuous actions.

  The policy consumes a chronological proprioceptive history. Gaussian noise is
  transported to an action with a small, fixed-step ODE solve. There are no
  discrete modes or expert identifiers: multimodality is represented by the one
  shared conditional vector field.
  """

  is_recurrent = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    *,
    embedding_dim: int = 128,
    num_heads: int = 4,
    num_transformer_layers: int = 1,
    num_flow_steps: int = 4,
    time_embedding_dim: int = 16,
    action_clip: float = 1.0,
    base_noise_std: float = 0.25,
    max_flow_velocity: float = 4.0,
    cfm_loss_reduction: Literal["mean", "sqrt", "sum"] = "mean",
    action_perturb_std: float = 0.0,
  ) -> None:
    super().__init__()
    if num_flow_steps < 1:
      raise ValueError("num_flow_steps must be positive.")
    if time_embedding_dim < 2 or time_embedding_dim % 2:
      raise ValueError("time_embedding_dim must be a positive even number.")
    if action_clip <= 0.0:
      raise ValueError("action_clip must be positive.")
    if base_noise_std <= 0.0:
      raise ValueError("base_noise_std must be positive.")
    if max_flow_velocity <= 0.0:
      raise ValueError("max_flow_velocity must be positive.")
    if cfm_loss_reduction not in ("mean", "sqrt", "sum"):
      raise ValueError("cfm_loss_reduction must be 'mean', 'sqrt', or 'sum'.")
    if action_perturb_std < 0.0:
      raise ValueError("action_perturb_std must be non-negative.")

    self.obs_groups = obs_groups[obs_set]
    if not self.obs_groups:
      raise ValueError("The flow actor requires at least one observation group.")
    sample_history = self._concatenate_observations(obs)
    if sample_history.ndim != 3:
      raise ValueError(
        "The flow actor expects observations shaped [batch, history, features], "
        f"got {tuple(sample_history.shape)}."
      )
    self.history_length = int(sample_history.shape[-2])
    self.obs_dim = int(sample_history.shape[-1])
    self.output_dim = int(output_dim)
    self.num_flow_steps = int(num_flow_steps)
    self.action_clip = float(action_clip)
    self.base_noise_std = float(base_noise_std)
    self.max_flow_velocity = float(max_flow_velocity)
    self.cfm_loss_reduction = cfm_loss_reduction
    self.action_perturb_std = float(action_perturb_std)
    self.obs_normalization = bool(obs_normalization)

    if self.obs_normalization:
      self.obs_normalizer: nn.Module = EmpiricalNormalization(self.obs_dim)
    else:
      self.obs_normalizer = nn.Identity()

    # A fixed-shape LayerNorm keeps heterogeneous proprioceptive quantities
    # numerically bounded without changing statistics between rollout and update.
    # This preserves the exact old/new FPO ratio identity at the start of an update.
    self.input_norm = nn.LayerNorm(self.obs_dim)
    self.input_projection = nn.Linear(self.obs_dim, embedding_dim)
    self.position_embedding = nn.Parameter(
      torch.zeros(1, self.history_length, embedding_dim)
    )
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=embedding_dim,
      nhead=num_heads,
      dim_feedforward=4 * embedding_dim,
      dropout=0.0,
      activation="gelu",
      batch_first=True,
      norm_first=False,
    )
    self.history_encoder = nn.TransformerEncoder(
      encoder_layer, num_layers=num_transformer_layers
    )
    self.history_norm = nn.LayerNorm(embedding_dim)

    frequencies = torch.pi * 2.0 ** torch.arange(time_embedding_dim // 2)
    self.register_buffer("time_frequencies", frequencies, persistent=False)
    vector_field_input = embedding_dim + self.output_dim + time_embedding_dim
    self.vector_field = MLP(
      vector_field_input,
      self.output_dim,
      hidden_dims,
      activation,
    )
    self.register_buffer("_last_output_std", torch.ones(self.output_dim))

    nn.init.normal_(self.position_embedding, std=0.02)
    last_linear = next(
      module for module in reversed(self.vector_field) if isinstance(module, nn.Linear)
    )
    nn.init.zeros_(last_linear.weight)
    nn.init.zeros_(last_linear.bias)

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    """Generate a stochastic rollout action or a deterministic deployment action.

    Training explicitly requests stochastic output so the same conditional flow
    can explore multiple recovery modes. Evaluation follows the zero-sampling
    convention and integrates from the center of the base distribution.
    ``masks`` and ``hidden_state`` are accepted for RSL-RL compatibility.
    """
    del masks, hidden_state
    history = self._concatenate_observations(obs)
    if stochastic_output:
      return self.sample_history(history)
    noise = torch.zeros(
      *history.shape[:-2],
      self.output_dim,
      device=history.device,
      dtype=history.dtype,
    )
    return self.sample_history(history, noise)

  def sample(self, obs: TensorDict, noise: torch.Tensor | None = None) -> torch.Tensor:
    """Return the squashed environment action sampled by the conditional flow."""
    _, action = self.sample_with_latent(obs, noise)
    return action

  def sample_with_latent(
    self, obs: TensorDict, noise: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Flow endpoint and its bounded environment action."""
    history = self._concatenate_observations(obs)
    latent_action = self.sample_latent_history(history, noise)
    # A small endpoint perturbation is the FPO++ entropy regularizer. The
    # perturbed latent is also stored by FPO, so the policy learns the density of
    # the action that was actually executed instead of treating the perturbation
    # as unmodelled environment noise.
    if self.training and self.action_perturb_std > 0.0:
      latent_action = latent_action + self.action_perturb_std * torch.randn_like(
        latent_action
      )
    action = self.squash_action(latent_action)
    self._record_output_std(action)
    return latent_action, action

  def sample_history(
    self, history: torch.Tensor, noise: torch.Tensor | None = None
  ) -> torch.Tensor:
    """Sample directly from an observation-history tensor for deployment."""
    latent_action = self.sample_latent_history(history, noise)
    action = self.squash_action(latent_action)
    self._record_output_std(action)
    return action

  def sample_latent_history(
    self, history: torch.Tensor, noise: torch.Tensor | None = None
  ) -> torch.Tensor:
    """Integrate and return the unsquashed endpoint used by the CFM ratio."""
    context = self.encode_history_tensor(history)
    if noise is None:
      noise = self.base_noise_std * torch.randn(
        *context.shape[:-1],
        self.output_dim,
        device=context.device,
        dtype=context.dtype,
      )
    return self.sample_latent_from_context(context, noise)

  def sample_latent_from_context(
    self, context: torch.Tensor, noise: torch.Tensor
  ) -> torch.Tensor:
    """Integrate the Flow from noise using an already encoded context."""
    if noise.shape != (*context.shape[:-1], self.output_dim):
      raise ValueError(
        f"Expected flow noise shape {(*context.shape[:-1], self.output_dim)}, "
        f"got {tuple(noise.shape)}."
      )
    action = noise
    step_size = 1.0 / self.num_flow_steps
    for step in range(self.num_flow_steps):
      tau = torch.full(
        (*action.shape[:-1], 1),
        (step + 0.5) * step_size,
        device=action.device,
        dtype=action.dtype,
      )
      action = action + step_size * self.velocity_from_context(context, action, tau)
    return action

  def diversity_loss(
    self,
    obs: TensorDict,
    target_std: float,
    max_batch_size: int,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preserve endpoint sensitivity to base noise for the same observation.

    The diagnostic uses two independent Flow samples for a small observation
    subset. It does not add mode, curriculum, assistance, or reference inputs to
    the Actor. Limiting the subset keeps the additional ODE solves inexpensive.
    """
    if target_std <= 0.0:
      raise ValueError("target_std must be positive.")
    if max_batch_size < 1:
      raise ValueError("max_batch_size must be positive.")
    batch_size = int(obs.batch_size[0])
    if batch_size > max_batch_size:
      indices = torch.randperm(batch_size, device=obs.device)[:max_batch_size]
      obs = cast(TensorDict, obs[indices])
    context = self.encode_history(obs)
    noise_a = self.base_noise_std * torch.randn(
      *context.shape[:-1],
      self.output_dim,
      device=context.device,
      dtype=context.dtype,
    )
    noise_b = self.base_noise_std * torch.randn_like(noise_a)
    paired_context = torch.cat((context, context))
    paired_noise = torch.cat((noise_a, noise_b))
    paired_action = self.squash_action(
      self.sample_latent_from_context(paired_context, paired_noise)
    )
    action_a, action_b = paired_action.chunk(2)
    # For two independent samples, E[(a-b)^2] / 2 estimates variance.
    pair_std = torch.sqrt(0.5 * (action_a - action_b).square().mean(dim=-1) + 1e-8)
    loss = torch.relu(target_std - pair_std).square().mean()
    saturation = paired_action.abs().gt(0.99).float().mean()
    return loss, pair_std.mean(), saturation

  def squash_action(self, latent_action: torch.Tensor) -> torch.Tensor:
    """Map a Flow endpoint continuously into the environment action range."""
    return self.action_clip * torch.tanh(latent_action / self.action_clip)

  def _record_output_std(self, action: torch.Tensor) -> None:
    if action.numel() > 0:
      output_std = cast(torch.Tensor, self._last_output_std)
      output_std.copy_(
        action.detach().reshape(-1, self.output_dim).std(0, unbiased=False)
      )

  def flow_matching_loss(
    self,
    obs: TensorDict,
    actions: torch.Tensor,
    tau: torch.Tensor,
    noise: torch.Tensor,
  ) -> torch.Tensor:
    """Return CFM losses for unsquashed Flow endpoints and fixed MC pairs."""
    if tau.ndim != actions.ndim + 1 or noise.ndim != actions.ndim + 1:
      raise ValueError("FPO tau and noise tensors must include an MC dimension.")
    context = self.encode_history(obs)
    mc_samples = tau.shape[-2]
    context = context.unsqueeze(-2).expand(*context.shape[:-1], mc_samples, -1)
    expanded_actions = actions.unsqueeze(-2).expand_as(noise)
    noisy_actions = tau * expanded_actions + (1.0 - tau) * noise
    target_velocity = expanded_actions - noise
    predicted_velocity = self.velocity_from_context(context, noisy_actions, tau)
    squared_error = (predicted_velocity - target_velocity).square()
    if self.cfm_loss_reduction == "mean":
      return squared_error.mean(dim=-1)
    if self.cfm_loss_reduction == "sum":
      return squared_error.sum(dim=-1)
    # Variance-preserving scaling used by FPO++ avoids weakening the policy
    # ratio by a factor equal to the action dimension.
    return squared_error.sum(dim=-1) / self.output_dim**0.5

  def make_fpo_samples(
    self, obs: TensorDict, actions: torch.Tensor, num_mc_samples: int
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw and evaluate the fixed noise/timestep pairs stored by FPO."""
    if num_mc_samples < 1:
      raise ValueError("num_mc_samples must be positive.")
    tau = 0.005 + 0.99 * torch.rand(
      *actions.shape[:-1],
      num_mc_samples,
      1,
      device=actions.device,
      dtype=actions.dtype,
    )
    noise = self.base_noise_std * torch.randn(
      *actions.shape[:-1],
      num_mc_samples,
      self.output_dim,
      device=actions.device,
      dtype=actions.dtype,
    )
    loss = self.flow_matching_loss(obs, actions, tau, noise)
    return loss, tau, noise

  def encode_history(self, obs: TensorDict) -> torch.Tensor:
    """Encode real-robot proprioceptive history without curriculum context."""
    return self.encode_history_tensor(self._concatenate_observations(obs))

  def encode_history_tensor(self, history: torch.Tensor) -> torch.Tensor:
    """Encode a chronological history tensor."""
    if history.shape[-2:] != (self.history_length, self.obs_dim):
      raise ValueError(
        "Flow observation shape changed after construction: expected "
        f"[..., {self.history_length}, {self.obs_dim}], got {tuple(history.shape)}."
      )
    history = self.input_norm(self.obs_normalizer(history))
    tokens = self.input_projection(history) + self.position_embedding
    causal_mask = torch.triu(
      torch.ones(
        self.history_length,
        self.history_length,
        dtype=torch.bool,
        device=tokens.device,
      ),
      diagonal=1,
    )
    encoded = self.history_encoder(tokens, mask=causal_mask)
    return self.history_norm(encoded[..., -1, :])

  def velocity_from_context(
    self, context: torch.Tensor, noisy_action: torch.Tensor, tau: torch.Tensor
  ) -> torch.Tensor:
    """Evaluate the one shared conditional action vector field."""
    angles = tau * cast(torch.Tensor, self.time_frequencies)
    time_embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    raw_velocity = self.vector_field(
      torch.cat((context, noisy_action, time_embedding), dim=-1)
    )
    return self.max_flow_velocity * torch.tanh(raw_velocity / self.max_flow_velocity)

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      history = self._concatenate_observations(obs)
      self.obs_normalizer.update(history.reshape(-1, self.obs_dim))  # type: ignore

  def reset(
    self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None
  ) -> None:
    del dones, hidden_state

  def get_hidden_state(self) -> HiddenState:
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    del dones

  @property
  def output_std(self) -> torch.Tensor:
    """Observed rollout spread, used only by the generic RSL-RL logger."""
    return cast(torch.Tensor, self._last_output_std)

  def as_jit(self) -> nn.Module:
    return _ExportedFlowPolicy(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _OnnxFlowPolicy(self)

  def _concatenate_observations(self, obs: TensorDict) -> torch.Tensor:
    tensors = [cast(torch.Tensor, obs[group]) for group in self.obs_groups]
    return torch.cat(tensors, dim=-1) if len(tensors) > 1 else tensors[0]


class _ExportedFlowPolicy(nn.Module):
  """Deterministic zero-sampling wrapper for deployment."""

  def __init__(self, model: ConditionalFlowPolicy) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.input_norm = copy.deepcopy(model.input_norm)
    self.input_projection = copy.deepcopy(model.input_projection)
    self.register_buffer(
      "position_embedding", model.position_embedding.detach().clone()
    )
    self.history_encoder = copy.deepcopy(model.history_encoder)
    self.history_norm = copy.deepcopy(model.history_norm)
    time_frequencies = cast(torch.Tensor, model.time_frequencies)
    self.time_frequencies = nn.Parameter(
      time_frequencies.detach().clone(), requires_grad=False
    )
    self.vector_field = copy.deepcopy(model.vector_field)
    self.history_length = model.history_length
    self.obs_dim = model.obs_dim
    self.output_dim = model.output_dim
    self.num_flow_steps = model.num_flow_steps
    self.action_clip = model.action_clip
    self.max_flow_velocity = model.max_flow_velocity

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    history = self.input_norm(self.obs_normalizer(history))
    tokens = self.input_projection(history) + self.position_embedding
    causal_mask = torch.triu(
      torch.ones(
        self.history_length,
        self.history_length,
        dtype=torch.bool,
        device=history.device,
      ),
      diagonal=1,
    )
    encoded = self.history_encoder(tokens, mask=causal_mask)
    context = self.history_norm(encoded[:, -1, :])
    action = torch.zeros(
      (history.shape[0], self.output_dim),
      device=history.device,
      dtype=history.dtype,
    )
    step_size = 1.0 / self.num_flow_steps
    for step in range(self.num_flow_steps):
      tau = torch.full_like(action[:, :1], (step + 0.5) * step_size)
      angles = tau * self.time_frequencies
      time_embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
      raw_velocity = self.vector_field(
        torch.cat((context, action, time_embedding), dim=-1)
      )
      velocity = self.max_flow_velocity * torch.tanh(
        raw_velocity / self.max_flow_velocity
      )
      action = action + step_size * velocity
    return self.action_clip * torch.tanh(action / self.action_clip)

  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxFlowPolicy(_ExportedFlowPolicy):
  input_names = ["observation_history"]
  output_names = ["actions"]
  is_recurrent = False

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.history_length, self.obs_dim),)
