"""PPO-compatible Flow Policy Optimization for conditional flow actors."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from mjlab.rl.flow_policy import ConditionalFlowPolicy


def _aspo_objective(
  ratio: torch.Tensor, advantages: torch.Tensor, clip_param: float
) -> torch.Tensor:
  """Return FPO++'s PPO-positive/SPO-negative sample objective."""
  objective = advantages * ratio
  ppo_objective = torch.minimum(
    objective,
    advantages * ratio.clamp(1.0 - clip_param, 1.0 + clip_param),
  )
  spo_objective = objective - advantages.abs() * ratio.sub(1.0).square() / (
    2.0 * clip_param
  )
  return torch.where(advantages >= 0.0, ppo_objective, spo_objective)


class FlowPolicyOptimization(PPO):
  """FPO++ with per-MC-sample ratios and an asymmetric SPO trust region."""

  actor: ConditionalFlowPolicy

  def __init__(
    self,
    *args,
    num_mc_samples: int = 16,
    ratio_log_clip: float = 3.0,
    cfm_loss_clamp: float = 3.0,
    advantage_clamp: tuple[float, float] = (5.0, 5.0),
    weight_decay: float = 1.0e-4,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    diversity_loss_coef: float = 0.0,
    diversity_target_std: float = 0.12,
    diversity_batch_size: int = 64,
    **kwargs,
  ) -> None:
    if num_mc_samples < 1:
      raise ValueError("num_mc_samples must be positive.")
    if ratio_log_clip <= 0.0:
      raise ValueError("ratio_log_clip must be positive.")
    if cfm_loss_clamp <= 0.0:
      raise ValueError("cfm_loss_clamp must be positive.")
    if len(advantage_clamp) != 2 or min(advantage_clamp) <= 0.0:
      raise ValueError("advantage_clamp must contain positive bounds.")
    if weight_decay < 0.0:
      raise ValueError("weight_decay must be non-negative.")
    if diversity_loss_coef < 0.0:
      raise ValueError("diversity_loss_coef must be non-negative.")
    if diversity_target_std <= 0.0:
      raise ValueError("diversity_target_std must be positive.")
    if diversity_batch_size < 1:
      raise ValueError("diversity_batch_size must be positive.")
    if float(kwargs.get("entropy_coef", 0.0)) != 0.0:
      raise ValueError("FPO does not use the Gaussian entropy coefficient.")
    super().__init__(*args, **kwargs)
    if not isinstance(self._raw_actor, ConditionalFlowPolicy):
      raise TypeError("FlowPolicyOptimization requires ConditionalFlowPolicy.")
    if self.rnd or self.symmetry:
      raise ValueError(
        "The initial FPO implementation does not support RND or symmetry."
      )
    self.num_mc_samples = int(num_mc_samples)
    self.ratio_log_clip = float(ratio_log_clip)
    self.cfm_loss_clamp = float(cfm_loss_clamp)
    self.advantage_clamp = tuple(float(value) for value in advantage_clamp)
    self.diversity_loss_coef = float(diversity_loss_coef)
    self.diversity_target_std = float(diversity_target_std)
    self.diversity_batch_size = int(diversity_batch_size)
    for group in self.optimizer.param_groups:
      group["weight_decay"] = float(weight_decay)
      group["betas"] = tuple(float(value) for value in adam_betas)

  def act(self, obs: TensorDict) -> torch.Tensor:
    """Sample flow actions and store fixed MC pairs for the FPO ratio."""
    self.transition.hidden_states = (
      self.actor.get_hidden_state(),
      self.critic.get_hidden_state(),
    )
    latent_actions, actions = self.actor.sample_with_latent(obs)
    latent_actions = latent_actions.detach()
    actions = actions.detach()
    old_loss, tau, noise = self.actor.make_fpo_samples(
      obs, latent_actions, self.num_mc_samples
    )
    self.transition.actions = actions
    self.transition.values = self.critic(obs).detach()
    self.transition.actions_log_prob = torch.zeros(
      actions.shape[0], device=actions.device, dtype=actions.dtype
    )
    self.transition.distribution_params = (
      old_loss.detach(),
      tau.detach(),
      noise.detach(),
      latent_actions,
    )
    self.transition.observations = obs
    return actions

  def update(self) -> dict[str, float]:
    """Optimize the clipped FPO surrogate and the standard PPO value loss."""
    if self.actor.is_recurrent or self.critic.is_recurrent:
      raise ValueError(
        "The initial FPO implementation supports feed-forward models only."
      )
    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_flow_loss = 0.0
    mean_ratio = 0.0
    mean_log_ratio_abs = 0.0
    mean_ratio_clipped_fraction = 0.0
    mean_log_ratio_bounded_fraction = 0.0
    mean_negative_advantage_fraction = 0.0
    mean_actor_grad_norm = 0.0
    mean_diversity_loss = 0.0
    mean_pair_std = 0.0
    mean_action_saturation = 0.0

    for batch in generator:
      assert batch.observations is not None
      assert batch.actions is not None
      assert batch.values is not None
      assert batch.advantages is not None
      assert batch.returns is not None
      assert batch.old_distribution_params is not None
      batch_advantages = batch.advantages
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch_advantages = (batch_advantages - batch_advantages.mean()) / (
            batch_advantages.std() + 1e-8
          )
          batch.advantages = batch_advantages
      with torch.no_grad():
        positive_bound, negative_bound = self.advantage_clamp
        batch_advantages = batch_advantages.clamp(-negative_bound, positive_bound)

      old_flow_loss, tau, noise, latent_actions = batch.old_distribution_params
      new_flow_loss = self.actor.flow_matching_loss(
        batch.observations, latent_actions, tau, noise
      )
      # FPO++ keeps a separate ratio for every fixed (tau, noise) pair. Clipping
      # after averaging the MC losses gives a much coarser trust region.
      bounded_old_loss = old_flow_loss.clamp(max=self.cfm_loss_clamp)
      bounded_new_loss = new_flow_loss.clamp(max=self.cfm_loss_clamp)
      raw_log_ratio = bounded_old_loss - bounded_new_loss
      # Bound exp() in the forward pass while preserving the unclipped gradient.
      clamped_log_ratio = raw_log_ratio.clamp(-self.ratio_log_clip, self.ratio_log_clip)
      log_ratio = raw_log_ratio + (clamped_log_ratio - raw_log_ratio).detach()
      ratio = torch.exp(log_ratio)
      advantages = batch_advantages.squeeze(-1).unsqueeze(-1)

      # Positive advantages retain PPO clipping. For negative advantages ASPO
      # supplies a quadratic restoring gradient instead of allowing CFM losses
      # to drift upward without bound.
      aspo_objective = _aspo_objective(ratio, advantages, self.clip_param)
      surrogate_loss = -aspo_objective.mean()

      values = self.critic(batch.observations)
      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).square()
        clipped_losses = (value_clipped - batch.returns).square()
        value_loss = torch.maximum(value_losses, clipped_losses).mean()
      else:
        value_loss = (batch.returns - values).square().mean()

      if self.diversity_loss_coef > 0.0:
        diversity_loss, pair_std, action_saturation = self.actor.diversity_loss(
          batch.observations,
          self.diversity_target_std,
          self.diversity_batch_size,
        )
      else:
        # Keep the same-observation spread as a diagnostic after replacing its
        # optimization objective with endpoint action perturbation.
        with torch.no_grad():
          diversity_loss, pair_std, action_saturation = self.actor.diversity_loss(
            batch.observations,
            self.diversity_target_std,
            self.diversity_batch_size,
          )
      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        + self.diversity_loss_coef * diversity_loss
      )
      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      actor_grad_norm = nn.utils.clip_grad_norm_(
        self.actor.parameters(), self.max_grad_norm
      )
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_flow_loss += new_flow_loss.mean().item()
      mean_ratio += ratio.mean().item()
      mean_log_ratio_abs += raw_log_ratio.detach().abs().mean().item()
      mean_ratio_clipped_fraction += (
        (
          (ratio.detach() < 1.0 - self.clip_param)
          | (ratio.detach() > 1.0 + self.clip_param)
        )
        .float()
        .mean()
        .item()
      )
      mean_log_ratio_bounded_fraction += (
        (raw_log_ratio.detach().abs() > self.ratio_log_clip).float().mean().item()
      )
      mean_negative_advantage_fraction += (advantages < 0.0).float().mean().item()
      mean_actor_grad_norm += actor_grad_norm.item()
      mean_diversity_loss += diversity_loss.item()
      mean_pair_std += pair_std.item()
      mean_action_saturation += action_saturation.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    return {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "flow_matching": mean_flow_loss / num_updates,
      "fpo_ratio": mean_ratio / num_updates,
      "fpo_log_ratio_abs": mean_log_ratio_abs / num_updates,
      "fpo_ratio_clipped_fraction": mean_ratio_clipped_fraction / num_updates,
      "fpo_log_ratio_bounded_fraction": mean_log_ratio_bounded_fraction / num_updates,
      "aspo_negative_fraction": mean_negative_advantage_fraction / num_updates,
      "actor_grad_norm": mean_actor_grad_norm / num_updates,
      "flow_diversity": mean_diversity_loss / num_updates,
      "flow_pair_std": mean_pair_std / num_updates,
      "action_saturation_fraction": mean_action_saturation / num_updates,
    }
