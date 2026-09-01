"""Tests for the multimodal conditional flow policy and FPO primitives."""

import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.rl.flow_policy import ConditionalFlowPolicy
from mjlab.rl.fpo import FlowPolicyOptimization, _aspo_objective


def _flow_policy(batch_size: int = 8) -> tuple[ConditionalFlowPolicy, TensorDict]:
  observations = TensorDict(
    {"actor": torch.randn(batch_size, 10, 93)}, batch_size=[batch_size]
  )
  policy = ConditionalFlowPolicy(
    observations,
    {"actor": ["actor"]},
    "actor",
    29,
    hidden_dims=(64, 32),
    embedding_dim=32,
    num_heads=4,
    num_flow_steps=2,
  )
  return policy, observations


def test_flow_policy_is_batched_bounded_and_noise_conditioned():
  policy, observations = _flow_policy()
  zero_noise = torch.zeros(8, 29)
  one_noise = torch.ones(8, 29)

  zero_actions = policy.sample(observations, zero_noise)
  one_actions = policy.sample(observations, one_noise)

  assert zero_actions.shape == (8, 29)
  assert torch.all(zero_actions.abs() < 1.0)
  assert torch.all(one_actions.abs() < 1.0)
  assert not torch.allclose(zero_actions, one_actions)


def test_flow_policy_uses_random_training_and_zero_noise_evaluation():
  policy, observations = _flow_policy()

  deterministic_a = policy(observations)
  deterministic_b = policy(observations)
  stochastic_a = policy(observations, stochastic_output=True)
  stochastic_b = policy(observations, stochastic_output=True)

  torch.testing.assert_close(deterministic_a, deterministic_b)
  assert not torch.allclose(stochastic_a, stochastic_b)


def test_flow_policy_action_perturbation_only_applies_during_training():
  policy, observations = _flow_policy()
  policy.action_perturb_std = 0.02
  noise = torch.zeros(8, 29)

  policy.train()
  train_latent_a, _ = policy.sample_with_latent(observations, noise)
  train_latent_b, _ = policy.sample_with_latent(observations, noise)
  policy.eval()
  eval_latent_a, _ = policy.sample_with_latent(observations, noise)
  eval_latent_b, _ = policy.sample_with_latent(observations, noise)

  assert not torch.allclose(train_latent_a, train_latent_b)
  torch.testing.assert_close(eval_latent_a, eval_latent_b)


def test_flow_policy_starts_with_low_amplitude_actions_and_bounded_velocity():
  policy, observations = _flow_policy(batch_size=512)
  actions = policy.sample(observations)
  context = policy.encode_history(observations)
  velocity = policy.velocity_from_context(
    context, torch.randn_like(actions), torch.full((512, 1), 0.5)
  )

  assert actions.std() < 0.35
  assert torch.all(velocity.abs() <= policy.max_flow_velocity)


def test_fpo_fixed_mc_pairs_give_identity_ratio_before_update():
  policy, observations = _flow_policy()
  latent_actions, actions = policy.sample_with_latent(observations)
  old_loss, tau, noise = policy.make_fpo_samples(observations, latent_actions, 2)
  new_loss = policy.flow_matching_loss(observations, latent_actions, tau, noise)
  ratio = torch.exp(old_loss - new_loss)

  torch.testing.assert_close(actions, policy.squash_action(latent_actions))
  torch.testing.assert_close(ratio, torch.ones_like(ratio))
  assert ratio.shape == (8, 2)
  assert old_loss.shape == (8, 2)
  assert tau.shape == (8, 2, 1)
  assert noise.shape == (8, 2, 29)


def test_sqrt_cfm_reduction_preserves_action_dimension_scale():
  policy, observations = _flow_policy()
  latent_actions, _ = policy.sample_with_latent(observations)
  _, tau, noise = policy.make_fpo_samples(observations, latent_actions, 2)

  policy.cfm_loss_reduction = "mean"
  mean_loss = policy.flow_matching_loss(observations, latent_actions, tau, noise)
  policy.cfm_loss_reduction = "sqrt"
  sqrt_loss = policy.flow_matching_loss(observations, latent_actions, tau, noise)

  torch.testing.assert_close(sqrt_loss, mean_loss * 29**0.5)


def test_flow_diversity_loss_uses_same_observation_without_new_inputs():
  policy, observations = _flow_policy()

  loss, pair_std, saturation = policy.diversity_loss(
    observations, target_std=0.12, max_batch_size=4
  )

  assert loss.ndim == 0
  assert pair_std.ndim == 0
  assert saturation.ndim == 0
  assert torch.isfinite(loss)
  assert 0.0 <= pair_std
  assert 0.0 <= saturation <= 1.0


def test_flow_policy_torchscript_export_keeps_history_interface():
  policy, _ = _flow_policy()
  exported = torch.jit.script(policy.as_jit())
  history = torch.zeros(3, 10, 93)
  actions = exported(history)
  repeated_actions = exported(history)

  assert actions.shape == (3, 29)
  assert torch.all(actions.abs() < 1.0)
  torch.testing.assert_close(actions, repeated_actions)


def test_aspo_uses_ppo_for_positive_and_spo_for_negative_advantages():
  ratio = torch.tensor([[1.2], [1.2]])
  advantages = torch.tensor([[1.0], [-1.0]])

  objective = _aspo_objective(ratio, advantages, clip_param=0.05)

  torch.testing.assert_close(objective[0], torch.tensor([1.05]))
  torch.testing.assert_close(objective[1], torch.tensor([-1.6]))


def test_fpo_runs_a_complete_parallel_rollout_update():
  batch_size = 4
  observations = TensorDict(
    {
      "actor": torch.randn(batch_size, 10, 93),
      "critic": torch.randn(batch_size, 7),
    },
    batch_size=[batch_size],
  )
  groups = {"actor": ["actor"], "critic": ["critic"]}
  actor = ConditionalFlowPolicy(
    observations,
    groups,
    "actor",
    29,
    hidden_dims=(32, 32),
    embedding_dim=32,
    num_heads=4,
    num_flow_steps=2,
  )
  critic = MLPModel(
    observations,
    groups,
    "critic",
    1,
    hidden_dims=(32, 32),
  )
  storage = RolloutStorage("rl", batch_size, 2, observations, [29], "cpu")
  algorithm = FlowPolicyOptimization(
    actor,
    critic,
    storage,
    num_learning_epochs=1,
    num_mini_batches=1,
    entropy_coef=0.0,
    schedule="fixed",
    num_mc_samples=2,
    diversity_loss_coef=0.0,
    diversity_batch_size=2,
  )

  for _ in range(2):
    algorithm.act(observations)
    algorithm.process_env_step(
      observations,
      torch.randn(batch_size),
      torch.zeros(batch_size, dtype=torch.long),
      {},
    )
  algorithm.compute_returns(observations)
  losses = algorithm.update()

  assert storage.step == 0
  assert set(losses) == {
    "value",
    "surrogate",
    "flow_matching",
    "fpo_ratio",
    "fpo_log_ratio_abs",
    "fpo_ratio_clipped_fraction",
    "fpo_log_ratio_bounded_fraction",
    "aspo_negative_fraction",
    "actor_grad_norm",
    "flow_diversity",
    "flow_pair_std",
    "action_saturation_fraction",
  }
  assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
  assert losses["fpo_ratio"] <= torch.exp(torch.tensor(3.0)).item()
  assert losses["flow_pair_std"] > 0.0
  assert algorithm.optimizer.param_groups[0]["weight_decay"] == 1.0e-4
