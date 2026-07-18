"""Tests for the recurrent RL_BOY velocity configuration."""

from mjlab.tasks.velocity.config.rlboy.env_cfgs import rlboy_flat_gru_env_cfg
from mjlab.tasks.velocity.config.rlboy.rl_cfg import rlboy_gru_ppo_runner_cfg


def test_gru_actor_uses_deployable_current_frame_observations() -> None:
  cfg = rlboy_flat_gru_env_cfg()
  actor_terms = cfg.observations["actor"].terms

  assert "base_lin_vel" not in actor_terms
  assert "base_ang_vel" in actor_terms
  assert "applied_torque_continuous_ratio" in actor_terms
  assert "applied_torque_peak_ratio" not in actor_terms
  assert "requested_torque_peak_ratio" in actor_terms
  assert all(term.history_length == 0 for term in actor_terms.values())


def test_gru_critic_keeps_complete_current_frame_observations() -> None:
  cfg = rlboy_flat_gru_env_cfg()
  critic_terms = cfg.observations["critic"].terms

  for term_name in (
    "base_lin_vel",
    "base_ang_vel",
    "applied_torque_continuous_ratio",
    "applied_torque_peak_ratio",
    "requested_torque_peak_ratio",
    "foot_height",
    "foot_air_time",
    "foot_contact",
    "foot_contact_forces",
  ):
    assert term_name in critic_terms
  assert all(term.history_length == 0 for term in critic_terms.values())


def test_rlboy_gru_runner_uses_recurrent_actor_only() -> None:
  cfg = rlboy_gru_ppo_runner_cfg()

  assert cfg.actor.class_name == "RNNModel"
  assert cfg.actor.rnn_type == "gru"
  assert cfg.actor.rnn_hidden_dim == 256
  assert cfg.actor.rnn_num_layers == 1
  assert cfg.critic.class_name == "MLPModel"
  assert cfg.critic.rnn_type is None
  assert cfg.experiment_name == "rlboy_velocity_gru"
