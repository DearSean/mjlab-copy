"""RL configuration for Unitree G1 velocity task."""

from mjlab.rl import (
  RslRlFpoAlgorithmCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_recovery_fpo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create conditional-flow FPO for autonomous recovery with frozen SMP."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_recovery_s2"
  cfg.run_name = "flow_fpo_sqrt_entropy_smp"
  cfg.clip_actions = 1.0
  cfg.actor.class_name = "mjlab.rl.flow_policy:ConditionalFlowPolicy"
  cfg.actor.distribution_cfg = None
  cfg.actor.hidden_dims = (512, 256, 128)
  cfg.actor.activation = "elu"
  cfg.actor.model_kwargs = {
    "num_flow_steps": 64,
    "base_noise_std": 0.5,
    "max_flow_velocity": 1.5,
    "cfm_loss_reduction": "sqrt",
    "action_perturb_std": 0.02,
  }
  # FPO stores old per-action CFM losses. Keeping the actor normalizer fixed makes
  # the old/new ratio exactly one before the first optimization step.
  cfg.actor.obs_normalization = False
  cfg.critic.hidden_dims = (1024, 512, 256)
  cfg.algorithm = RslRlFpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=False,
    clip_param=0.05,
    entropy_coef=0.0,
    num_learning_epochs=8,
    num_mini_batches=8,
    learning_rate=1.0e-4,
    schedule="fixed",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
    optimizer="adamw",
    num_mc_samples=32,
    ratio_log_clip=3.0,
    cfm_loss_clamp=3.0,
    advantage_clamp=(5.0, 5.0),
    weight_decay=1.0e-4,
    diversity_loss_coef=0.0,
    diversity_target_std=0.12,
    diversity_batch_size=64,
  )
  cfg.max_iterations = 30_000
  return cfg


def unitree_g1_recovery_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Compatibility alias for the recovery runner now implemented with FPO."""
  return unitree_g1_recovery_fpo_runner_cfg()
