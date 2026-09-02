"""Dedicated flat G1 recovery training configuration in the velocity task family."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.recovery_prior.smp_model import FrozenG1SmpPriorCfg
from mjlab.tasks.velocity.recovery_prior.smp_reward import G1SmpReward, g1_smp_metric

from .env_cfgs import unitree_g1_flat_env_cfg
from .recovery_events import (
  G1RecoveryReset,
  g1_recovery_assist_curriculum,
  step_g1_recovery_state,
)
from .recovery_rewards import (
  BroadRecoveryPosture,
  FallenDurationPenalty,
  GatedUpright,
  RecoveryCompositeProgress,
  RecoveryMilestoneLift,
  gated_action_rate_l2,
  gated_angular_momentum_penalty,
  gated_body_angular_velocity_penalty,
  gated_self_collision_cost,
  recovery_assistance_metric,
  recovery_failure_penalty,
  recovery_hold_reward,
  recovery_mode_metric,
  recovery_mode_success_metric,
  recovery_progress_bin_metric,
  recovery_progress_bin_success_metric,
  recovery_progress_metric,
  recovery_standing_potential,
  recovery_succeeded,
  recovery_success_bonus,
)

_RECOVERY_EVENT_NAME = "g1_recovery_reset"
_ASSIST_FORCE_RANGES = (
  (160.0, 200.0),
  (120.0, 160.0),
  (90.0, 120.0),
  (65.0, 90.0),
  (45.0, 65.0),
  (25.0, 45.0),
  (10.0, 25.0),
  (0.0, 10.0),
  (0.0, 0.0),
)
_POSTURE_MODE_PROBABILITIES = (
  (1.00, 0.00, 0.00),
  (1.00, 0.00, 0.00),
  (1.00, 0.00, 0.00),
  (1.00, 0.00, 0.00),
  (1.00, 0.00, 0.00),
  (1.00, 0.00, 0.00),
  (1.00, 0.00, 0.00),
  (1.00, 0.00, 0.00),
  (1.00, 0.00, 0.00),
  (0.85, 0.15, 0.00),
  (0.70, 0.30, 0.00),
  (0.60, 0.35, 0.05),
  (0.45, 0.45, 0.10),
)
_POSTURE_REFERENCE_MIN_PROGRESS = (
  0.70,
  0.55,
  0.40,
  0.35,
  0.30,
  0.25,
  0.20,
  0.15,
  0.10,
  0.10,
  0.10,
  0.10,
  0.10,
)
_POSTURE_FALLEN_MIN_PROGRESS = (
  0.15,
  0.15,
  0.15,
  0.15,
  0.15,
  0.15,
  0.15,
  0.15,
  0.15,
  0.15,
  0.10,
  0.05,
  0.00,
)
_POSTURE_SUCCESS_WINDOWS = (
  500,
  750,
  1000,
  1250,
  1500,
  1750,
  2000,
  2500,
  3000,
  3500,
  4000,
  5000,
)
_ASSIST_SUCCESS_WINDOWS = (6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000)
_RECOVERY_PROGRESS_BINS = (
  ("070_085", 0.70, 0.85),
  ("055_070", 0.55, 0.70),
  ("040_055", 0.40, 0.55),
  ("035_040", 0.35, 0.40),
  ("030_035", 0.30, 0.35),
  ("025_030", 0.25, 0.30),
  ("020_025", 0.20, 0.25),
  ("015_020", 0.15, 0.20),
  ("010_015", 0.10, 0.15),
  ("000_010", 0.00, 0.10),
)
_RECOVERY_GATE_PARAMS = {
  "event_name": _RECOVERY_EVENT_NAME,
  "gate_low": 0.55,
  "gate_high": 0.85,
}


def unitree_g1_flat_recovery_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create autonomous G1 recovery PPO with a frozen motion prior."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  smp = FrozenG1SmpPriorCfg()
  # The 16-sample CFM update is substantially heavier than Gaussian PPO. Keep
  # the default inside an 8 GiB GPU budget; larger cards can override this.
  cfg.scene.num_envs = 16 if play else 1024
  cfg.episode_length_s = 6.0
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (0.0, 0.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.rel_standing_envs = 1.0
  cfg.terminations.pop("fell_over", None)
  cfg.events.pop("push_robot", None)
  cfg.curriculum.pop("command_vel", None)

  cfg.events[_RECOVERY_EVENT_NAME] = EventTermCfg(
    func=G1RecoveryReset,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "dataset_dir": str(Path("artifacts/g1_recovery")),
      "physical_init_file": str(Path("artifacts/g1_recovery/physical_init.npz")),
      "require_physical_init": not play,
      "force_ranges": _ASSIST_FORCE_RANGES,
      "assist_success_windows": _ASSIST_SUCCESS_WINDOWS,
      "posture_mode_probabilities": _POSTURE_MODE_PROBABILITIES,
      "posture_reference_min_progress": _POSTURE_REFERENCE_MIN_PROGRESS,
      "posture_fallen_min_progress": _POSTURE_FALLEN_MIN_PROGRESS,
      "posture_success_windows": _POSTURE_SUCCESS_WINDOWS,
      # Keep each success window comparable when a larger GPU raises num_envs.
      # The cooldown prevents one frozen policy from crossing several levels.
      "curriculum_reference_num_envs": 1024,
      "curriculum_minimum_level_steps": 24,
      "reference_frontier_probability": 0.5,
      "adaptive_bin_duration_s": 0.2,
      "adaptive_ema_rate": 0.01,
      # Restore the pre-adaptive frontier-balanced experiment: every recovery
      # reset uses the fixed 50% frontier / 50% mastered row sampler and every
      # completed recovery contributes to the curriculum window.  Keep the
      # temporal-bin statistics only as diagnostics for the long run.
      "adaptive_uniform_probability": 1.0,
      "adaptive_max_difficulty": 1.0,
      "adaptive_max_probability_ratio": 1.0,
      "curriculum_probe_probability": 1.0,
      "hard_bin_report_count": 5,
      "hard_bin_min_attempts": 20,
      "hard_bin_report_interval": 10000,
      "reference_max_progress": 0.85,
      "fallen_max_progress": 0.25,
      "initial_posture_level": len(_POSTURE_MODE_PROBABILITIES) - 1 if play else 0,
      "initial_assist_level": len(_ASSIST_FORCE_RANGES) - 1 if play else 0,
    },
  )
  cfg.events["g1_recovery_step"] = EventTermCfg(
    func=step_g1_recovery_state,
    mode="step",
    params={"event_name": _RECOVERY_EVENT_NAME},
  )
  if not play:
    cfg.curriculum["g1_recovery_assist"] = CurriculumTermCfg(
      func=g1_recovery_assist_curriculum,
      params={
        "event_name": _RECOVERY_EVENT_NAME,
        "success_threshold": 0.9,
      },
    )

  # The policy must infer the recovery behavior from deployable proprioception.
  # Curriculum mode, assistance, and the constant zero-velocity command are not
  # observable on the real robot and must not leak into either network.
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms.pop("command", None)
    cfg.observations[group_name].terms.pop("recovery_context", None)

  # Preserve short-term dynamics without adding a reference command. The actor
  # keeps the explicit [environment, time, feature] axes for its causal encoder.
  cfg.observations["actor"].history_length = 10
  cfg.observations["actor"].flatten_history_dim = False

  # Zero-velocity rewards oppose the transient motion needed to rise. The pose and
  # upright terms remain useful as the global standing attractor.
  cfg.rewards["track_linear_velocity"].weight = 0.0
  cfg.rewards["track_angular_velocity"].weight = 0.0
  cfg.rewards["upright"].weight = 2.0
  cfg.rewards["upright"].func = GatedUpright
  cfg.rewards["upright"].params.update({**_RECOVERY_GATE_PARAMS, "gate_minimum": 0.5})
  cfg.rewards["pose"] = RewardTermCfg(
    func=BroadRecoveryPosture,
    weight=2.0,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "event_name": _RECOVERY_EVENT_NAME,
      "gate_low": 0.65,
      "gate_high": 0.85,
      "gain": 1.0,
      "gate_minimum": 0.0,
    },
  )
  cfg.rewards["action_rate_l2"].func = gated_action_rate_l2
  cfg.rewards["action_rate_l2"].params.update(
    {**_RECOVERY_GATE_PARAMS, "gate_minimum": 0.3}
  )
  cfg.rewards["body_ang_vel"].func = gated_body_angular_velocity_penalty
  cfg.rewards["body_ang_vel"].params.update(
    {**_RECOVERY_GATE_PARAMS, "gate_minimum": 0.2}
  )
  cfg.rewards["angular_momentum"].func = gated_angular_momentum_penalty
  cfg.rewards["angular_momentum"].params.update(
    {**_RECOVERY_GATE_PARAMS, "gate_minimum": 0.2}
  )
  cfg.rewards["dof_pos_limits"].weight = -0.5
  cfg.rewards["self_collisions"].func = gated_self_collision_cost
  cfg.rewards["self_collisions"].weight = -0.2
  cfg.rewards["self_collisions"].params.update(
    {**_RECOVERY_GATE_PARAMS, "gate_minimum": 0.1}
  )
  cfg.rewards["recovery_potential"] = RewardTermCfg(
    func=recovery_standing_potential,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      **_RECOVERY_GATE_PARAMS,
      "upright_floor": 0.2,
    },
  )
  cfg.rewards["recovery_progress"] = RewardTermCfg(
    func=RecoveryCompositeProgress,
    weight=2.0,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "height_weight": 0.6,
      "upright_weight": 0.4,
      "max_progress_rate": 5.0,
      "max_regression_rate": 5.0,
    },
  )
  cfg.rewards["recovery_lift"] = RewardTermCfg(
    func=RecoveryMilestoneLift,
    weight=1.0,
    params={
      "event_name": _RECOVERY_EVENT_NAME,
      "max_progress_rate": 5.0,
    },
  )
  cfg.rewards["recovery_hold"] = RewardTermCfg(
    func=recovery_hold_reward,
    weight=2.0,
    params={
      "event_name": _RECOVERY_EVENT_NAME,
      "gate_low": 0.65,
      "gate_high": 0.85,
    },
  )
  cfg.rewards["fallen_duration"] = RewardTermCfg(
    func=FallenDurationPenalty,
    weight=-0.02,
    params={
      "event_name": _RECOVERY_EVENT_NAME,
      "threshold": 0.85,
      "tau_s": 1.0,
      "max_penalty": 1.0,
    },
  )
  cfg.rewards["recovery_success"] = RewardTermCfg(
    func=recovery_success_bonus,
    weight=10.0,
    params={"event_name": _RECOVERY_EVENT_NAME},
  )
  cfg.rewards["recovery_failure"] = RewardTermCfg(
    func=recovery_failure_penalty,
    weight=-1.0,
    params={"event_name": _RECOVERY_EVENT_NAME, "timeout_name": "time_out"},
  )
  if not play:
    cfg.rewards["smp"] = RewardTermCfg(
      func=G1SmpReward,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "checkpoint": str(smp.checkpoint),
        "checkpoint_schema_version": smp.checkpoint_schema_version,
        "normalizer_file": str(smp.normalizer_file),
        "event_name": _RECOVERY_EVENT_NAME,
        "esm_timesteps": smp.esm_timesteps,
        "esm_error_means": smp.esm_error_means,
        "smp_scale": smp.smp_scale,
        "reward_weight": smp.reward_weight,
        "terminal_reward_weight": smp.terminal_reward_weight,
        "handoff_progress": smp.handoff_progress,
        "nominal_height_m": 0.76,
        "energy_descent_weight": smp.energy_descent_weight,
        "energy_descent_max_rate": smp.energy_descent_max_rate,
        "direction_weight": smp.direction_weight,
        "direction_minimum_motion_rms": smp.direction_minimum_motion_rms,
        "ood_score_range": smp.ood_score_range,
      },
    )
  cfg.terminations["recovery_success"] = TerminationTermCfg(
    func=recovery_succeeded,
    time_out=False,
    params={
      "event_name": _RECOVERY_EVENT_NAME,
      "threshold": 0.85,
      "hold_steps": 25,
    },
  )

  cfg.metrics["recovery_progress"] = MetricsTermCfg(
    func=recovery_progress_metric,
    params={"event_name": _RECOVERY_EVENT_NAME},
  )
  cfg.metrics["recovery_assistance_n"] = MetricsTermCfg(
    func=recovery_assistance_metric,
    params={"event_name": _RECOVERY_EVENT_NAME},
  )
  for mode in ("reference", "fallen", "stand"):
    cfg.metrics[f"recovery_mode_{mode}"] = MetricsTermCfg(
      func=recovery_mode_metric,
      params={"event_name": _RECOVERY_EVENT_NAME, "mode": mode},
      reduce="last",
    )
    cfg.metrics[f"recovery_success_{mode}"] = MetricsTermCfg(
      func=recovery_mode_success_metric,
      params={"event_name": _RECOVERY_EVENT_NAME, "mode": mode},
      reduce="last",
    )
  for label, lower, upper in _RECOVERY_PROGRESS_BINS:
    cfg.metrics[f"recovery_bin_{label}"] = MetricsTermCfg(
      func=recovery_progress_bin_metric,
      params={
        "event_name": _RECOVERY_EVENT_NAME,
        "lower": lower,
        "upper": upper,
      },
      reduce="last",
    )
    cfg.metrics[f"recovery_success_bin_{label}"] = MetricsTermCfg(
      func=recovery_progress_bin_success_metric,
      params={
        "event_name": _RECOVERY_EVENT_NAME,
        "lower": lower,
        "upper": upper,
      },
      reduce="last",
    )
  if not play:
    for field in (
      "raw_reward",
      "capped_reward",
      "absolute_reward",
      "energy",
      "energy_descent",
      "direction_alignment",
      "ood_gate",
      "guidance_gate",
      "energy_descent_reward",
      "direction_reward",
      "weighted_reward",
      "task_cap",
      "prior_weight",
    ):
      cfg.metrics[f"smp_{field}"] = MetricsTermCfg(
        func=g1_smp_metric,
        params={"reward_name": "smp", "field": field},
      )
  return cfg
