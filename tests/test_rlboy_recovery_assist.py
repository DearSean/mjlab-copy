"""Tests for the RL_BOY fallen-recovery assistance curriculum."""

import torch

from mjlab.tasks.velocity.config.rlboy.env_cfgs import rlboy_flat_env_cfg
from mjlab.tasks.velocity.config.rlboy.recovery_assist import (
  RECOVERY_ASSIST_EVENT_NAME,
  RlBoyRecoveryAssist,
  recovery_assist_curriculum,
  recovery_bad_orientation,
)
from mjlab.tasks.velocity.config.rlboy.rl_cfg import rlboy_ppo_runner_cfg


def test_flat_rlboy_enables_recovery_assist_only_during_training() -> None:
  train_cfg = rlboy_flat_env_cfg()
  play_cfg = rlboy_flat_env_cfg(play=True)

  assist_cfg = train_cfg.events[RECOVERY_ASSIST_EVENT_NAME]
  assert assist_cfg.func is RlBoyRecoveryAssist
  assert assist_cfg.mode == "step"
  assert assist_cfg.params["asset_cfg"].body_names == ("head_yaw_link",)
  assert assist_cfg.params["force_levels"][-1] == 0.0
  assert train_cfg.curriculum["recovery_assist"].func is recovery_assist_curriculum
  assert train_cfg.terminations["fell_over"].func is recovery_bad_orientation

  assert "randomize_fallen_pose" not in train_cfg.events
  assert next(iter(train_cfg.events)) == "prepare_recovery_group"
  assert "push_robot" in train_cfg.events
  assert "base_payload" in train_cfg.events
  assert RECOVERY_ASSIST_EVENT_NAME not in play_cfg.events
  assert "recovery_assist" not in play_cfg.curriculum


def test_recovery_level_requires_each_fallen_direction() -> None:
  assist = RlBoyRecoveryAssist.__new__(RlBoyRecoveryAssist)
  assist.level = 0
  assist._force_levels = torch.tensor((50.0, 40.0, 30.0))
  assist.attempts = torch.tensor((50, 50, 50, 50))
  assist.successes = torch.tensor((45, 45, 45, 45))

  assist.update_level(
    window_size=200,
    success_threshold=0.8,
    direction_threshold=0.7,
    min_direction_attempts=30,
    failure_threshold=0.4,
  )
  assert assist.level == 1
  assert assist.attempts.sum() == 0

  assist.attempts = torch.tensor((80, 80, 35, 5))
  assist.successes = torch.tensor((75, 75, 34, 5))
  assist.update_level(
    window_size=200,
    success_threshold=0.8,
    direction_threshold=0.7,
    min_direction_attempts=30,
    failure_threshold=0.4,
  )
  assert assist.level == 1


def test_rlboy_curricula_fit_four_thousand_iterations() -> None:
  env_cfg = rlboy_flat_env_cfg()
  runner_cfg = rlboy_ppo_runner_cfg()

  assert runner_cfg.max_iterations == 4_000
  stages = env_cfg.curriculum["command_vel"].params["velocity_stages"]
  assert [stage["step"] for stage in stages] == [
    0,
    800 * runner_cfg.num_steps_per_env,
    1600 * runner_cfg.num_steps_per_env,
    3200 * runner_cfg.num_steps_per_env,
  ]

  randomization_stages = env_cfg.curriculum["normal_randomization"].params["stages"]
  assert [stage["step"] for stage in randomization_stages] == [
    0,
    1200 * runner_cfg.num_steps_per_env,
    2000 * runner_cfg.num_steps_per_env,
    2800 * runner_cfg.num_steps_per_env,
    3400 * runner_cfg.num_steps_per_env,
  ]
  assert [stage["payload_range"][1] for stage in randomization_stages] == [
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
  ]
