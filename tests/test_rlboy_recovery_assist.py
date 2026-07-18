"""Tests for the RL_BOY fallen-recovery assistance curriculum."""

import inspect
from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.rlboy.env_cfgs import (
  gated_track_angular_velocity,
  gated_track_linear_velocity,
  gated_upright,
  gated_variable_posture,
  recovery_potential_progress,
  recovery_state_potential,
  recovery_time_penalty,
  rlboy_flat_env_cfg,
)
from mjlab.tasks.velocity.config.rlboy.recovery_assist import (
  RECOVERY_ASSIST_EVENT_NAME,
  RlBoyRecoveryAssist,
  actuator_torque_limit_excess_penalty,
  recovery_assist_curriculum,
  recovery_assist_reward_weight_curriculum,
  recovery_success_bonus,
)
from mjlab.tasks.velocity.config.rlboy.rl_cfg import rlboy_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.posture_phase import PosturePhaseEstimatorCfg


def test_flat_rlboy_enables_recovery_assist_only_during_training() -> None:
  train_cfg = rlboy_flat_env_cfg()
  play_cfg = rlboy_flat_env_cfg(play=True)

  assist_cfg = train_cfg.events[RECOVERY_ASSIST_EVENT_NAME]
  assert assist_cfg.func is RlBoyRecoveryAssist
  assert assist_cfg.mode == "step"
  call_params = inspect.signature(RlBoyRecoveryAssist.__call__).parameters
  assert set(assist_cfg.params) <= set(call_params)
  assert assist_cfg.params["asset_cfg"].body_names == ("waist_yaw_link",)
  assert assist_cfg.params["force_ranges"] == (
    (50.0, 50.0),
    (40.0, 45.0),
    (32.0, 38.0),
    (25.0, 30.0),
    (20.0, 24.0),
    (12.0, 19.0),
    (6.0, 11.0),
    (0.0, 5.0),
    (0.0, 0.0),
  )
  assert assist_cfg.params["posture_cfg"] == PosturePhaseEstimatorCfg()
  assert assist_cfg.params["frame_files"] == (
    "getup*.csv",
    "fall*.csv",
  )
  assert assist_cfg.params["source_names"] == ("getup", "fall", "canonical")
  assert assist_cfg.params["pose_stage_source_weights"] == (
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.15, 0.25, 0.6),
  )
  assert assist_cfg.params["angle_noise_ramp_attempts"] == 300
  assert assist_cfg.params["recovery_stage_probabilities"] == (0.6, 0.5, 0.4)
  assert assist_cfg.params["post_stage_recovery_probability"] == 0.35
  assert assist_cfg.params["low_force_recovery_probability"] == 0.3
  assert assist_cfg.params["recovery_probability_limits"] == (0.25, 0.65)
  assert assist_cfg.params["recovery_probability_min_attempts"] == 50
  assert assist_cfg.params["frame_dir"].endswith("motions72/motions/getup_frame_data")
  assert len(assist_cfg.params["csv_joint_names"]) == 20
  for removed_param in (
    "upright_height",
    "upright_angle",
    "fall_height",
    "fall_angle",
  ):
    assert removed_param not in assist_cfg.params
  assert assist_cfg.params["fall_confirm_s"] == 0.12
  assert assist_cfg.params["upright_hold_s"] == 0.5
  assert assist_cfg.params["force_ramp_up_s"] == 0.3
  assert assist_cfg.params["force_ramp_down_s"] == 0.5
  assert assist_cfg.params["root_height_range"] == (0.1, 0.13)
  assert assist_cfg.params["root_lin_vel_range"] == (-0.1, 0.1)
  assert assist_cfg.params["root_ang_vel_range"] == (-0.2, 0.2)
  assert assist_cfg.params["joint_position_ranges"]["head_yaw_joint"] == (0.0, 0.0)
  assert assist_cfg.params["joint_velocity_ranges"][r".*(shoulder|elbow).*"] == (
    -0.4,
    0.4,
  )
  assert "command_name" not in assist_cfg.params
  assert train_cfg.curriculum["recovery_assist"].func is recovery_assist_curriculum
  assert train_cfg.curriculum["recovery_assist"].params["window_size"] == 500
  assert train_cfg.curriculum["recovery_assist"].params["success_threshold"] == 0.9
  assert (
    train_cfg.rewards["continuous_torque_excess"].func
    is actuator_torque_limit_excess_penalty
  )
  assert train_cfg.rewards["continuous_torque_excess"].weight == 0.0
  assert train_cfg.rewards["peak_torque_saturation"].weight == 0.0
  assert train_cfg.rewards["peak_torque_saturation"].params["threshold_ratio"] == 0.85
  torque_curriculum = train_cfg.curriculum["torque_penalties"]
  assert torque_curriculum.func is recovery_assist_reward_weight_curriculum
  assert torque_curriculum.params["assist_level"] == 6
  assert torque_curriculum.params["assist_weights"] == {
    "continuous_torque_excess": -0.02,
    "peak_torque_saturation": -0.01,
  }
  assert torque_curriculum.params["complete_weights"] == {
    "continuous_torque_excess": -0.05,
    "peak_torque_saturation": -0.02,
  }
  assert not train_cfg.curriculum["command_vel"].log
  assert "mean_action_acc" not in train_cfg.metrics
  for reward_name in (
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
    "self_collisions",
  ):
    assert not train_cfg.rewards[reward_name].log
  assert "fell_over" not in train_cfg.terminations
  assert train_cfg.rewards["track_linear_velocity"].func is gated_track_linear_velocity
  assert (
    train_cfg.rewards["track_angular_velocity"].func is gated_track_angular_velocity
  )
  assert train_cfg.rewards["action_rate_l2"].weight == -0.03
  potential_cfg = train_cfg.rewards["recovery_potential_progress"]
  assert potential_cfg.func is recovery_potential_progress
  assert potential_cfg.weight == 2.0
  assert potential_cfg.params == {
    "event_name": RECOVERY_ASSIST_EVENT_NAME,
    "height_weight": 0.6,
    "upright_weight": 0.4,
  }
  assert train_cfg.rewards["recovery_time"].func is recovery_time_penalty
  assert train_cfg.rewards["recovery_time"].weight == -0.05
  assert train_cfg.rewards["recovery_success"].func is recovery_success_bonus
  assert train_cfg.rewards["recovery_success"].weight == 1.0
  assert train_cfg.rewards["upright"].func is gated_upright
  assert train_cfg.rewards["upright"].params["gate_min_scale"] == 0.5
  assert train_cfg.rewards["pose"].func is gated_variable_posture
  assert train_cfg.rewards["pose"].params["gate_min_scale"] == 0.1
  assert train_cfg.rewards["action_rate_l2"].params["gate_min_scale"] == 0.3
  assert train_cfg.rewards["air_time"].params["gate_min_scale"] == 0.0
  assert train_cfg.rewards["foot_slip"].params["gate_min_scale"] == 0.1
  assert train_cfg.rewards["base_height_recovery"].params["recovery_event_name"] == (
    RECOVERY_ASSIST_EVENT_NAME
  )
  for removed_reward in (
    "base_height_recovery_success",
    "recovery_progress",
    "fallen_duration",
  ):
    assert removed_reward not in train_cfg.rewards
  assert train_cfg.rewards["recovery_failure"].weight == -2.0
  assert train_cfg.rewards["recovery_failure"].params == {
    "event_name": RECOVERY_ASSIST_EVENT_NAME
  }

  assert "randomize_fallen_pose" not in train_cfg.events
  assert next(iter(train_cfg.events)) == "prepare_recovery_group"
  assert "push_robot" in train_cfg.events
  assert "knockdown_robot" in train_cfg.events
  knockdown_cfg = train_cfg.events["knockdown_robot"]
  assert knockdown_cfg.interval_range_s == (13.0, 15.0)
  assert knockdown_cfg.params["stages"][0]["velocity_range"] == {
    "x": (-1.5, 1.5),
    "y": (-1.5, 1.5),
    "roll": (-2.5, 2.5),
    "pitch": (-2.5, 2.5),
    "yaw": (-1.0, 1.0),
  }
  assert "base_payload" in train_cfg.events
  assert RECOVERY_ASSIST_EVENT_NAME not in play_cfg.events
  assert "recovery_assist" not in play_cfg.curriculum
  assert "command_vel" not in play_cfg.curriculum
  play_twist = play_cfg.commands["twist"]
  assert isinstance(play_twist, UniformVelocityCommandCfg)
  assert play_twist.ranges.lin_vel_x == (-1.0, 1.5)
  assert play_twist.ranges.ang_vel_z == (-0.5, 0.5)
  assert "fell_over" not in play_cfg.terminations
  assert set(play_cfg.terminations) == {"time_out"}


def _make_recovery_reward_env(
  heights: torch.Tensor,
  projected_gravity: torch.Tensor,
  active: torch.Tensor,
) -> tuple[Any, RlBoyRecoveryAssist]:
  assist = RlBoyRecoveryAssist.__new__(RlBoyRecoveryAssist)
  assist.assist_active = active
  assist.just_succeeded = torch.zeros_like(active)
  asset = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.stack(
        (torch.zeros_like(heights), torch.zeros_like(heights), heights), dim=1
      ),
      projected_gravity_b=projected_gravity,
      default_root_state=torch.tensor(((0.0, 0.0, 0.41, 1.0, 0.0, 0.0, 0.0),)).repeat(
        len(heights), 1
      ),
    )
  )
  env = SimpleNamespace(
    num_envs=len(heights),
    device="cpu",
    step_dt=0.02,
    scene={"robot": asset},
    event_manager=SimpleNamespace(
      get_term_cfg=lambda _name: SimpleNamespace(func=assist)
    ),
  )
  return cast(Any, env), assist


def test_recovery_upright_score_is_zero_at_roll_or_pitch_90_degrees() -> None:
  env, _ = _make_recovery_reward_env(
    heights=torch.full((4,), 0.10),
    projected_gravity=torch.tensor(
      (
        (0.0, 0.0, -1.0),  # Upright.
        (0.0, -1.0, 0.0),  # 90-degree roll.
        (1.0, 0.0, 0.0),  # 90-degree pitch.
        (0.0, 0.0, 1.0),  # Upside down.
      )
    ),
    active=torch.ones(4, dtype=torch.bool),
  )

  upright_score = recovery_state_potential(
    env,
    height_weight=0.6,
    upright_weight=0.4,
  )

  expected_upright = 0.6 * (0.10 / 0.41) + 0.4
  torch.testing.assert_close(
    upright_score, torch.tensor((expected_upright, 0.0, 0.0, 0.0))
  )


def test_recovery_potential_progress_has_no_activation_spike() -> None:
  env, assist = _make_recovery_reward_env(
    heights=torch.tensor((0.10,)),
    projected_gravity=torch.tensor(((0.0, -1.0, 0.0),)),
    active=torch.ones(1, dtype=torch.bool),
  )
  params = {
    "event_name": RECOVERY_ASSIST_EVENT_NAME,
    "height_weight": 0.6,
    "upright_weight": 0.4,
  }
  cfg = RewardTermCfg(func=recovery_potential_progress, weight=2.0, params=params)
  progress = recovery_potential_progress(cfg, env)

  assert progress(env, **params) == 0.0

  env.scene["robot"].data.root_link_pos_w[:, 2] = 0.24
  env.scene["robot"].data.projected_gravity_b[:, 2] = -0.5
  improvement = progress(env, **params)
  assert improvement > 0.0

  env.scene["robot"].data.root_link_pos_w[:, 2] = 0.10
  env.scene["robot"].data.projected_gravity_b[:, 2] = 0.0
  regression = progress(env, **params)
  torch.testing.assert_close(regression, -improvement)

  assist.assist_active[:] = False
  assert progress(env, **params) == 0.0
  assist.assist_active[:] = True
  env.scene["robot"].data.root_link_pos_w[:, 2] = 0.38
  assert progress(env, **params) == 0.0


def test_recovery_time_and_success_terms_use_assist_state() -> None:
  env, assist = _make_recovery_reward_env(
    heights=torch.full((2,), 0.10),
    projected_gravity=torch.tensor(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0))),
    active=torch.tensor((True, False)),
  )
  assist.just_succeeded[:] = torch.tensor((False, True))

  torch.testing.assert_close(
    recovery_time_penalty(env, RECOVERY_ASSIST_EVENT_NAME),
    torch.tensor((1.0, 0.0)),
  )
  torch.testing.assert_close(
    recovery_success_bonus(env, RECOVERY_ASSIST_EVENT_NAME),
    torch.tensor((0.0, 50.0)),
  )


def test_recovery_pose_stages_advance_before_assistance_level() -> None:
  assist = RlBoyRecoveryAssist.__new__(RlBoyRecoveryAssist)
  assist.level = 0
  assist.pose_stage = 0
  assist._pose_stage_source_weights = torch.zeros(3, 3)
  assist._force_ranges = torch.tensor(((50.0, 50.0), (40.0, 45.0), (32.0, 38.0)))
  assist.attempts = torch.tensor(500)
  assist.successes = torch.tensor(450)

  assist.update_level(window_size=500, success_threshold=0.9)
  assert assist.pose_stage == 1
  assert assist.level == 0
  assert assist.attempts == 0

  assist.attempts = torch.tensor(500)
  assist.successes = torch.tensor(449)
  assist.update_level(window_size=500, success_threshold=0.9)
  assert assist.pose_stage == 1
  assert assist.attempts == 0

  assist.attempts = torch.tensor(500)
  assist.successes = torch.tensor(450)
  assist.update_level(window_size=500, success_threshold=0.9)
  assert assist.pose_stage == 2

  assist.attempts = torch.tensor(500)
  assist.successes = torch.tensor(475)
  assist.update_level(window_size=500, success_threshold=0.9)
  assert assist.pose_stage == 2
  assert assist.level == 1


def test_torque_limit_penalty_only_counts_excess() -> None:
  asset = SimpleNamespace(
    actuator_names=(
      "left_shoulder_pitch_joint",
      "left_hip_pitch_joint",
      "left_ankle_pitch_joint",
    ),
    data=SimpleNamespace(
      actuator_force=torch.tensor(
        (
          (0.8, 8.0, 4.0),
          (1.6, 12.0, 2.0),
        )
      )
    ),
  )
  env = SimpleNamespace(
    device="cpu",
    scene={"robot": asset},
    extras={"log": {}},
  )
  cfg = RewardTermCfg(
    func=actuator_torque_limit_excess_penalty,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_ids=[0, 1, 2]),
      "limit_by_actuator": {
        r".*shoulder.*": 0.8,
        r".*hip.*": 8.0,
        r".*ankle.*": 4.0,
      },
    },
  )

  penalty = actuator_torque_limit_excess_penalty(cfg, cast(Any, env))

  assert torch.allclose(
    penalty(cast(Any, env), **cfg.params), torch.tensor((0.0, 1.25))
  )


def test_peak_torque_penalty_starts_near_saturation() -> None:
  asset = SimpleNamespace(
    actuator_names=(
      "left_shoulder_pitch_joint",
      "left_hip_pitch_joint",
      "left_ankle_pitch_joint",
    ),
    data=SimpleNamespace(
      actuator_force=torch.tensor(
        (
          (3.0 * 0.85, 20.0 * 0.85, 11.0 * 0.85),
          (3.0, 20.0, 11.0),
        )
      )
    ),
  )
  env = SimpleNamespace(
    device="cpu",
    scene={"robot": asset},
    extras={"log": {}},
  )
  cfg = RewardTermCfg(
    func=actuator_torque_limit_excess_penalty,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_ids=[0, 1, 2]),
      "limit_by_actuator": {
        r".*shoulder.*": 3.0,
        r".*hip.*": 20.0,
        r".*ankle.*": 11.0,
      },
      "threshold_ratio": 0.85,
    },
  )

  penalty = actuator_torque_limit_excess_penalty(cfg, cast(Any, env))

  assert torch.allclose(
    penalty(cast(Any, env), **cfg.params), torch.tensor((0.0, 3 * 0.15**2))
  )


def test_torque_penalty_curriculum_tracks_recovery_assist_progress() -> None:
  assist = RlBoyRecoveryAssist.__new__(RlBoyRecoveryAssist)
  assist.pose_stage = 2
  assist.level = 5
  assist._force_ranges = torch.tensor(
    (
      (50.0, 50.0),
      (40.0, 45.0),
      (32.0, 38.0),
      (25.0, 30.0),
      (20.0, 24.0),
      (12.0, 19.0),
      (6.0, 11.0),
      (0.0, 5.0),
      (0.0, 0.0),
    )
  )
  reward_cfgs: dict[str, Any] = {
    "continuous_torque_excess": SimpleNamespace(weight=123.0),
    "peak_torque_saturation": SimpleNamespace(weight=456.0),
  }
  env = SimpleNamespace(
    device="cpu",
    event_manager=SimpleNamespace(
      get_term_cfg=lambda name: SimpleNamespace(func=assist)
    ),
    reward_manager=SimpleNamespace(get_term_cfg=lambda name: reward_cfgs[name]),
  )

  state = recovery_assist_reward_weight_curriculum(
    cast(Any, env),
    torch.arange(1),
    event_name=RECOVERY_ASSIST_EVENT_NAME,
    assist_level=6,
    assist_weights={
      "continuous_torque_excess": -0.02,
      "peak_torque_saturation": -0.01,
    },
    complete_weights={
      "continuous_torque_excess": -0.05,
      "peak_torque_saturation": -0.02,
    },
  )

  assert state["active"] == 0.0
  assert state["complete"] == 0.0
  assert reward_cfgs["continuous_torque_excess"].weight == 0.0
  assert reward_cfgs["peak_torque_saturation"].weight == 0.0

  assist.level = 6
  state = recovery_assist_reward_weight_curriculum(
    cast(Any, env),
    torch.arange(1),
    event_name=RECOVERY_ASSIST_EVENT_NAME,
    assist_level=6,
    assist_weights={
      "continuous_torque_excess": -0.02,
      "peak_torque_saturation": -0.01,
    },
    complete_weights={
      "continuous_torque_excess": -0.05,
      "peak_torque_saturation": -0.02,
    },
  )

  assert state["active"] == 1.0
  assert state["complete"] == 0.0
  assert reward_cfgs["continuous_torque_excess"].weight == -0.02
  assert reward_cfgs["peak_torque_saturation"].weight == -0.01

  assist.level = 8
  state = recovery_assist_reward_weight_curriculum(
    cast(Any, env),
    torch.arange(1),
    event_name=RECOVERY_ASSIST_EVENT_NAME,
    assist_level=6,
    assist_weights={
      "continuous_torque_excess": -0.02,
      "peak_torque_saturation": -0.01,
    },
    complete_weights={
      "continuous_torque_excess": -0.05,
      "peak_torque_saturation": -0.02,
    },
  )

  assert state["active"] == 1.0
  assert state["complete"] == 1.0
  assert reward_cfgs["continuous_torque_excess"].weight == -0.05
  assert reward_cfgs["peak_torque_saturation"].weight == -0.02


def test_recovery_angle_noise_is_disabled_then_smoothly_enabled() -> None:
  assist = RlBoyRecoveryAssist.__new__(RlBoyRecoveryAssist)
  assist._joint_position_ranges = torch.tensor(((-1.0, 1.0), (2.0, 2.0)))
  joint_pos = torch.zeros(4, 2)

  assist.pose_stage = 0
  assert torch.equal(
    assist._sample_joint_position_noise(joint_pos), torch.zeros_like(joint_pos)
  )

  assist.pose_stage = 1
  assist._angle_noise_ramp_attempts = 300
  assist.attempts = torch.tensor(300)
  assist.successes = torch.tensor(210)
  stage_two_noise = assist._sample_joint_position_noise(joint_pos)
  assert torch.allclose(stage_two_noise[:, 1], torch.ones(4))

  assist.pose_stage = 2
  stage_three_noise = assist._sample_joint_position_noise(joint_pos)
  assert torch.allclose(stage_three_noise[:, 1], torch.full((4,), 2.0))


def test_recovery_csv_quaternion_is_reordered_and_normalized() -> None:
  assist = RlBoyRecoveryAssist.__new__(RlBoyRecoveryAssist)
  assist._env = cast(Any, SimpleNamespace(device="cpu"))
  assist.sample_source = torch.tensor((0,))
  assist._csv_frames = (
    torch.tensor([[0.0, 0.0, 0.2, 1.0, 2.0, 3.0, 4.0, *([0.0] * 20)]]),
  )
  assist._canonical_source = 1
  assist._poses = [{"pos": (0.0, 0.0, 0.1), "quat": (1.0, 0.0, 0.0, 0.0)}]
  assist._root_height_range = (0.1, 0.13)

  root_pos, root_quat, _ = assist._sample_initial_states(torch.tensor((0,)))

  assert torch.equal(root_pos, torch.tensor(((0.0, 0.0, 0.2),)))
  expected = torch.tensor(((4.0, 1.0, 2.0, 3.0),))
  expected /= expected.norm(dim=-1, keepdim=True)
  assert torch.allclose(root_quat, expected)


def test_recovery_group_probability_tracks_stage_and_success() -> None:
  assist = RlBoyRecoveryAssist.__new__(RlBoyRecoveryAssist)
  assist._recovery_stage_probabilities = (0.6, 0.5, 0.4)
  assist._post_stage_recovery_probability = 0.35
  assist._low_force_recovery_probability = 0.3
  assist._recovery_probability_limits = (0.25, 0.65)
  assist._recovery_probability_feedback_gain = 0.5
  assist._recovery_probability_smoothing = 0.1
  assist._recovery_probability_min_attempts = 50
  assist._force_ranges = torch.zeros(9, 2)
  assist.pose_stage = 0
  assist.level = 0
  assist._recovery_probability = 0.6
  assist._target_recovery_probability = 0.6
  assist.attempts = torch.tensor(0)
  assist.successes = torch.tensor(0)

  assist.update_recovery_probability(success_target=0.9)
  assert assist._target_recovery_probability == 0.6

  assist.pose_stage = 1
  assist.attempts = torch.tensor(100)
  assist.successes = torch.tensor(60)
  assist.update_recovery_probability(success_target=0.9)
  assert torch.isclose(
    torch.tensor(assist._target_recovery_probability), torch.tensor(0.6)
  )
  assert torch.isclose(torch.tensor(assist._recovery_probability), torch.tensor(0.6))

  assist.pose_stage = 2
  assist.level = 0
  assist.attempts = torch.tensor(0)
  assist.update_recovery_probability(success_target=0.9)
  assert torch.isclose(
    torch.tensor(assist._target_recovery_probability), torch.tensor(0.4)
  )

  assist.level = 1
  assist.attempts = torch.tensor(100)
  assist.successes = torch.tensor(100)
  assist.update_recovery_probability(success_target=0.9)
  assert torch.isclose(
    torch.tensor(assist._target_recovery_probability), torch.tensor(0.3)
  )
  assert torch.isclose(torch.tensor(assist._recovery_probability), torch.tensor(0.552))

  assist.level = 8
  assist.attempts = torch.tensor(0)
  assist.update_recovery_probability(success_target=0.9)
  assert torch.isclose(
    torch.tensor(assist._target_recovery_probability), torch.tensor(0.3)
  )


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
    0.125,
    0.25,
    0.5,
    1.0,
  ]
  assert [
    max(
      (max(abs(low), abs(high)) for low, high in stage["velocity_range"].values()),
      default=0.0,
    )
    for stage in randomization_stages
  ] == [
    0.0,
    0.0,
    0.06,
    0.16,
    0.5,
  ]
