"""Tests for RL_BOY actuator configuration."""

import pytest

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots.RL_BOY.rlboy_constants import (
  ACTUATOR_J3507,
  ACTUATOR_J6006,
  ACTUATOR_J8006,
  DAMPING_J3507,
  DAMPING_J6006,
  DAMPING_J8006,
  DAMPING_RATIO,
  DAMPING_RATIO_LEG,
  DOWN_LYING_KEYFRAME,
  HOME_KEYFRAME,
  LEFT_LYING_KEYFRAME,
  NATURAL_FREQ,
  RL_BOY_ACTION_SCALE,
  RL_BOY_ACTUATOR_ARM,
  RL_BOY_ACTUATOR_HIP_PITCH_KNEE,
  RL_BOY_ACTUATOR_HIP_YAW_ROLL,
  RL_BOY_ACTUATOR_WAIST_FOOT,
  RL_BOY_JOINT_POS_LIMITS,
  UP_LYING_KEYFRAME,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.tracking.config.rlboy.env_cfgs import rlboy_flat_tracking_env_cfg
from mjlab.tasks.velocity.config.rlboy.env_cfgs import (
  _FALLEN_POSES,
  rlboy_rough_env_cfg,
)


def test_rlboy_uses_dc_motor_peak_torque_speed_envelopes() -> None:
  actuator_groups = (
    (RL_BOY_ACTUATOR_ARM, ACTUATOR_J3507),
    (RL_BOY_ACTUATOR_HIP_YAW_ROLL, ACTUATOR_J6006),
    (RL_BOY_ACTUATOR_HIP_PITCH_KNEE, ACTUATOR_J8006),
    (RL_BOY_ACTUATOR_WAIST_FOOT, ACTUATOR_J6006),
  )

  for cfg, motor in actuator_groups:
    assert isinstance(cfg, DcMotorActuatorCfg)
    assert cfg.effort_limit == motor.effort_limit
    assert cfg.saturation_effort == motor.effort_limit
    assert cfg.velocity_limit == motor.velocity_limit
    assert cfg.armature == motor.reflected_inertia


def test_rlboy_group_damping_ratios() -> None:
  damping_groups = (
    (DAMPING_J3507, DAMPING_RATIO, ACTUATOR_J3507),
    (DAMPING_J6006, DAMPING_RATIO, ACTUATOR_J6006),
    (DAMPING_J8006, DAMPING_RATIO_LEG, ACTUATOR_J8006),
  )

  for damping, ratio, motor in damping_groups:
    expected = 2.0 * ratio * motor.reflected_inertia * NATURAL_FREQ
    assert damping == pytest.approx(expected)


def test_rlboy_group_action_scale_weights() -> None:
  for name in RL_BOY_ACTUATOR_ARM.target_names_expr:
    expected = 0.5 * ACTUATOR_J3507.effort_limit / RL_BOY_ACTUATOR_ARM.stiffness
    assert RL_BOY_ACTION_SCALE[name] == pytest.approx(expected)

  for cfg, motor in (
    (RL_BOY_ACTUATOR_HIP_YAW_ROLL, ACTUATOR_J6006),
    (RL_BOY_ACTUATOR_HIP_PITCH_KNEE, ACTUATOR_J8006),
  ):
    expected = 0.7 * motor.effort_limit / cfg.stiffness
    for name in cfg.target_names_expr:
      assert RL_BOY_ACTION_SCALE[name] == pytest.approx(expected)

  for name in RL_BOY_ACTUATOR_WAIST_FOOT.target_names_expr:
    if name == "head_yaw_joint":
      continue
    expected = 0.5 * ACTUATOR_J6006.effort_limit / RL_BOY_ACTUATOR_WAIST_FOOT.stiffness
    assert RL_BOY_ACTION_SCALE[name] == pytest.approx(expected)

  assert RL_BOY_ACTION_SCALE["head_yaw_joint"] == 0.0


def test_rlboy_tasks_clip_policy_targets_to_reference_joint_limits() -> None:
  for cfg in (rlboy_rough_env_cfg(), rlboy_flat_tracking_env_cfg()):
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    assert joint_pos_action.clip == RL_BOY_JOINT_POS_LIMITS
    assert cfg.scene.entities["robot"].init_state == HOME_KEYFRAME


def test_rlboy_velocity_recovery_poses_use_reference_keyframes() -> None:
  assert _FALLEN_POSES == tuple(
    {"pos": keyframe.pos, "quat": keyframe.rot}
    for keyframe in (
      UP_LYING_KEYFRAME,
      DOWN_LYING_KEYFRAME,
      LEFT_LYING_KEYFRAME,
    )
  )
