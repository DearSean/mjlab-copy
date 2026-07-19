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
  DAMPING_RATIO_J3507,
  DAMPING_RATIO_J6006,
  DAMPING_RATIO_J8006,
  NATURAL_FREQ,
  RL_BOY_ACTION_SCALE,
  RL_BOY_ACTUATOR_ARM,
  RL_BOY_ACTUATOR_LEG,
  RL_BOY_ACTUATOR_WAIST_FOOT,
)


def test_rlboy_uses_dc_motor_peak_torque_speed_envelopes() -> None:
  actuator_groups = (
    (RL_BOY_ACTUATOR_ARM, ACTUATOR_J3507),
    (RL_BOY_ACTUATOR_LEG, ACTUATOR_J8006),
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
    (DAMPING_J3507, DAMPING_RATIO_J3507, ACTUATOR_J3507),
    (DAMPING_J6006, DAMPING_RATIO_J6006, ACTUATOR_J6006),
    (DAMPING_J8006, DAMPING_RATIO_J8006, ACTUATOR_J8006),
  )

  for damping, ratio, motor in damping_groups:
    expected = 2.0 * ratio * motor.reflected_inertia * NATURAL_FREQ
    assert damping == pytest.approx(expected)


def test_rlboy_group_action_scale_weights() -> None:
  for name in RL_BOY_ACTUATOR_ARM.target_names_expr:
    expected = 0.5 * ACTUATOR_J3507.effort_limit / RL_BOY_ACTUATOR_ARM.stiffness
    assert RL_BOY_ACTION_SCALE[name] == pytest.approx(expected)

  for name in RL_BOY_ACTUATOR_LEG.target_names_expr:
    expected = 0.7 * ACTUATOR_J8006.effort_limit / RL_BOY_ACTUATOR_LEG.stiffness
    assert RL_BOY_ACTION_SCALE[name] == pytest.approx(expected)

  for name in RL_BOY_ACTUATOR_WAIST_FOOT.target_names_expr:
    if name == "head_yaw_joint":
      continue
    expected = 0.5 * ACTUATOR_J6006.effort_limit / RL_BOY_ACTUATOR_WAIST_FOOT.stiffness
    assert RL_BOY_ACTION_SCALE[name] == pytest.approx(expected)

  assert RL_BOY_ACTION_SCALE["head_yaw_joint"] == 0.0
