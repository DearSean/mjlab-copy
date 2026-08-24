"""Tests specific to velocity tasks."""

from types import SimpleNamespace

import pytest
import torch

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  QLMINI2_ACTION_SCALE,
  RL_BOY_ACTION_SCALE,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity.mdp import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
  adaptive_velocity_progression,
)


@pytest.fixture(scope="module")
def velocity_task_ids() -> list[str]:
  """Get all velocity task IDs."""
  return [t for t in list_tasks() if "Velocity" in t]


@pytest.fixture(scope="module")
def rough_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all rough terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Rough" in t]


@pytest.fixture(scope="module")
def flat_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all flat terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Flat" in t]


def test_velocity_tasks_have_twist_command(velocity_task_ids: list[str]) -> None:
  """All velocity tasks should have a velocity command."""
  for task_id in velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "twist" in cfg.commands, f"Task {task_id} missing 'twist' command"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg), (
      f"Task {task_id} twist command is not UniformVelocityCommandCfg"
    )


def test_qlmini2_uses_categorical_direct_velocity_commands() -> None:
  cfg = load_env_cfg("Mjlab-Velocity-Flat-qlmini2")
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  assert twist_cmd.heading_command is False
  assert twist_cmd.rel_heading_envs == 0.0
  assert twist_cmd.mode_probabilities is not None
  assert twist_cmd.mode_probabilities.values() == (0.10, 0.15, 0.15, 0.15, 0.20, 0.25)
  twist_cmd.validate()
  progression = cfg.curriculum["command_vel"]
  assert progression.func is adaptive_velocity_progression
  stages = progression.params["stages"]
  assert [stage["step"] for stage in stages] == [
    0,
    500 * 24,
    1000 * 24,
    1750 * 24,
    2400 * 24,
    3000 * 24,
  ]
  assert stages[0]["mode_probabilities"]["forward"] == 0.60
  assert stages[0]["mode_probabilities"]["backward"] == 0.0
  assert stages[-1]["lin_vel_x"] == (-0.80, 1.20)


def test_qlmini2_gait_rewards_discourage_shuffling() -> None:
  cfg = load_env_cfg("Mjlab-Velocity-Flat-qlmini2")
  assert cfg.rewards["air_time"].params["command_threshold"] == 0.05
  assert cfg.rewards["foot_clearance"].weight == -0.75
  assert cfg.rewards["foot_clearance"].params["normalize_by_target"] is True
  assert cfg.rewards["foot_swing_height"].weight == -0.75
  assert cfg.rewards["foot_slip"].weight == -0.35
  assert cfg.rewards["foot_slip"].params["l1_weight"] == 0.10
  assert cfg.rewards["soft_landing"].weight == 0.0
  assert cfg.rewards["soft_landing"].params["force_threshold"] == 200.0
  assert cfg.rewards["soft_landing"].params["force_scale"] == 140.0
  assert cfg.rewards["soft_landing"].params["squared"] is True
  stages = cfg.curriculum["command_vel"].params["stages"]
  assert stages[0]["reward_weights"]["soft_landing"] == 0.0
  assert stages[-1]["reward_weights"]["soft_landing"] == -0.10


def test_qlmini2_penalizes_requested_torque_beyond_motor_limits() -> None:
  cfg = load_env_cfg("Mjlab-Velocity-Flat-qlmini2")
  continuous = cfg.rewards["torque_continuous_excess"]
  peak = cfg.rewards["torque_peak_usage"]
  assert continuous.weight == -0.05
  assert continuous.params["limit"] == "continuous"
  assert peak.weight == -0.02
  assert peak.params["limit"] == "peak"
  assert peak.params["peak_threshold"] == 0.95


def test_qlmini2_pushes_follow_the_adaptive_curriculum() -> None:
  cfg = load_env_cfg("Mjlab-Velocity-Flat-qlmini2")
  push_stages = cfg.curriculum["command_vel"].params["stages"]
  assert [stage["step"] for stage in push_stages] == [
    0,
    500 * 24,
    1000 * 24,
    1750 * 24,
    2400 * 24,
    3000 * 24,
  ]
  assert push_stages[0]["velocity_range"]["x"] == (0.0, 0.0)
  assert push_stages[1]["interval_range_s"] == (10.0, 14.0)
  assert push_stages[-1]["velocity_range"]["yaw"] == (-0.78, 0.78)


def test_qlmini2_categorical_commands_cover_pure_axes() -> None:
  cfg = load_env_cfg("Mjlab-Velocity-Flat-qlmini2").commands["twist"]
  assert isinstance(cfg, UniformVelocityCommandCfg)

  num_samples = 20_000
  command = object.__new__(UniformVelocityCommand)
  command.cfg = cfg
  command._env = SimpleNamespace(device="cpu")
  command.vel_command_b = torch.zeros(num_samples, 3)
  command.vel_command_w = torch.zeros_like(command.vel_command_b)
  command.is_heading_env = torch.zeros(num_samples, dtype=torch.bool)
  command.is_standing_env = torch.zeros(num_samples, dtype=torch.bool)
  command.is_world_env = torch.zeros(num_samples, dtype=torch.bool)
  command.is_forward_env = torch.zeros(num_samples, dtype=torch.bool)
  command._resample_categorical_command(torch.arange(num_samples))

  sampled = command.vel_command_b
  groups = {
    "standing": (sampled == 0).all(dim=1),
    "forward": (sampled[:, 0] > 0) & (sampled[:, 1:] == 0).all(dim=1),
    "backward": (sampled[:, 0] < 0) & (sampled[:, 1:] == 0).all(dim=1),
    "lateral": (sampled[:, 1] != 0) & (sampled[:, (0, 2)] == 0).all(dim=1),
    "yaw": (sampled[:, 2] != 0) & (sampled[:, :2] == 0).all(dim=1),
  }
  groups["mixed"] = ~torch.stack(tuple(groups.values())).any(dim=0)
  expected = (0.10, 0.15, 0.15, 0.15, 0.20, 0.25)
  for sampled_fraction, target_fraction in zip(
    (group.float().mean().item() for group in groups.values()), expected, strict=True
  ):
    assert sampled_fraction == pytest.approx(target_fraction, abs=0.02)
  assert torch.equal(command.vel_command_b, command.vel_command_w)


def test_velocity_task_set_is_supported(velocity_task_ids: list[str]) -> None:
  assert set(velocity_task_ids) == {
    "Mjlab-Velocity-Flat",
    "Mjlab-Velocity-Rough",
    "Mjlab-Velocity-Flat-Recovery",
    "Mjlab-Velocity-Rough-Recovery",
    "Mjlab-Velocity-Flat-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Flat-qlmini2",
  }


def test_velocity_tasks_have_required_sensors(velocity_task_ids: list[str]) -> None:
  """Velocity tasks should have feet/ground and self collision sensors."""
  for task_id in velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_flat_velocity_tasks_have_plane_terrain(
  flat_velocity_task_ids: list[str],
) -> None:
  """Flat velocity tasks should have terrain_type='plane' and no terrain_generator."""
  for task_id in flat_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "plane", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, expected 'plane'"
    )
    assert cfg.scene.terrain.terrain_generator is None, (
      f"Task {task_id} has terrain_generator, expected None for flat terrain"
    )


def test_rough_velocity_tasks_have_generator_terrain(
  rough_velocity_task_ids: list[str],
) -> None:
  """Rough velocity tasks should have generator terrain."""
  for task_id in rough_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "generator", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, "
      "expected 'generator'"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator, expected one for rough terrain"
    )


def test_rough_velocity_training_has_curriculum_enabled() -> None:
  """Rough velocity training tasks should have terrain curriculum enabled."""
  rough_training_tasks = [
    "Mjlab-Velocity-Rough",
    "Mjlab-Velocity-Rough-Recovery",
    "Mjlab-Velocity-Rough-Unitree-G1",
  ]

  for task_id in rough_training_tasks:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is True, (
      f"Task {task_id} curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected True"
    )


def test_rough_velocity_play_has_curriculum_disabled() -> None:
  """Rough velocity play tasks should have terrain curriculum disabled."""
  rough_training_tasks = [
    "Mjlab-Velocity-Rough",
    "Mjlab-Velocity-Rough-Recovery",
    "Mjlab-Velocity-Rough-Unitree-G1",
  ]

  for task_id in rough_training_tasks:
    cfg = load_env_cfg(task_id, play=True)

    assert cfg.scene.terrain is not None, (
      f"Task {task_id} (play mode) has no terrain config"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} (play mode) has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is False, (
      f"Task {task_id} (play mode) curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected False"
    )


def test_velocity_tasks_have_correct_action_scale(
  velocity_task_ids: list[str],
) -> None:
  """Velocity tasks should use the action scale of their configured robot."""
  for task_id in velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    if task_id.endswith("-Unitree-G1"):
      expected_scale = G1_ACTION_SCALE
    elif task_id.endswith("-qlmini2"):
      expected_scale = QLMINI2_ACTION_SCALE
    else:
      expected_scale = RL_BOY_ACTION_SCALE
    assert joint_pos_action.scale == expected_scale, (
      f"Task {task_id} action scale mismatch"
    )


def test_only_recovery_tasks_enable_recovery(velocity_task_ids: list[str]) -> None:
  for task_id in velocity_task_ids:
    cfg = load_env_cfg(task_id)
    if task_id.endswith("-Recovery"):
      assert "recovery_assist" in cfg.events
      assert "fell_over" not in cfg.terminations
    else:
      assert "recovery_assist" not in cfg.events
      assert "fell_over" in cfg.terminations
