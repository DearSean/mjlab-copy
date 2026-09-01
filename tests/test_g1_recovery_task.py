"""Tests for G1 recovery standing success and shaping primitives."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlFpoAlgorithmCfg
from mjlab.tasks.velocity.config.g1.recovery_env_cfg import (
  unitree_g1_flat_recovery_env_cfg,
)
from mjlab.tasks.velocity.config.g1.recovery_events import (
  FALLEN_MODE,
  REFERENCE_MODE,
  STAND_MODE,
  G1RecoveryReset,
  g1_recovery_assist_curriculum,
)
from mjlab.tasks.velocity.config.g1.recovery_reference import G1ReferenceLibrary
from mjlab.tasks.velocity.config.g1.recovery_rewards import (
  RecoveryCompositeProgress,
  RecoveryMilestoneLift,
  recovery_failure_penalty,
  recovery_gate,
  recovery_hold_reward,
  recovery_progress_bin_metric,
  recovery_progress_bin_success_metric,
  recovery_progress_reward,
  recovery_succeeded,
  recovery_success_bonus,
)
from mjlab.tasks.velocity.config.g1.recovery_task import (
  G1StandingCfg,
  G1StandingTracker,
  potential_progress,
  standing_potential,
)
from mjlab.tasks.velocity.config.g1.rl_cfg import (
  unitree_g1_recovery_fpo_runner_cfg,
)
from mjlab.tasks.velocity.recovery_data.g1_schema import (
  G1_PHYSICAL_INIT_SCHEMA_VERSION,
)
from mjlab.tasks.velocity.recovery_prior.smp_reward import (
  G1SmpReward,
  normalized_esm_reward,
  progress_handoff_weight,
)


def test_standing_requires_a_continuous_25_step_hold():
  cfg = G1StandingCfg()
  tracker = G1StandingTracker(1, torch.device("cpu"), cfg)
  values = {
    "pelvis_height_m": torch.tensor([0.76]),
    "projected_gravity": torch.tensor([[0.0, 0.0, -1.0]]),
    "base_linear_velocity": torch.zeros(1, 3),
    "base_angular_velocity": torch.zeros(1, 3),
    "joint_velocity": torch.zeros(1, 29),
    "normalized_pose_error": torch.zeros(1, 29),
    "left_foot_contact": torch.tensor([True]),
    "right_foot_contact": torch.tensor([True]),
    "nonfoot_support": torch.tensor([False]),
  }
  for _ in range(24):
    standing, success = tracker.update(**values)
    assert standing.item()
    assert not success.item()
  _, success = tracker.update(**values)
  assert success.item()
  values["nonfoot_support"] = torch.tensor([True])
  standing, success = tracker.update(**values)
  assert not standing.item()
  assert not success.item()


def test_standing_potential_rewards_height_uprightness_and_pose():
  cfg = G1StandingCfg()
  low = standing_potential(
    torch.tensor([0.1]), torch.tensor([[0.0, 0.0, 1.0]]), torch.ones(1, 29), cfg
  )
  high = standing_potential(
    torch.tensor([0.76]), torch.tensor([[0.0, 0.0, -1.0]]), torch.zeros(1, 29), cfg
  )
  assert high > low
  assert potential_progress(high, low) > 0.0


def test_reference_library_masks_boundaries_and_samples_fallen(tmp_path: Path):
  clips_dir = tmp_path / "clips"
  clips_dir.mkdir()
  root_position = np.zeros((4, 3), dtype=np.float32)
  root_position[:, 2] = (0.1, 0.2, 0.7, 0.76)
  quaternion = np.zeros((4, 4), dtype=np.float32)
  quaternion[:, 3] = 1.0
  np.savez(
    clips_dir / "clip.npz",
    root_position=root_position,
    root_quaternion_xyzw=quaternion,
    root_linear_velocity=np.zeros((4, 3), dtype=np.float32),
    root_angular_velocity=np.zeros((4, 3), dtype=np.float32),
    joint_position=np.zeros((4, 29), dtype=np.float32),
    joint_velocity=np.zeros((4, 29), dtype=np.float32),
  )
  manifest = {
    "clips": [
      {
        "clip_id": "001_test",
        "split": "train",
        "file": "clips/clip.npz",
        "target_fps": 50.0,
        "source_fps": 25.0,
        "source_support_start_frame": 100,
        "source_recording_id": "test.csv",
      }
    ]
  }
  (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

  library = G1ReferenceLibrary(tmp_path, "cpu")
  command, valid = library.window(torch.tensor([0]), torch.tensor([0]))
  assert command.shape == (1, 21, 38)
  assert valid.sum().item() == 4
  assert not valid[0, :10].any()
  fallen_clip, fallen_frame = library.sample_fallen(64)
  assert not fallen_clip.any()
  assert torch.all(fallen_frame < 2)
  progress_clip, progress_frame = library.sample_progress(64, 0.55, 1.0)
  assert not progress_clip.any()
  assert torch.all(progress_frame >= 2)
  _, balanced_frame = library.sample_frontier_balanced(4096, 0.1, 0.5, 1.0, 0.5)
  frontier_ratio = (balanced_frame < 2).float().mean()
  assert abs(frontier_ratio.item() - 0.5) < 0.05
  assert library.confidence(torch.tensor([0]), torch.tensor([3])).item() == 0.0

  adaptive_library = G1ReferenceLibrary(tmp_path, "cpu", temporal_bin_duration_s=0.04)
  assert adaptive_library.num_temporal_bins == 2
  difficulty = torch.tensor((100.0, 1.0))
  adaptive_clip, adaptive_frame = adaptive_library.sample_progress(
    4096, 0.1, 1.0, difficulty, uniform_probability=0.0
  )
  assert not adaptive_clip.any()
  assert (adaptive_frame < 2).float().mean() > 0.97
  _, capped_frame = adaptive_library.sample_progress(
    4096,
    0.1,
    1.0,
    difficulty,
    uniform_probability=0.0,
    maximum_probability_ratio=1.0,
  )
  assert abs((capped_frame < 2).float().mean().item() - 0.5) < 0.05
  torch.testing.assert_close(
    adaptive_library.temporal_bin(torch.tensor((0, 0)), torch.tensor((1, 2))),
    torch.tensor((0, 1)),
  )
  torch.testing.assert_close(
    adaptive_library.bin_source_start_frame, torch.tensor((100, 101))
  )


def test_reference_library_uses_separate_physical_reset_bank(tmp_path: Path) -> None:
  clips_dir = tmp_path / "clips"
  clips_dir.mkdir()
  root_position = np.zeros((4, 3), dtype=np.float32)
  root_position[:, 2] = (0.1, 0.2, 0.3, 0.4)
  quaternion = np.zeros((4, 4), dtype=np.float32)
  quaternion[:, 3] = 1.0
  np.savez(
    clips_dir / "clip.npz",
    root_position=root_position,
    root_quaternion_xyzw=quaternion,
    root_linear_velocity=np.zeros((4, 3), dtype=np.float32),
    root_angular_velocity=np.zeros((4, 3), dtype=np.float32),
    joint_position=np.zeros((4, 29), dtype=np.float32),
    joint_velocity=np.zeros((4, 29), dtype=np.float32),
  )
  manifest = {
    "clips": [
      {
        "clip_id": "001_test",
        "split": "train",
        "file": "clips/clip.npz",
        "target_fps": 50.0,
        "source_fps": 25.0,
        "source_support_start_frame": 100,
        "source_recording_id": "test.csv",
      }
    ]
  }
  (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
  np.savez(
    tmp_path / "physical_init.npz",
    schema_version=np.asarray(G1_PHYSICAL_INIT_SCHEMA_VERSION),
    clip_id=np.asarray(("001_test",)),
    frame=np.asarray((2,), dtype=np.int32),
    root_position=np.asarray(((0.01, -0.02, 0.31),), dtype=np.float32),
    root_quaternion_xyzw=np.asarray(((0.0, 0.0, 0.0, 1.0),), dtype=np.float32),
    root_linear_velocity=np.zeros((1, 3), dtype=np.float32),
    root_angular_velocity=np.zeros((1, 3), dtype=np.float32),
    joint_position=np.full((1, 29), 0.1, dtype=np.float32),
    joint_velocity=np.zeros((1, 29), dtype=np.float32),
  )

  library = G1ReferenceLibrary(
    tmp_path,
    "cpu",
    physical_init_file=tmp_path / "physical_init.npz",
    require_physical_init=True,
  )
  clip_id, frame = library.sample_progress(32, 0.3, 0.5)
  assert not clip_id.any()
  assert torch.all(frame == 2)
  torch.testing.assert_close(library.root_position[0, 2, 2], torch.tensor(0.3))
  torch.testing.assert_close(
    library.reset_root_position[0, 2], torch.tensor((0.01, -0.02, 0.31))
  )
  torch.testing.assert_close(library.reset_joint_position[0, 2], torch.full((29,), 0.1))


def test_g1_assistance_curriculum_uses_growing_success_windows():
  cfg = unitree_g1_flat_recovery_env_cfg()
  play_cfg = unitree_g1_flat_recovery_env_cfg(play=True)
  event_params = cfg.events["g1_recovery_reset"].params
  posture_windows = event_params["posture_success_windows"]
  assist_windows = event_params["assist_success_windows"]

  assert event_params["force_ranges"][0] == (160.0, 200.0)
  assert event_params["force_ranges"][-1] == (0.0, 0.0)
  force_ranges = event_params["force_ranges"]
  assert all(
    current[0] == following[1]
    for current, following in zip(force_ranges[:-1], force_ranges[1:], strict=True)
  )
  assert posture_windows == (
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
  assert assist_windows == (6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000)
  windows = posture_windows + assist_windows
  assert all(
    later > earlier for earlier, later in zip(windows, windows[1:], strict=False)
  )
  curriculum = cfg.curriculum["g1_recovery_assist"]
  assert curriculum.func is g1_recovery_assist_curriculum
  assert curriculum.params["success_threshold"] == 0.9
  assert "minimum_smp_weight" not in curriculum.params
  assert "maximum_smp_weight" not in curriculum.params
  assert "g1_recovery_assist" not in play_cfg.curriculum
  play_params = play_cfg.events["g1_recovery_reset"].params
  assert play_params["initial_posture_level"] == 12
  assert play_params["initial_assist_level"] == 8
  assert event_params["posture_mode_probabilities"][:9] == ((1.0, 0.0, 0.0),) * 9
  assert event_params["posture_mode_probabilities"][9] == (0.85, 0.15, 0.0)
  assert event_params["posture_mode_probabilities"][-1] == (0.45, 0.45, 0.10)
  assert event_params["posture_reference_min_progress"] == (
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
  assert event_params["posture_fallen_min_progress"] == (
    (0.15,) * 10 + (0.10, 0.05, 0.00)
  )
  assert event_params["curriculum_reference_num_envs"] == 1024
  assert event_params["curriculum_minimum_level_steps"] == 24
  assert event_params["fallen_max_progress"] == 0.25
  assert event_params["reference_frontier_probability"] == 0.5
  assert event_params["adaptive_bin_duration_s"] == 0.2
  assert event_params["adaptive_ema_rate"] == 0.01
  assert event_params["adaptive_uniform_probability"] == 1.0
  assert event_params["adaptive_max_probability_ratio"] == 1.0
  assert event_params["curriculum_probe_probability"] == 1.0
  assert event_params["hard_bin_report_count"] == 5
  assert event_params["physical_init_file"].endswith("physical_init.npz")
  assert event_params["require_physical_init"]
  assert not play_cfg.events["g1_recovery_reset"].params["require_physical_init"]
  assert cfg.scene.num_envs == 1024
  assert play_cfg.scene.num_envs == 16

  smp = cfg.rewards["smp"]
  assert smp.func is G1SmpReward
  assert smp.weight == 1.0
  assert smp.params["esm_timesteps"] == (22, 15, 8)
  assert smp.params["reward_weight"] == 10.0
  assert smp.params["terminal_reward_weight"] == 2.5
  assert smp.params["handoff_progress"] == (0.65, 0.85)
  assert cfg.rewards["pose"].weight == 2.0
  assert cfg.rewards["pose"].params["gate_minimum"] == 0.0
  assert cfg.rewards["pose"].params["gate_low"] == 0.65
  assert cfg.rewards["pose"].params["gate_high"] == 0.85
  assert "recovery_reference_tracking" not in cfg.rewards
  assert "recovery_reference" not in cfg.observations["actor"].terms
  assert "recovery_reference" not in cfg.observations["critic"].terms
  assert "recovery_context" not in cfg.observations["actor"].terms
  assert "recovery_context" not in cfg.observations["critic"].terms
  assert "command" not in cfg.observations["actor"].terms
  assert "command" not in cfg.observations["critic"].terms
  assert cfg.observations["actor"].history_length == 10
  assert not cfg.observations["actor"].flatten_history_dim
  assert cfg.rewards["recovery_progress"].weight == 2.0
  assert cfg.rewards["recovery_lift"].weight == 1.0
  assert cfg.rewards["recovery_hold"].weight == 2.0
  assert cfg.rewards["recovery_success"].weight == 10.0
  assert cfg.rewards["recovery_failure"].weight == -1.0
  assert cfg.rewards["self_collisions"].weight == -0.2
  assert cfg.rewards["dof_pos_limits"].weight == -0.5
  assert not cfg.terminations["recovery_success"].time_out
  assert "recovery_assistance" not in cfg.rewards
  assert "recovery_assistance_n" in cfg.metrics
  for mode in ("reference", "fallen", "stand"):
    assert cfg.metrics[f"recovery_mode_{mode}"].reduce == "last"
    assert cfg.metrics[f"recovery_success_{mode}"].reduce == "last"
  for label in (
    "070_085",
    "055_070",
    "040_055",
    "035_040",
    "030_035",
    "025_030",
    "020_025",
    "015_020",
    "010_015",
    "000_010",
  ):
    assert cfg.metrics[f"recovery_bin_{label}"].reduce == "last"
    assert cfg.metrics[f"recovery_success_bin_{label}"].reduce == "last"
  assert "smp" not in play_cfg.rewards


def test_g1_two_stage_curriculum_uses_total_nonstand_success():
  state = G1RecoveryReset.__new__(G1RecoveryReset)
  state._env = cast(Any, SimpleNamespace(device="cpu", common_step_counter=0))
  state._force_ranges = torch.tensor(((0.0, 200.0), (0.0, 100.0), (0.0, 0.0)))
  state._posture_mode_probabilities = (
    (1.0, 0.0, 0.0),
    (0.75, 0.25, 0.0),
    (0.45, 0.45, 0.1),
  )
  state._posture_reference_min_progress = (0.7, 0.4, 0.1)
  state._posture_fallen_min_progress = (0.15, 0.1, 0.0)
  state._posture_success_windows = (2, 4)
  state._assist_success_windows = (6, 8)
  state._curriculum_window_scale = 1.0
  state._minimum_level_steps = 0
  state._level_enter_step = 0
  state.posture_level = 0
  state.assist_level = 0
  state.attempts = torch.zeros((), dtype=torch.long)
  state.successes = torch.zeros_like(state.attempts)
  state.last_window_attempts = torch.zeros_like(state.attempts)
  state.last_success_rate = torch.zeros(())
  state.training_attempts = torch.zeros_like(state.attempts)
  state.training_successes = torch.zeros_like(state.attempts)
  state.last_training_success_rate = torch.zeros(())
  state.excluded_stale_attempts = torch.zeros_like(state.attempts)
  state.last_excluded_stale_attempts = torch.zeros_like(state.attempts)
  state.reset_temporal_bin = torch.tensor((0, 0, 1, 1, -1))
  state.bin_failure_ema = torch.ones(2)
  state.bin_attempts = torch.zeros(2, dtype=torch.long)
  state.bin_successes = torch.zeros(2, dtype=torch.long)
  state._adaptive_ema_rate = 0.01
  state._adaptive_total_attempts = 0
  state._adaptive_attempts_since_report = 0
  state._hard_bin_report_interval = 1_000_000
  state.mode = torch.tensor(
    (FALLEN_MODE, FALLEN_MODE, REFERENCE_MODE, FALLEN_MODE, STAND_MODE)
  )
  state.curriculum_probe = torch.tensor((True, True, True, True, False))
  state.reset_posture_level = torch.zeros(5, dtype=torch.long)
  state.reset_assist_level = torch.zeros(5, dtype=torch.long)
  state.succeeded = torch.tensor((True, True, True, False, True))
  state.episode_started = torch.zeros(5, dtype=torch.bool)

  state.record_outcomes(torch.tensor((0, 1, 2, 3, 4)))
  assert state.attempts.item() == 0
  state.episode_started.fill_(True)

  state.record_outcomes(torch.tensor((0, 1, 2)))
  torch.testing.assert_close(state.bin_attempts, torch.tensor((2, 1)))
  torch.testing.assert_close(state.bin_successes, torch.tensor((2, 1)))
  torch.testing.assert_close(state.bin_failure_ema, torch.tensor((0.9801, 0.99)))
  assert state.update_curriculum(success_threshold=0.9)
  assert state.posture_level == 1
  assert state.assist_level == 0
  assert state.level == 1
  assert state.required_window == 4
  assert state.attempts.item() == 0

  # Outcomes from episodes reset at level 0 must not certify level 1.
  state.record_outcomes(torch.tensor((0, 1, 2, 3, 4)))
  assert state.attempts.item() == 0
  assert state.excluded_stale_attempts.item() == 4
  state.reset_posture_level.fill_(1)
  assert not state.update_curriculum(success_threshold=0.9)
  state.record_outcomes(torch.tensor((0, 1, 2, 3, 4)))
  assert not state.update_curriculum(success_threshold=0.9)
  assert state.level == 1
  assert state.attempts.item() == 0
  assert state.last_window_attempts.item() == 4
  torch.testing.assert_close(state.last_success_rate, torch.tensor(0.75))
  assert state.last_excluded_stale_attempts.item() == 4

  state.succeeded[:] = True
  state.record_outcomes(torch.tensor((0, 1, 3, 4)))
  assert not state.update_curriculum(success_threshold=0.9)
  assert state.attempts.item() == 3
  state.record_outcomes(torch.tensor((2,)))
  assert state.update_curriculum(success_threshold=0.9)
  assert state.level == 2
  assert state.posture_complete
  assert state.assist_level == 0
  assert state.required_window == 6

  state.reset_posture_level.fill_(2)
  state.record_outcomes(torch.tensor((0, 1, 2, 3)))
  state.record_outcomes(torch.tensor((0, 1)))
  assert state.update_curriculum(success_threshold=0.9)
  assert state.posture_level == 2
  assert state.assist_level == 1
  assert state.level == 3
  assert state.required_window == 8


def test_g1_curriculum_scales_evidence_and_enforces_rollout_cooldown():
  state = G1RecoveryReset.__new__(G1RecoveryReset)
  state._env = cast(Any, SimpleNamespace(device="cpu", common_step_counter=23))
  state._force_ranges = torch.tensor(((0.0, 100.0), (0.0, 0.0)))
  state._posture_mode_probabilities = ((1.0, 0.0, 0.0), (0.5, 0.5, 0.0))
  state._posture_reference_min_progress = (0.7, 0.1)
  state._posture_fallen_min_progress = (0.15, 0.0)
  state._posture_success_windows = (500,)
  state._assist_success_windows = (1000,)
  state._curriculum_window_scale = 4.0
  state._minimum_level_steps = 24
  state._level_enter_step = 0
  state.posture_level = 0
  state.assist_level = 0
  state.attempts = torch.tensor(2000, dtype=torch.long)
  state.successes = torch.tensor(2000, dtype=torch.long)
  state.last_window_attempts = torch.zeros((), dtype=torch.long)
  state.last_success_rate = torch.zeros(())
  state.training_attempts = torch.tensor(2000, dtype=torch.long)
  state.training_successes = torch.tensor(2000, dtype=torch.long)
  state.last_training_success_rate = torch.zeros(())
  state.excluded_stale_attempts = torch.zeros((), dtype=torch.long)
  state.last_excluded_stale_attempts = torch.zeros((), dtype=torch.long)

  assert state.base_required_window == 500
  assert state.required_window == 2000
  assert not state.update_curriculum(success_threshold=0.9)
  assert state.posture_level == 0
  assert state.attempts.item() == 2000

  state._env.common_step_counter = 24
  assert state.update_curriculum(success_threshold=0.9)
  assert state.posture_level == 1
  assert state._level_enter_step == 24


def test_smp_reward_is_calibrated_without_assistance_level_scaling():
  errors = torch.tensor(((2.0, 4.0), (3.0, 6.0), (4.0, 8.0)))
  calibration = torch.tensor((2.0, 3.0, 4.0))
  reward = normalized_esm_reward(errors, calibration, scale=1.0)
  torch.testing.assert_close(reward, torch.exp(torch.tensor((-1.0, -2.0))))


def test_smp_weight_hands_off_smoothly_to_terminal_pose():
  progress = torch.tensor((0.0, 0.65, 0.75, 0.85, 1.0))
  weight = progress_handoff_weight(progress, 10.0, 2.5, 0.65, 0.85)
  torch.testing.assert_close(weight, torch.tensor((10.0, 10.0, 6.25, 2.5, 2.5)))


def test_recovery_reward_scaling_and_gate_are_dt_invariant():
  state = G1RecoveryReset.__new__(G1RecoveryReset)
  state.previous_progress = torch.tensor((0.2, 0.4))
  state.current_progress = torch.tensor((0.3, 0.35))
  state.just_succeeded = torch.tensor((True, False))
  state.succeeded = torch.tensor((True, False))
  event_manager = SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(func=state))
  termination_manager = SimpleNamespace(get_term=lambda _: torch.tensor((False, True)))
  env = cast(
    Any,
    SimpleNamespace(
      step_dt=0.02,
      event_manager=event_manager,
      termination_manager=termination_manager,
    ),
  )

  torch.testing.assert_close(
    recovery_progress_reward(env, "reset", 10.0, 10.0),
    torch.tensor((5.0, -2.5)),
  )
  torch.testing.assert_close(
    recovery_success_bonus(env, "reset"), torch.tensor((50.0, 0.0))
  )
  torch.testing.assert_close(
    recovery_failure_penalty(env, "reset", "time_out"),
    torch.tensor((0.0, 50.0)),
  )
  torch.testing.assert_close(
    recovery_gate(torch.tensor((0.4, 0.7, 0.9)), 0.55, 0.85),
    torch.tensor((0.0, 0.5, 1.0)),
  )


def test_composite_progress_has_no_reset_spike_and_rewards_height_or_upright():
  class Scene(dict[str, Any]):
    env_origins: torch.Tensor

  asset = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor(((0.0, 0.0, 0.10),)),
      default_root_state=torch.tensor(((0.0, 0.0, 0.76),)),
      projected_gravity_b=torch.tensor(((0.0, 0.0, 0.0),)),
    )
  )
  scene = Scene(robot=asset)
  scene.env_origins = torch.zeros(1, 3)
  env = cast(
    Any,
    SimpleNamespace(num_envs=1, device="cpu", step_dt=0.02, scene=scene),
  )
  reward = RecoveryCompositeProgress(cast(Any, None), env)
  asset_cfg = SceneEntityCfg("robot")

  torch.testing.assert_close(reward(env, asset_cfg, 0.6, 0.4, 5.0, 5.0), torch.zeros(1))
  asset.data.root_link_pos_w[:, 2] = 0.20
  assert reward(env, asset_cfg, 0.6, 0.4, 5.0, 5.0).item() > 0.0
  asset.data.projected_gravity_b[:, 2] = -0.5
  assert reward(env, asset_cfg, 0.6, 0.4, 5.0, 5.0).item() > 0.0
  reward.reset(torch.tensor((0,)))
  torch.testing.assert_close(reward(env, asset_cfg, 0.6, 0.4, 5.0, 5.0), torch.zeros(1))


def test_milestone_lift_cannot_reward_reset_velocity_or_repeated_height():
  state = G1RecoveryReset.__new__(G1RecoveryReset)
  state.mode = torch.tensor((FALLEN_MODE, 2))
  state.current_progress = torch.tensor((0.30, 0.75))
  state.succeeded = torch.zeros(2, dtype=torch.bool)
  state.applied_force = torch.tensor((100.0, 0.0))
  env = cast(
    Any,
    SimpleNamespace(
      num_envs=2,
      device="cpu",
      step_dt=0.02,
      event_manager=SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(func=state)),
    ),
  )
  lift = RecoveryMilestoneLift(cast(Any, None), env)

  # The inherited reset state produces no reward, regardless of its velocity.
  torch.testing.assert_close(lift(env, "reset", 5.0), torch.zeros(2))
  state.current_progress = torch.tensor((0.35, 0.80))
  torch.testing.assert_close(lift(env, "reset", 5.0), torch.tensor((2.5, 0.0)))
  # Regression and returning to an already rewarded milestone both produce zero.
  state.current_progress[0] = 0.32
  torch.testing.assert_close(lift(env, "reset", 5.0), torch.zeros(2))
  state.current_progress[0] = 0.35
  torch.testing.assert_close(lift(env, "reset", 5.0), torch.zeros(2))

  hold = recovery_hold_reward(env, "reset", 0.65, 0.85)
  torch.testing.assert_close(hold, torch.tensor((0.0, 0.84375)))


def test_recovery_success_allows_the_current_curriculum_force():
  state = G1RecoveryReset.__new__(G1RecoveryReset)
  state.current_progress = torch.tensor((0.9,))
  state.applied_force = torch.tensor((200.0,))
  state.hold_count = torch.zeros(1, dtype=torch.long)
  state.succeeded = torch.zeros(1, dtype=torch.bool)
  state.just_succeeded = torch.zeros(1, dtype=torch.bool)
  env = cast(
    Any,
    SimpleNamespace(
      event_manager=SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(func=state))
    ),
  )

  for _ in range(24):
    assert not recovery_succeeded(env, "reset", 0.85, 25).item()
  assert recovery_succeeded(env, "reset", 0.85, 25).item()


def test_recovery_force_stays_full_for_reference_and_fallen_modes():
  class Scene:
    env_origins = torch.zeros(3, 3)

  class Asset:
    data = SimpleNamespace(
      root_link_pos_w=torch.tensor(
        ((0.0, 0.0, 0.70), (0.0, 0.0, 0.20), (0.0, 0.0, 0.76))
      ),
      default_root_state=torch.tensor(
        ((0.0, 0.0, 0.76), (0.0, 0.0, 0.76), (0.0, 0.0, 0.76))
      ),
      projected_gravity_b=torch.tensor(
        ((0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
      ),
    )

    def write_external_wrench_to_sim(self, *args: object, **kwargs: object) -> None:
      del args, kwargs

  state = G1RecoveryReset.__new__(G1RecoveryReset)
  state._env = cast(Any, SimpleNamespace(num_envs=3, device="cpu", scene=Scene()))
  state._asset = cast(Any, Asset())
  state._body_ids = (0,)
  state.mode = torch.tensor((REFERENCE_MODE, FALLEN_MODE, STAND_MODE))
  state.succeeded = torch.zeros(3, dtype=torch.bool)
  state.episode_started = torch.zeros(3, dtype=torch.bool)
  state.sampled_force = torch.tensor((180.0, 170.0, 0.0))
  state.applied_force = torch.zeros(3)
  state.previous_progress = torch.zeros(3)
  state.current_progress = torch.zeros(3)

  state.step(0.02)

  torch.testing.assert_close(state.applied_force, torch.tensor((180.0, 170.0, 0.0)))


def test_recovery_progress_bin_metrics_use_reset_state_and_exclude_stand():
  state = G1RecoveryReset.__new__(G1RecoveryReset)
  state.mode = torch.tensor((REFERENCE_MODE, FALLEN_MODE, FALLEN_MODE, STAND_MODE))
  state.reset_progress = torch.tensor((0.27, 0.29, 0.31, 0.27))
  state.succeeded = torch.tensor((True, False, True, True))
  env = cast(
    Any,
    SimpleNamespace(
      event_manager=SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(func=state))
    ),
  )

  denominator = recovery_progress_bin_metric(env, "reset", 0.25, 0.30)
  numerator = recovery_progress_bin_success_metric(env, "reset", 0.25, 0.30)
  torch.testing.assert_close(denominator, torch.tensor((1.0, 1.0, 0.0, 0.0)))
  torch.testing.assert_close(numerator, torch.tensor((1.0, 0.0, 0.0, 0.0)))


def test_reset_distribution_shifts_from_easy_reference_to_stratified():
  state = G1RecoveryReset.__new__(G1RecoveryReset)
  state._env = cast(Any, SimpleNamespace(device="cpu"))
  state._force_ranges = torch.tensor(((160.0, 200.0), (120.0, 160.0), (0.0, 0.0)))
  state._posture_mode_probabilities = (
    (1.00, 0.00, 0.00),
    (1.00, 0.00, 0.00),
    (1.00, 0.00, 0.00),
    (1.00, 0.00, 0.00),
    (0.75, 0.25, 0.00),
    (0.60, 0.35, 0.05),
    (0.45, 0.45, 0.10),
  )
  state._posture_reference_min_progress = (
    0.70,
    0.55,
    0.40,
    0.25,
    0.10,
    0.10,
    0.10,
  )
  state._reference_max_progress = 0.85
  state.assist_level = 0

  state.posture_level = 0
  torch.testing.assert_close(state.mode_probabilities, torch.tensor((1.0, 0.0, 0.0)))
  assert state.reference_min_progress == 0.70
  assert state.reference_frontier_max_progress is None
  assert state.force_range == (160.0, 200.0)

  state.posture_level = 3
  torch.testing.assert_close(state.mode_probabilities, torch.tensor((1.0, 0.0, 0.0)))
  assert np.isclose(state.reference_min_progress, 0.25)
  assert np.isclose(state.reference_frontier_max_progress, 0.40)
  assert state.assist_level == 0

  state.posture_level = 4
  torch.testing.assert_close(state.mode_probabilities, torch.tensor((0.75, 0.25, 0.0)))
  assert np.isclose(state.reference_min_progress, 0.10)
  assert np.isclose(state.reference_frontier_max_progress, 0.25)

  state.posture_level = 6
  torch.testing.assert_close(state.mode_probabilities, torch.tensor((0.45, 0.45, 0.10)))
  assert np.isclose(state.reference_min_progress, 0.10)
  assert state.reference_frontier_max_progress is None
  assert state.posture_complete

  state.assist_level = 1
  assert state.force_range == (120.0, 160.0)
  torch.testing.assert_close(state.mode_probabilities, torch.tensor((0.45, 0.45, 0.10)))


def test_g1_recovery_uses_conditional_flow_fpo():
  cfg = unitree_g1_recovery_fpo_runner_cfg()

  assert cfg.actor.class_name == "mjlab.rl.flow_policy:ConditionalFlowPolicy"
  assert cfg.actor.distribution_cfg is None
  assert not cfg.actor.obs_normalization
  assert isinstance(cfg.algorithm, RslRlFpoAlgorithmCfg)
  assert cfg.algorithm.class_name == "mjlab.rl.fpo:FlowPolicyOptimization"
  assert cfg.algorithm.clip_param == 0.05
  assert cfg.algorithm.entropy_coef == 0.0
  assert cfg.actor.model_kwargs["num_flow_steps"] == 64
  assert cfg.actor.model_kwargs["base_noise_std"] == 0.5
  assert cfg.actor.model_kwargs["max_flow_velocity"] == 1.5
  assert cfg.actor.model_kwargs["cfm_loss_reduction"] == "sqrt"
  assert cfg.actor.model_kwargs["action_perturb_std"] == 0.02
  assert cfg.algorithm.num_mc_samples == 32
  assert cfg.algorithm.ratio_log_clip == 3.0
  assert cfg.algorithm.cfm_loss_clamp == 3.0
  assert cfg.algorithm.advantage_clamp == (5.0, 5.0)
  assert cfg.algorithm.optimizer == "adamw"
  assert not cfg.algorithm.use_clipped_value_loss
  assert cfg.algorithm.num_learning_epochs == 8
  assert cfg.algorithm.num_mini_batches == 8
  assert cfg.algorithm.learning_rate == 1.0e-4
  assert cfg.algorithm.weight_decay == 1.0e-4
  assert cfg.algorithm.diversity_loss_coef == 0.0
  assert cfg.algorithm.diversity_target_std == 0.12
  assert cfg.algorithm.diversity_batch_size == 64
