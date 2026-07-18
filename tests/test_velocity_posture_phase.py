"""Tests for morphology-independent velocity posture phase estimation."""

from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.tasks.velocity.mdp.posture_phase import PosturePhaseEstimator


def _make_env(
  nominal_heights: torch.Tensor,
  height_ratios: torch.Tensor,
  projected_gravity: torch.Tensor,
  vertical_speed_ratio: float = 0.0,
) -> Any:
  gravity = 9.81
  root_height = nominal_heights * height_ratios
  lin_vel = torch.zeros(len(root_height), 3)
  lin_vel[:, 2] = vertical_speed_ratio * torch.sqrt(gravity * nominal_heights)
  asset = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.stack(
        (torch.zeros_like(root_height), torch.zeros_like(root_height), root_height),
        dim=1,
      ),
      default_root_state=torch.stack(
        (
          torch.zeros_like(nominal_heights),
          torch.zeros_like(nominal_heights),
          nominal_heights,
          torch.ones_like(nominal_heights),
          torch.zeros_like(nominal_heights),
          torch.zeros_like(nominal_heights),
          torch.zeros_like(nominal_heights),
        ),
        dim=1,
      ),
      projected_gravity_b=projected_gravity,
      root_link_lin_vel_w=lin_vel,
      root_link_ang_vel_b=torch.zeros(len(root_height), 3),
      gravity_vec_w=torch.tensor((0.0, 0.0, -gravity)),
    )
  )
  return cast(Any, SimpleNamespace(scene={"robot": asset}))


def test_posture_phase_normalizes_different_robot_heights() -> None:
  env = _make_env(
    nominal_heights=torch.tensor((0.41, 0.76)),
    height_ratios=torch.tensor((0.5, 0.5)),
    projected_gravity=torch.tensor(((0.0, 0.0, -1.0), (0.0, 0.0, -1.0))),
  )

  state = PosturePhaseEstimator().estimate(env)

  torch.testing.assert_close(state.height, torch.full((2,), 0.5))
  torch.testing.assert_close(state.progress, torch.full((2,), 0.5))
  torch.testing.assert_close(state.motion_stability, torch.ones(2))


def test_posture_phase_is_zero_at_roll_or_pitch_90_degrees() -> None:
  env = _make_env(
    nominal_heights=torch.full((3,), 0.41),
    height_ratios=torch.ones(3),
    projected_gravity=torch.tensor(
      ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    ),
  )

  state = PosturePhaseEstimator().estimate(env)

  torch.testing.assert_close(state.uprightness, torch.zeros(3))
  torch.testing.assert_close(state.progress, torch.zeros(3))
  torch.testing.assert_close(state.walk_gate, torch.zeros(3))
  torch.testing.assert_close(state.fallen, torch.ones(3))


def test_posture_phase_memberships_form_partition_of_unity() -> None:
  env = _make_env(
    nominal_heights=torch.full((5,), 0.41),
    height_ratios=torch.linspace(0.0, 1.0, 5),
    projected_gravity=torch.tensor(((0.0, 0.0, -1.0),)).repeat(5, 1),
    vertical_speed_ratio=0.5,
  )

  state = PosturePhaseEstimator().estimate(env)
  membership_sum = (
    state.fallen + state.recovering + state.upright_unstable + state.ready
  )

  torch.testing.assert_close(membership_sum, torch.ones(5))
  assert torch.all(state.fallen >= 0.0)
  assert torch.all(state.recovering >= 0.0)
  assert torch.all(state.upright_unstable >= 0.0)
  assert torch.all(state.ready >= 0.0)


def test_posture_walk_gate_uses_dimensionless_confidence_boundaries() -> None:
  env = _make_env(
    nominal_heights=torch.full((3,), 0.76),
    height_ratios=torch.tensor((0.60, 0.725, 0.85)),
    projected_gravity=torch.tensor(((0.0, 0.0, -1.0),)).repeat(3, 1),
  )

  state = PosturePhaseEstimator().estimate(env)

  torch.testing.assert_close(state.walk_gate, torch.tensor((0.0, 0.5, 1.0)))
  assert torch.equal(state.needs_recovery, torch.tensor((False, False, False)))
  assert torch.equal(state.is_ready, torch.tensor((False, False, True)))
