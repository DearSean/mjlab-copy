"""Tests for the standalone MuJoCo deployment script."""

import numpy as np

from mjlab.deploy.deploy_mujcoco import build_velocity_observation


def test_velocity_observation_matches_actor_term_order() -> None:
  """Deployment observations exclude base velocity and preserve term order."""
  num_actions = 2
  observation = build_velocity_observation(
    base_ang_vel=np.full(3, 1.0),
    gravity_orientation=np.full(3, 2.0),
    joint_pos=np.full(num_actions, 3.0),
    joint_vel=np.full(num_actions, 4.0),
    action=np.full(num_actions, 5.0),
    command=np.full(3, 6.0),
  )

  expected = np.array(
    [1.0] * 3
    + [2.0] * 3
    + [3.0] * num_actions
    + [4.0] * num_actions
    + [5.0] * num_actions
    + [6.0] * 3,
    dtype=np.float32,
  )
  np.testing.assert_array_equal(observation, expected)
