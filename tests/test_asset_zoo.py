import mujoco
import pytest

from mjlab.asset_zoo.robots import get_g1_robot_cfg, get_qlmini2_robot_cfg
from mjlab.entity import Entity


@pytest.mark.parametrize(
  "robot_name,robot_cfg_fn",
  [
    ("G1", get_g1_robot_cfg),
    ("qlmini2", get_qlmini2_robot_cfg),
  ],
)
def test_robot_compiles_parametrized(robot_name: str, robot_cfg_fn) -> None:
  """Tests that all robots in the asset zoo compile without errors."""
  robot_cfg = robot_cfg_fn()
  assert isinstance(Entity(robot_cfg).compile(), mujoco.MjModel)


def test_qlmini2_mass_matches_source_urdf() -> None:
  """The compiled qlmini2 model should not add mass for collision-only hands."""
  model = Entity(get_qlmini2_robot_cfg()).compile()

  assert model.body_mass.sum() == pytest.approx(14.2412)
  for body_name in ("right_hand", "left_hand"):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    assert model.body_mass[body_id] == 0.0
