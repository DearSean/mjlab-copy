"""qlmini2 robot and actuator constants."""

import math
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

QLMINI2_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "qlmini2" / "urdf" / "qlmini2.xml"
)
assert QLMINI2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(QLMINI2_XML))


# Specs below are digitized from the supplied RobStride manuals dated
# 2025-12-10.  The curve points are the 48 V output-shaft T-N envelopes, not
# MIT protocol ranges.  Their continuous ratings remain separate because the
# manuals' overload tables show that peak torque is only sustainable briefly.
# The manuals do not specify rotor inertia. These conservative output-side
# armature estimates are provisionally scaled by motor class from RL Boy's
# documented reflected inertias; replace them with measured values when known.
def _rpm_to_rad_s(rpm: float) -> float:
  return rpm * 2.0 * math.pi / 60.0


EL05_RATED_TORQUE = 1.8
EL05_PEAK_TORQUE = 6.0
EL05_NO_LOAD_SPEED = _rpm_to_rad_s(430.0)
EL05_ARMATURE = 0.0005
EL05_TORQUE_SPEED_POINTS = tuple(
  (_rpm_to_rad_s(rpm), torque)
  for rpm, torque in (
    (0.0, 5.6),
    (100.0, 5.3),
    (220.0, 4.0),
    (310.0, 3.0),
    (430.0, 0.0),
  )
)

RS00_RATED_TORQUE = 5.0
RS00_PEAK_TORQUE = 14.0
RS00_NO_LOAD_SPEED = _rpm_to_rad_s(315.0)
RS00_ARMATURE = 0.002
RS00_TORQUE_SPEED_POINTS = tuple(
  (_rpm_to_rad_s(rpm), torque)
  for rpm, torque in (
    (0.0, 14.0),
    (50.0, 14.0),
    (130.0, 12.0),
    (170.0, 11.0),
    (220.0, 8.0),
    (260.0, 6.0),
    (300.0, 3.0),
    (315.0, 0.0),
  )
)

RS02_RATED_TORQUE = 6.0
RS02_PEAK_TORQUE = 17.0
RS02_NO_LOAD_SPEED = _rpm_to_rad_s(410.0)
RS02_ARMATURE = 0.003
RS02_TORQUE_SPEED_POINTS = tuple(
  (_rpm_to_rad_s(rpm), torque)
  for rpm, torque in (
    (0.0, 17.0),
    (165.0, 17.0),
    (195.0, 16.0),
    (240.0, 14.0),
    (285.0, 10.0),
    (330.0, 7.0),
    (375.0, 4.0),
    (410.0, 0.0),
  )
)

RS06_RATED_TORQUE = 11.0
RS06_PEAK_TORQUE = 36.0
RS06_NO_LOAD_SPEED = _rpm_to_rad_s(510.0)
RS06_ARMATURE = 0.005
RS06_TORQUE_SPEED_POINTS = tuple(
  (_rpm_to_rad_s(rpm), torque)
  for rpm, torque in (
    (0.0, 36.0),
    (270.0, 35.5),
    (290.0, 32.0),
    (310.0, 30.0),
    (330.0, 27.5),
    (350.0, 23.0),
    (370.0, 20.0),
    (390.0, 17.0),
    (410.0, 16.0),
    (430.0, 15.0),
    (450.0, 11.0),
    (470.0, 7.0),
    (490.0, 1.5),
    (510.0, 0.0),
  )
)

# Initial simulation-space PD gains. These are controller tuning values, not
# claimed motor-manual parameters, and should be identified on the real robot.
STIFFNESS_EL05 = 5.0
DAMPING_EL05 = 0.3
STIFFNESS_RS00 = 8.0
DAMPING_RS00 = 0.5
STIFFNESS_RS02 = 16.0
DAMPING_RS02 = 0.7
STIFFNESS_RS06 = 16.0
DAMPING_RS06 = 0.7

QLMINI2_ACTUATOR_RS06 = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=STIFFNESS_RS06,
  damping=DAMPING_RS06,
  effort_limit=RS06_RATED_TORQUE,
  saturation_effort=RS06_PEAK_TORQUE,
  velocity_limit=RS06_NO_LOAD_SPEED,
  torque_speed_points=RS06_TORQUE_SPEED_POINTS,
  armature=RS06_ARMATURE,
)

QLMINI2_ACTUATOR_RS00 = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint", ".*_shoulder_pitch_joint"),
  stiffness=STIFFNESS_RS00,
  damping=DAMPING_RS00,
  effort_limit=RS00_RATED_TORQUE,
  saturation_effort=RS00_PEAK_TORQUE,
  velocity_limit=RS00_NO_LOAD_SPEED,
  torque_speed_points=RS00_TORQUE_SPEED_POINTS,
  armature=RS00_ARMATURE,
)

QLMINI2_ACTUATOR_RS02 = DcMotorActuatorCfg(
  target_names_expr=(
    ".*_hip_roll_joint",
    ".*_knee_joint",
    ".*_foot_pitch_joint",
    "waist_yaw_joint",
  ),
  stiffness=STIFFNESS_RS02,
  damping=DAMPING_RS02,
  effort_limit=RS02_RATED_TORQUE,
  saturation_effort=RS02_PEAK_TORQUE,
  velocity_limit=RS02_NO_LOAD_SPEED,
  torque_speed_points=RS02_TORQUE_SPEED_POINTS,
  armature=RS02_ARMATURE,
)

QLMINI2_ACTUATOR_EL05 = DcMotorActuatorCfg(
  target_names_expr=(
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
  ),
  stiffness=STIFFNESS_EL05,
  damping=DAMPING_EL05,
  effort_limit=EL05_RATED_TORQUE,
  saturation_effort=EL05_PEAK_TORQUE,
  velocity_limit=EL05_NO_LOAD_SPEED,
  torque_speed_points=EL05_TORQUE_SPEED_POINTS,
  armature=EL05_ARMATURE,
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.48),
  joint_pos={
    ".*_hip_pitch_joint": -0.15,
    ".*_knee_joint": 0.3,
    ".*_foot_pitch_joint": -0.15,
    ".*_shoulder_pitch_joint": 0.15,
    ".*_elbow_joint": 0.5,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

_FOOT_COLLISION_EXPR = r"^(left|right)_foot[1-7]_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  condim={_FOOT_COLLISION_EXPR: 3, ".*_collision.*": 1},
  priority={_FOOT_COLLISION_EXPR: 1},
  friction={_FOOT_COLLISION_EXPR: (0.8,)},
)

QLMINI2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    QLMINI2_ACTUATOR_RS06,
    QLMINI2_ACTUATOR_RS00,
    QLMINI2_ACTUATOR_RS02,
    QLMINI2_ACTUATOR_EL05,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_qlmini2_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=QLMINI2_ARTICULATION,
  )


QLMINI2_ACTION_SCALE: dict[str, float] = {}
for actuator in QLMINI2_ARTICULATION.actuators:
  assert isinstance(actuator, DcMotorActuatorCfg)
  for name_expr in actuator.target_names_expr:
    QLMINI2_ACTION_SCALE[name_expr] = 0.25 * actuator.effort_limit / actuator.stiffness
