"""RL Boy constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##
RL_BOY_XML: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "RL_BOY" / "RLBOY.xml"
assert RL_BOY_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(RL_BOY_XML))


##
# Actuator config.
##

ROTOR_INERTIAS_J3507 = (
  8.70e-6,
  0.0,
  0.0,
)
# Two-stage planetary placeholder: the first element is the input stage (1:1).
# The built-in motor reducer is the second stage; the third stage is 1:1 here.
GEARS_J3507 = (
  1,
  7,
  1,
)
ARMATURE_J3507 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_J3507, GEARS_J3507
)

# 2. DM-J6006-2EC (中载，对标原 5020)
ROTOR_INERTIAS_J6006 = (
  5.80e-5,
  0.0,
  0.0,
)
GEARS_J6006 = (
  1,
  6,
  1,
)
ARMATURE_J6006 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_J6006, GEARS_J6006
)

# 3. DM-J8006-2EC V1.1 (重载，对标原 7520 系列)
ROTOR_INERTIAS_J8006 = (
  1.15e-4,
  0.0,
  0.0,
)
GEARS_J8006 = (
  1,
  6,
  1,
)
ARMATURE_J8006 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_J8006, GEARS_J8006
)

ACTUATOR_J3507 = ElectricActuator(
  reflected_inertia=ARMATURE_J3507,
  velocity_limit=40,
  effort_limit=3.0,
)

ACTUATOR_J6006 = ElectricActuator(
  reflected_inertia=ARMATURE_J6006,
  velocity_limit=23,
  effort_limit=11.0,
)

ACTUATOR_J8006 = ElectricActuator(
  reflected_inertia=ARMATURE_J8006,
  velocity_limit=20,
  effort_limit=20.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz 固有角频率,腿软时增大
DAMPING_RATIO_J3507 = 2.0
DAMPING_RATIO_J6006 = 2.0
DAMPING_RATIO_J8006 = 1.2

# 刚度计算: K = J_ref * ω_n²
STIFFNESS_J3507 = ARMATURE_J3507 * (NATURAL_FREQ**2)
STIFFNESS_J6006 = ARMATURE_J6006 * (NATURAL_FREQ**2)
STIFFNESS_J8006 = ARMATURE_J8006 * (NATURAL_FREQ**2)

# 阻尼计算: D = 2*ζ*J_ref*ω_n
DAMPING_J3507 = 2.0 * DAMPING_RATIO_J3507 * ARMATURE_J3507 * NATURAL_FREQ
DAMPING_J6006 = 2.0 * DAMPING_RATIO_J6006 * ARMATURE_J6006 * NATURAL_FREQ
DAMPING_J8006 = 2.0 * DAMPING_RATIO_J8006 * ARMATURE_J8006 * NATURAL_FREQ

RL_BOY_ACTUATOR_ARM = DcMotorActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_pitch_joint",
  ),
  stiffness=STIFFNESS_J3507,
  damping=DAMPING_J3507,
  effort_limit=ACTUATOR_J3507.effort_limit,
  saturation_effort=ACTUATOR_J3507.effort_limit,
  velocity_limit=ACTUATOR_J3507.velocity_limit,
  armature=ACTUATOR_J3507.reflected_inertia,
)
RL_BOY_ACTUATOR_LEG = DcMotorActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_pitch_joint",
  ),
  stiffness=STIFFNESS_J8006,
  damping=DAMPING_J8006,
  effort_limit=ACTUATOR_J8006.effort_limit,
  saturation_effort=ACTUATOR_J8006.effort_limit,
  velocity_limit=ACTUATOR_J8006.velocity_limit,
  armature=ACTUATOR_J8006.reflected_inertia,
)

# 腰部和脚部执行器配置
RL_BOY_ACTUATOR_WAIST_FOOT = DcMotorActuatorCfg(
  target_names_expr=(
    "waist_yaw_joint",
    "head_yaw_joint",
    ".*_ankle_pitch_joint",
  ),
  stiffness=STIFFNESS_J6006,
  damping=DAMPING_J6006,
  effort_limit=ACTUATOR_J6006.effort_limit,
  saturation_effort=ACTUATOR_J6006.effort_limit,
  velocity_limit=ACTUATOR_J6006.velocity_limit,
  armature=ACTUATOR_J6006.reflected_inertia,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.41),  # 根据 XML 中 base_link 的初始高度
  joint_pos={
    # 腿部关节初始位置
    "left_hip_pitch_joint": -0.2,
    "left_knee_pitch_joint": 0.4,
    "left_ankle_pitch_joint": -0.2,
    # 右腿
    "right_hip_pitch_joint": -0.2,
    "right_knee_pitch_joint": 0.4,
    "right_ankle_pitch_joint": -0.2,
    # 其余所有腿部关节默认0（hip_yaw/hip_roll）
    ".*_hip_yaw_joint": 0,
    ".*_hip_roll_joint": 0,
    # 手臂关节初始位置
    ".*_shoulder_pitch_joint": 0.15,
    ".*_elbow_pitch_joint": 0.9,
    "left_shoulder_roll_joint": 0.3,
    "right_shoulder_roll_joint": -0.3,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

_foot_regex = "^(left|right)_foot[1-7]_collision$"

# Enable frictional contact for every collision geom. Feet retain higher-dimensional
# contact and stronger sliding friction; the remaining links use standard tangential
# friction so hands, knees, and the torso can support recovery motions.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  # Harden all collision geoms.
  solref=(0.01, 1),
  condim={_foot_regex: 6, ".*_collision": 3},
  priority={_foot_regex: 1},
  friction={
    _foot_regex: (1.0, 5e-3, 5e-4),
    ".*_collision": (0.6, 5e-3, 5e-4),
  },
)

##
# Final config.
##

RL_BOY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    RL_BOY_ACTUATOR_ARM,
    RL_BOY_ACTUATOR_LEG,
    RL_BOY_ACTUATOR_WAIST_FOOT,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_rlboy_robot_cfg() -> EntityCfg:
  """Get a fresh RL Boy robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=RL_BOY_ARTICULATION,
  )


# 动作缩放表。每个关节组使用独立权重：腿部 0.7，手臂和腰/脚踝 0.5。
_ACTION_SCALE_GROUPS = (
  (RL_BOY_ACTUATOR_ARM, 0.5),
  (RL_BOY_ACTUATOR_LEG, 0.7),
  (RL_BOY_ACTUATOR_WAIST_FOOT, 0.5),
)
RL_BOY_ACTION_SCALE: dict[str, float] = {}
for a, weight in _ACTION_SCALE_GROUPS:
  assert isinstance(a, DcMotorActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    RL_BOY_ACTION_SCALE[n] = weight * e / s

# Lock the head: the policy still outputs an action dimension for it, but the
# target position is always the default position, so the head stays still.
RL_BOY_ACTION_SCALE["head_yaw_joint"] = 0.0

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_rlboy_robot_cfg())

  viewer.launch(robot.spec.compile())
