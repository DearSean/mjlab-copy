"""RL Boy constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

RL_BOY_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "RL_BOY" / "rlboy2.xml"
)
assert RL_BOY_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(RL_BOY_XML))


##
# Actuator config.
##

# 上半身手臂 PD 参数 (30kp, 2kd)
STIFFNESS_ARM = 30.0
DAMPING_ARM = 2.0
EFFORT_LIMIT_ARM = 50  # TODO: 根据实际电机型号确定5050

# 下半身腿部和腰部 PD 参数 (60kp, 3kd)
STIFFNESS_LEG_WAIST = 60.0
DAMPING_LEG_WAIST = 3.0
EFFORT_LIMIT_LEG = 60  # TODO: 根据实际电机型号确定
EFFORT_LIMIT_WAIST = 60 # TODO: 根据实际电机型号确定

# 上半身手臂执行器配置 (使用 BuiltinPositionActuatorCfg 直接使用 XML motor 并设置 kp/kd)
# 关节名称: left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow_pitch
#           right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow_pitch
RL_BOY_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_pitch_joint",
  ),
  stiffness=STIFFNESS_ARM,
  damping=DAMPING_ARM,
  effort_limit=EFFORT_LIMIT_ARM,
)

# 下半身腿部执行器配置
# 关节名称: left_hip_yaw_joint, left_hip_roll_joint, left_hip_pitch_joint
#           left_knee_pitch_joint, left_ankle_pitch_joint
#           right_hip_yaw_joint, right_hip_roll_joint, right_hip_pitch_joint
#           right_knee_pitch_joint, right_ankle_pitch_joint
RL_BOY_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_pitch_joint",
    ".*_ankle_pitch_joint",
  ),
  stiffness=STIFFNESS_LEG_WAIST,
  damping=DAMPING_LEG_WAIST,
  effort_limit=EFFORT_LIMIT_LEG,
)

# 腰部执行器配置
RL_BOY_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_yaw_joint",
  ),
  stiffness=STIFFNESS_LEG_WAIST,
  damping=DAMPING_LEG_WAIST,
  effort_limit=EFFORT_LIMIT_WAIST,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.45),  # 根据 XML 中 base_link 的初始高度
  joint_pos={
    # 腿部关节初始位置
    "left_hip_pitch_joint": -0.1,
    "left_knee_pitch_joint": 0.3,
    "left_ankle_pitch_joint": 0.18,
    # 右腿
    "right_hip_pitch_joint": 0.1,
    "right_knee_pitch_joint": -0.3,
    "right_ankle_pitch_joint": -0.18,
    # 其余所有腿部关节默认0（hip_yaw/hip_roll）
    ".*_hip_yaw_joint": 0,
    ".*_hip_roll_joint": 0, 
    # 手臂关节初始位置
    ".*_shoulder_pitch_joint": 0,  # TODO: 根据实际机器人调整
    ".*_elbow_pitch_joint": 0,  # TODO: 根据实际机器人调整
    "left_shoulder_roll_joint": 1.3,  # TODO: 根据实际机器人调整
    "right_shoulder_roll_joint": -1.3,  # TODO: 根据实际机器人调整
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# 碰撞配置需要根据实际 geom 名称确定
# 从 XML 中可以看到碰撞相关的 geom，需要添加 _collision 后缀的 geom
# 基于 RL_BOY.xml 的碰撞配置
FULL_COLLISION = CollisionCfg(
    # 模型中碰撞 geom 无 _collision 后缀，直接匹配所有 mesh 类型的 geom（排除视觉辅助 geom）
    geom_names_expr=(
        r"^left_hip.*", r"^right_hip.*", r"^left_knee.*", r"^right_knee.*",
        r"^left_ankle.*", r"^right_ankle.*", r"^base_link$", r"^waist_yaw_link$",
        r"^left_shoulder.*", r"^right_shoulder.*", r"^left_elbow.*", r"^right_elbow.*",
        r"^head_yaw_link$"
    ),
    # 碰撞维度：脚踝（脚部）设为 3，其他碰撞体设为 1
    condim={
        r".*_ankle_pitch_link": 3,  # 匹配左右脚踝 geom
        r".*_collision": 1,         # 兼容通用碰撞体（兜底）
        r".*_hip.*|.*_knee.*|.*_shoulder.*|.*_elbow.*|.*_waist.*|.*_base.*": 1
    },
    # 优先级：脚部（脚踝）设为最高优先级 1
    priority={
        r".*_ankle_pitch_link": 1
    },
    # 摩擦系数：脚踝（脚部）设为 0.6，匹配模型中默认摩擦系数（1.0 0.3 0.3）的主摩擦系数
    friction={
        r".*_ankle_pitch_link": (0.6,),  # 脚部主摩擦系数
        r".*_hip.*|.*_knee.*": (1.0,)    # 腿部其他部位沿用模型默认摩擦系数
    }
)

##
# Final config.
##

RL_BOY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    RL_BOY_ACTUATOR_ARM,
    RL_BOY_ACTUATOR_LEG,
    RL_BOY_ACTUATOR_WAIST,
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


# 动作缩放表
# 计算方式: scale = 0.25 * effort_limit / stiffness
RL_BOY_ACTION_SCALE: dict[str, float] = {}
for a in RL_BOY_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    RL_BOY_ACTION_SCALE[n] = 0.25 * e / s

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_rlboy_robot_cfg())

  viewer.launch(robot.spec.compile())

