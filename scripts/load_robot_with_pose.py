"""使用 MuJoCo 加载机器人 XML 并可自定义位置和姿态的脚本。"""

import argparse
import json
import math
import random

import mujoco
import mujoco.viewer as viewer


# 四种躺倒姿态的基四元数
BASE_QUATS = {
    "prone": (0.707, 0.707, 0, 0),      # 俯卧（面朝下）
    "supine": (0.707, -0.707, 0, 0),    # 仰卧（面朝上）
    "left": (0.707, 0, 0.707, 0),       # 左侧卧
    "right": (0.707, 0, -0.707, 0),    # 右侧卧
}


def quat_multiply(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]):
    """四元数乘法：q = q1 * q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_from_z_rotation(angle: float) -> tuple[float, float, float, float]:
    """绕 Z 轴旋转给定角度的四元数"""
    half = angle / 2
    return (math.cos(half), 0, 0, math.sin(half))


def load_robot_with_pose(
  xml_path: str,
  pos: tuple[float, float, float] = (0, 0, 0),
  quat: tuple[float, float, float, float] = (1, 0, 0, 0),
  joint_pos: dict[str, float] | None = None,
  random_z_deg: float = 0,
):
  """加载机器人 XML 并设置初始位置、姿态和关节角。

  Args:
      xml_path: 机器人 XML 文件路径
      pos: 机器人 freejoint 位置 (x, y, z)
      quat: 机器人 freejoint 四元数姿态 (w, x, y, z)，默认为单位四元数（无旋转）
      joint_pos: 关节名称到关节角度的映射，覆盖 XML 默认/关键帧值
      random_z_deg: 额外绕全局 Z 轴随机旋转的角度范围（度），设为 0 则不旋转
  """
  # 加载模型
  model = mujoco.MjModel.from_xml_path(xml_path)
  data = mujoco.MjData(model)

  # 重置数据到默认状态（含 XML 关键帧，如果有的话）
  mujoco.mj_resetData(model, data)

  # 应用绕 Z 轴的随机旋转
  final_quat = quat
  if random_z_deg > 0:
      z_rotation = random.uniform(-random_z_deg, random_z_deg)
      z_quat = quat_from_z_rotation(math.radians(z_rotation))
      final_quat = quat_multiply(quat, z_quat)

  # MuJoCo freejoint 的 qpos 布局为 [pos(3), quat(4)] = [x, y, z, w, x, y, z]
  data.qpos[0:3] = pos   # 位置
  data.qpos[3:7] = final_quat  # 四元数

  # 应用关节角度覆盖
  if joint_pos:
    for joint_name, value in joint_pos.items():
      joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
      if joint_id < 0:
        raise ValueError(f"找不到关节 '{joint_name}'，请检查 XML 中的关节名称")

      qpos_adr = model.jnt_qposadr[joint_id]
      joint_dof = model.jnt_dofadr[joint_id]
      # 对于滑动/旋转关节，dof 数等于 1
      if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(
          f"关节 '{joint_name}' 是 freejoint，请用 --pos/--quat 设置根部位姿"
        )

      # 检查关节范围并 clamp，避免非法初始角度导致仿真发散
      joint_range = model.jnt_range[joint_id]
      if not (joint_range[0] == 0.0 and joint_range[1] == 0.0):
        value = max(joint_range[0], min(joint_range[1], value))

      data.qpos[qpos_adr] = value
      # 同时清零对应关节速度，保证从静止开始
      if joint_dof >= 0:
        data.qvel[joint_dof] = 0.0

  # 前向运动学，确保所有体坐标、碰撞等状态与 qpos 一致
  mujoco.mj_forward(model, data)

  return model, data


def _parse_joint_pos(value: str | None) -> dict[str, float] | None:
  """解析 --joint-pos 参数，支持 JSON 对象字符串。"""
  if not value:
    return None
  parsed = json.loads(value)
  if not isinstance(parsed, dict):
    raise argparse.ArgumentTypeError(
      "--joint-pos 必须是 JSON 对象，例如 '{\"joint_name\": 0.5}'"
    )
  return {str(k): float(v) for k, v in parsed.items()}


def main():
  parser = argparse.ArgumentParser(description="加载机器人并设置初始姿态")
  parser.add_argument(
    "--xml",
    type=str,
    default="src/mjlab/asset_zoo/robots/RL_BOY/RLBOY2sim.xml",
    help="机器人 XML 文件路径",
  )
  parser.add_argument(
    "--pos",
    type=float,
    nargs=3,
    default=[0, 0, 0.41],
    help="机器人 freejoint 位置 (x, y, z)",
  )
  parser.add_argument(
    "--quat",
    type=float,
    nargs=4,
    default=None,
    help="机器人 freejoint 四元数姿态 (w, x, y, z)，如果指定则忽略 --pose",
  )
  parser.add_argument(
    "--pose",
    type=str,
    choices=["prone", "supine", "left", "right"],
    default=None,
    help="预定义躺倒姿态：prone(俯卧), supine(仰卧), left(左侧卧), right(右侧卧)",
  )
  parser.add_argument(
    "--random-z",
    type=float,
    default=0,
    help="绕全局 Z 轴随机旋转角度范围（度），例如 360 表示随机旋转 0-360 度",
  )
  parser.add_argument(
    "--joint-pos",
    type=str,
    default=None,
    help='关节初始角度，JSON 对象格式，例如 \'{"left_hip_pitch_joint": -0.2, "left_knee_pitch_joint": 0.4}\'',
  )
  args = parser.parse_args()

  # 确定基础四元数
  if args.quat is not None:
    base_quat = tuple(args.quat)
  elif args.pose is not None:
    base_quat = BASE_QUATS[args.pose]
  else:
    base_quat = (1, 0, 0, 0)

  joint_pos = _parse_joint_pos(args.joint_pos)

  model, data = load_robot_with_pose(
    args.xml,
    pos=tuple(args.pos),
    quat=base_quat,
    joint_pos=joint_pos,
    random_z_deg=args.random_z,
  )

  print(f"加载 XML: {args.xml}")
  print(f"初始位置: {args.pos}")
  print(f"基础姿态: {base_quat}")
  if args.random_z > 0:
    print(f"绕 Z 轴随机旋转: ±{args.random_z}°")
  print(f"最终姿态 (quat w,x,y,z): ({data.qpos[3]:.3f}, {data.qpos[4]:.3f}, {data.qpos[5]:.3f}, {data.qpos[6]:.3f})")
  if joint_pos:
    print(f"关节覆盖: {joint_pos}")

  # 启动交互式 viewer
  viewer.launch(model, data)


if __name__ == "__main__":
  main()
