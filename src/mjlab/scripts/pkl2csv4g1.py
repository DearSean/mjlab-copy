import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

G1_DOF = 29

G1_CSV_DOF_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]

LEGGED_LAB_G1_DOF_NAMES = [
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
]


def _load_motion_data(pkl_path: Path):
  try:
    with open(pkl_path, "rb") as f:
      return pickle.load(f)
  except (ModuleNotFoundError, pickle.UnpicklingError):
    try:
      import joblib
    except ModuleNotFoundError as joblib_exc:
      raise RuntimeError(
        f"{pkl_path} 不是标准 pickle 文件，可能由 joblib 保存；"
        "请在当前环境安装 joblib 后重试。"
      ) from joblib_exc

    try:
      return joblib.load(pkl_path)
    except Exception as joblib_exc:
      raise RuntimeError(
        f"无法读取 {pkl_path}，标准 pickle 和 joblib.load 都失败。"
      ) from joblib_exc
  except Exception as exc:
    raise RuntimeError(f"无法读取 {pkl_path}: {exc}") from exc


def _required_array(motion_data: dict, *keys: str) -> np.ndarray:
  for key in keys:
    if key in motion_data:
      return np.array(motion_data[key])
  keys_text = " / ".join(keys)
  raise RuntimeError(f"PKL 缺少必要字段: {keys_text}")


def _source_dof_names(motion_data: dict, source_order: str) -> list[str]:
  if "dof_names" in motion_data:
    return list(motion_data["dof_names"])
  if source_order == "csv":
    return G1_CSV_DOF_NAMES
  return LEGGED_LAB_G1_DOF_NAMES


def _reorder_dof_pos(
  dof_pos: np.ndarray, motion_data: dict, source_order: str
) -> np.ndarray:
  source_dof_names = _source_dof_names(motion_data, source_order)
  missing_names = [name for name in G1_CSV_DOF_NAMES if name not in source_dof_names]
  if missing_names:
    raise RuntimeError(f"PKL dof_names 缺少 G1 关节: {missing_names}")

  reorder_indices = [source_dof_names.index(name) for name in G1_CSV_DOF_NAMES]
  if reorder_indices == list(range(G1_DOF)):
    return dof_pos
  return dof_pos[:, reorder_indices]


def pkl_to_csv_g1(pkl_path: Path, csv_path: Path, source_order: str = "lab"):
  """
  Unitree G1 29DoF 专用：PKL 运动数据 转 CSV
  仅保留必要字段：基座位置 + 基座四元数(qx,qy,qz,qw) + 29个关节角度
  输出格式兼容 csv_to_npz.py
  """
  # 1. 加载 pkl 文件
  motion_data = _load_motion_data(pkl_path)

  # 2. 校验数据结构（根据常规运动pkl格式适配）
  # 约定 pkl 内部字段（可根据你的实际pkl键名修改）：
  #   root_pos: 基座位置 [N, 3] (x,y,z)
  #   root_rot/root_quat: 基座四元数 [N, 4] (qx,qy,qz,qw)
  #   dof_pos: 关节角度 [N, 29]
  # 输出 CSV 的关节顺序与 csv_to_npz.py 中的 G1 joint_names 一致。
  root_pos = _required_array(motion_data, "root_pos")
  root_quat = _required_array(motion_data, "root_rot", "root_quat")
  dof_pos = _required_array(motion_data, "dof_pos")
  dof_pos = _reorder_dof_pos(dof_pos, motion_data, source_order)

  # 维度校验
  n_frames = root_pos.shape[0]
  assert root_pos.shape == (n_frames, 3), "基座位置维度必须为 [帧数, 3]"
  assert root_quat.shape == (n_frames, 4), "基座四元数维度必须为 [帧数, 4]"
  assert dof_pos.shape == (n_frames, G1_DOF), (
    f"关节角度必须为 [帧数, {G1_DOF}] (Unitree G1 29DoF)"
  )

  print(f"加载完成，总帧数: {n_frames}")
  print(f"开始写入 CSV: {csv_path}")

  # 3. 逐帧拼接数据并写入 CSV
  with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file, delimiter=",")
    for frame_idx in range(n_frames):
      # 拼接一行: pos(3) + quat(4) + dof(29) -> 共36列
      row = []
      # 基座位置 x,y,z
      row.extend(root_pos[frame_idx].tolist())
      # 基座四元数 qx, qy, qz, qw (和 csv_to_npz.py 输入要求一致)
      row.extend(root_quat[frame_idx].tolist())
      # 29个关节角度
      row.extend(dof_pos[frame_idx].tolist())

      writer.writerow(row)

  print(f"转换成功！文件已保存至: {csv_path}")


def convert_path(pkl_path: str, csv_path: str, source_order: str = "lab"):
  input_path = Path(pkl_path)
  output_path = Path(csv_path)

  if not input_path.exists():
    raise FileNotFoundError(f"--pkl 路径不存在: {input_path}")

  if input_path.is_file():
    if input_path.suffix.lower() != ".pkl":
      raise ValueError(f"--pkl 输入文件必须是 .pkl 文件: {input_path}")

    if output_path.exists() and not output_path.is_dir():
      raise ValueError("--csv 必须是输出文件夹路径，不能是文件")

    output_path.mkdir(parents=True, exist_ok=True)
    pkl_to_csv_g1(input_path, output_path / f"{input_path.stem}.csv", source_order)
    return

  if not input_path.is_dir():
    raise ValueError(f"--pkl 必须是 .pkl 文件或文件夹: {input_path}")

  pkl_files = sorted(input_path.glob("*.pkl"))
  if not pkl_files:
    raise RuntimeError(f"输入文件夹中没有找到 .pkl 文件: {input_path}")

  if output_path.exists() and not output_path.is_dir():
    raise ValueError("--pkl 为文件夹时，--csv 必须是输出文件夹路径")

  output_path.mkdir(parents=True, exist_ok=True)
  print(f"发现 {len(pkl_files)} 个 PKL 文件，输出文件夹: {output_path}")

  failed_files = []
  for pkl_file in pkl_files:
    csv_file = output_path / f"{pkl_file.stem}.csv"
    try:
      pkl_to_csv_g1(pkl_file, csv_file, source_order)
    except Exception as exc:
      failed_files.append((pkl_file, exc))
      print(f"转换失败: {pkl_file}，原因: {exc}")

  if failed_files:
    failed_text = "\n".join(f"- {path}: {exc}" for path, exc in failed_files)
    raise RuntimeError(
      f"批量转换完成，但有 {len(failed_files)} 个文件失败:\n{failed_text}"
    )

  print(f"批量转换成功，共转换 {len(pkl_files)} 个文件")


def main():
  parser = argparse.ArgumentParser(description="Unitree G1 29DoF PKL 转 CSV 工具")
  parser.add_argument(
    "--pkl", type=str, required=True, help="输入 .pkl 文件或文件夹路径"
  )
  parser.add_argument(
    "--csv",
    type=str,
    required=True,
    help="输出文件夹路径，生成的 .csv 与源 .pkl 同名",
  )
  parser.add_argument(
    "--source-order",
    type=str,
    choices=("lab", "csv"),
    default="lab",
    help="PKL 中 dof_pos 的关节顺序；lab 表示 legged_lab retarget 输出顺序，csv 表示已是 csv_to_npz.py 期望顺序",
  )
  args = parser.parse_args()

  convert_path(args.pkl, args.csv, args.source_order)


if __name__ == "__main__":
  main()
