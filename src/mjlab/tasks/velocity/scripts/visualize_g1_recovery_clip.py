"""Play a compiled G1 recovery clip in the native MuJoCo viewer.

Example:
  uv run python -m mjlab.tasks.velocity.scripts.visualize_g1_recovery_clip \
    --motion-file \
    artifacts/g1_recovery/clips/032_fallAndGetUp1_subject5_frames_01233-01291.npz \
    --frame-range "(0, 10)" --speed 0.25
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro

import mjlab
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from mjlab.tasks.velocity.recovery_data.g1_schema import (
  G1_JOINT_NAMES,
  G1_RECOVERY_TARGET_FPS,
)


def _load_clip(
  motion_file: Path,
  frame_range: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
  if not motion_file.is_file():
    raise FileNotFoundError(f"G1 recovery clip does not exist: {motion_file}")
  with np.load(motion_file, allow_pickle=False) as clip:
    required = ("root_position", "root_quaternion_xyzw", "joint_position")
    missing = [name for name in required if name not in clip]
    if missing:
      raise ValueError(f"Recovery clip is missing arrays: {', '.join(missing)}")
    root_position = np.asarray(clip["root_position"], dtype=np.float64)
    root_quaternion_xyzw = np.asarray(clip["root_quaternion_xyzw"], dtype=np.float64)
    joint_position = np.asarray(clip["joint_position"], dtype=np.float64)

  frame_count = len(root_position)
  expected = (
    ("root_position", root_position.shape, (frame_count, 3)),
    ("root_quaternion_xyzw", root_quaternion_xyzw.shape, (frame_count, 4)),
    ("joint_position", joint_position.shape, (frame_count, len(G1_JOINT_NAMES))),
  )
  for name, actual, target in expected:
    if actual != target:
      raise ValueError(f"{name} must have shape {target}, got {actual}.")
  if frame_count == 0:
    raise ValueError("G1 recovery clip contains no frames.")
  if not all(
    np.isfinite(values).all()
    for values in (root_position, root_quaternion_xyzw, joint_position)
  ):
    raise ValueError("G1 recovery clip contains NaN or infinity.")

  start, end = (0, frame_count) if frame_range is None else frame_range
  if not 0 <= start < end <= frame_count:
    raise ValueError(
      f"frame_range must satisfy 0 <= start < end <= {frame_count}, got {(start, end)}."
    )
  return (
    root_position[start:end],
    root_quaternion_xyzw[start:end],
    joint_position[start:end],
    start,
    end,
  )


def main(
  motion_file: Path,
  frame_range: tuple[int, int] | None = None,
  speed: float = 0.25,
  loop: bool = True,
  check_only: bool = False,
) -> None:
  """Play all or part of one compiled 50 Hz G1 recovery clip.

  Args:
    motion_file: Compiled recovery NPZ under ``artifacts/g1_recovery/clips``.
    frame_range: Optional half-open target-frame range ``(start, end)``.
    speed: Playback speed multiplier; 0.25 gives four-times slow motion.
    loop: Whether to restart the selected frames after the last frame.
    check_only: Validate and describe the selection without opening a window.
  """
  if speed <= 0.0:
    raise ValueError(f"speed must be positive, got {speed}.")
  root_position, root_quaternion_xyzw, joint_position, start, end = _load_clip(
    motion_file, frame_range
  )
  duration = len(root_position) / G1_RECOVERY_TARGET_FPS
  print(
    f"Loaded {motion_file}: target frames [{start}, {end}), "
    f"{duration:.3f} s at {G1_RECOVERY_TARGET_FPS:g} Hz"
  )
  if check_only:
    return

  robot = Entity(get_g1_robot_cfg())
  if tuple(robot.joint_names) != G1_JOINT_NAMES:
    raise ValueError("Compiled recovery joint order does not match the G1 model.")
  spec = robot.spec
  spec.worldbody.add_geom(
    name="recovery_viewer_ground",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[5.0, 5.0, 0.1],
    rgba=[0.35, 0.35, 0.35, 1.0],
  )
  model = spec.compile()
  data = mujoco.MjData(model)

  free_joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
  if free_joint_ids.size != 1:
    raise RuntimeError(f"Expected one free joint, found {free_joint_ids.size}.")
  root_qpos_adr = int(model.jnt_qposadr[free_joint_ids[0]])
  joint_qpos_adrs = []
  for name in G1_JOINT_NAMES:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
      raise RuntimeError(f"G1 joint is missing from the model: {name}")
    joint_qpos_adrs.append(int(model.jnt_qposadr[joint_id]))

  def apply_frame(frame: int) -> None:
    data.qpos[root_qpos_adr : root_qpos_adr + 3] = root_position[frame]
    data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = root_quaternion_xyzw[
      frame, [3, 0, 1, 2]
    ]
    data.qpos[joint_qpos_adrs] = joint_position[frame]
    mujoco.mj_forward(model, data)

  apply_frame(0)
  root_body_id = int(model.jnt_bodyid[free_joint_ids[0]])
  with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
  ) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = root_body_id
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -10.0
    viewer.cam.azimuth = 135.0
    viewer.sync()

    frame = 0
    frame_period = 1.0 / (G1_RECOVERY_TARGET_FPS * speed)
    deadline = time.monotonic()
    while viewer.is_running():
      apply_frame(frame)
      viewer.sync()
      frame += 1
      if frame == len(root_position):
        if not loop:
          break
        frame = 0
      deadline += frame_period
      remaining = deadline - time.monotonic()
      if remaining > 0.0:
        time.sleep(remaining)
      else:
        deadline = time.monotonic()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
