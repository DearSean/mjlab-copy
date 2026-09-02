"""Play raw and simulator-settled G1 recovery poses in MuJoCo."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import mujoco.viewer
import numpy as np
import tyro

import mjlab
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from mjlab.tasks.velocity.recovery_data.g1_schema import (
  G1_JOINT_NAMES,
  G1_PHYSICAL_INIT_SCHEMA_VERSION,
  G1_RECOVERY_TARGET_FPS,
)

PoseSource = Literal["motion", "physical", "compare"]


@dataclass(frozen=True)
class _PoseSequence:
  frame: np.ndarray
  root_position: np.ndarray
  root_quaternion_xyzw: np.ndarray
  joint_position: np.ndarray


def _load_motion(
  motion_file: Path,
  frame_range: tuple[int, int] | None,
) -> tuple[_PoseSequence, int, int]:
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
    _PoseSequence(
      frame=np.arange(start, end, dtype=np.int64),
      root_position=root_position[start:end],
      root_quaternion_xyzw=root_quaternion_xyzw[start:end],
      joint_position=joint_position[start:end],
    ),
    start,
    end,
  )


def _load_physical(
  physical_init_file: Path,
  clip_id: str,
  start: int,
  end: int,
) -> tuple[_PoseSequence, dict[str, np.ndarray]]:
  if not physical_init_file.is_file():
    raise FileNotFoundError(
      f"G1 physical initialization bank does not exist: {physical_init_file}"
    )
  with np.load(physical_init_file, allow_pickle=False) as states:
    schema = str(np.asarray(states["schema_version"]).item())
    if schema != G1_PHYSICAL_INIT_SCHEMA_VERSION:
      raise ValueError(
        f"Unsupported physical initialization schema {schema!r}; "
        f"expected {G1_PHYSICAL_INIT_SCHEMA_VERSION!r}."
      )
    required = (
      "clip_id",
      "frame",
      "root_position",
      "root_quaternion_xyzw",
      "joint_position",
    )
    missing = [name for name in required if name not in states]
    if missing:
      raise ValueError(f"Physical initialization bank is missing: {', '.join(missing)}")
    clip_ids = np.asarray(states["clip_id"]).astype(str)
    frames = np.asarray(states["frame"], dtype=np.int64)
    selected = np.flatnonzero(
      (clip_ids == clip_id) & (frames >= start) & (frames < end)
    )
    if len(selected) == 0:
      raise ValueError(
        f"No simulator-validated reset frame for {clip_id} in [{start}, {end})."
      )
    selected = selected[np.argsort(frames[selected])]
    selected_frames = frames[selected]
    if len(np.unique(selected_frames)) != len(selected_frames):
      raise ValueError("Physical initialization bank contains duplicate frames.")
    sequence = _PoseSequence(
      frame=selected_frames,
      root_position=np.asarray(states["root_position"][selected], dtype=np.float64),
      root_quaternion_xyzw=np.asarray(
        states["root_quaternion_xyzw"][selected], dtype=np.float64
      ),
      joint_position=np.asarray(states["joint_position"][selected], dtype=np.float64),
    )
    audit_names = (
      "root_projection_m",
      "joint_projection_max_rad",
      "settle_translation_m",
      "settle_rotation_rad",
      "support_fraction",
      "max_contact_force_n",
      "max_self_penetration_m",
      "max_ground_penetration_m",
    )
    audit = {
      name: np.asarray(states[name][selected]) for name in audit_names if name in states
    }
  return sequence, audit


def _select_motion_frames(motion: _PoseSequence, frames: np.ndarray) -> _PoseSequence:
  lookup = {int(frame): index for index, frame in enumerate(motion.frame)}
  try:
    indices = np.asarray([lookup[int(frame)] for frame in frames], dtype=np.int64)
  except KeyError as error:
    raise ValueError(
      f"Physical frame is outside the motion selection: {error}"
    ) from error
  return _PoseSequence(
    frame=motion.frame[indices],
    root_position=motion.root_position[indices],
    root_quaternion_xyzw=motion.root_quaternion_xyzw[indices],
    joint_position=motion.joint_position[indices],
  )


def _build_model(prefixes: tuple[str, ...]) -> mujoco.MjModel:
  spec = mujoco.MjSpec()
  for prefix in prefixes:
    robot = Entity(get_g1_robot_cfg())
    if tuple(robot.joint_names) != G1_JOINT_NAMES:
      raise ValueError("Compiled recovery joint order does not match the G1 model.")
    while robot.spec.keys:
      robot.spec.delete(robot.spec.keys[0])
    frame = spec.worldbody.add_frame()
    spec.attach(robot.spec, prefix=f"{prefix}/", frame=frame)
  spec.worldbody.add_geom(
    name="recovery_viewer_ground",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[5.0, 5.0, 0.1],
    rgba=[0.35, 0.35, 0.35, 1.0],
    condim=3,
    friction=[0.6, 0.005, 0.0001],
  )
  return spec.compile()


def _qpos_addresses(model: mujoco.MjModel, prefix: str) -> tuple[int, list[int], int]:
  prefix_with_separator = f"{prefix}/"
  free_joint_ids = [
    joint_id
    for joint_id in range(model.njnt)
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id).startswith(
      prefix_with_separator
    )
  ]
  if len(free_joint_ids) != 1:
    raise RuntimeError(
      f"Expected one free joint for {prefix}, found {len(free_joint_ids)}."
    )
  free_joint_id = free_joint_ids[0]
  root_qpos_adr = int(model.jnt_qposadr[free_joint_id])
  joint_qpos_adrs = []
  for name in G1_JOINT_NAMES:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/{name}")
    if joint_id < 0:
      raise RuntimeError(f"G1 joint is missing from the model: {prefix}/{name}")
    joint_qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
  return root_qpos_adr, joint_qpos_adrs, int(model.jnt_bodyid[free_joint_id])


def _ground_contact_names(
  model: mujoco.MjModel, data: mujoco.MjData, prefix: str
) -> tuple[str, ...]:
  ground_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "recovery_viewer_ground"
  )
  names: set[str] = set()
  prefix_with_separator = f"{prefix}/"
  for contact in data.contact[: data.ncon]:
    if contact.geom1 == ground_id:
      geom_id = int(contact.geom2)
    elif contact.geom2 == ground_id:
      geom_id = int(contact.geom1)
    else:
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name is not None and name.startswith(prefix_with_separator):
      names.add(name.removeprefix(prefix_with_separator))
  return tuple(sorted(names))


def _print_physical_audit(
  clip_id: str,
  physical: _PoseSequence,
  audit: dict[str, np.ndarray],
) -> None:
  print(f"Physical reset {clip_id}: accepted target frames {physical.frame.tolist()}")
  for name, values in audit.items():
    print(f"  {name}: min={float(values.min()):.5g}, max={float(values.max()):.5g}")


def main(
  motion_file: Path,
  frame_range: tuple[int, int] | None = None,
  speed: float = 0.25,
  loop: bool = True,
  check_only: bool = False,
  pose_source: PoseSource = "motion",
  physical_init_file: Path = Path("artifacts/g1_recovery/physical_init.npz"),
  compare_separation_m: float = 1.2,
) -> None:
  """Play raw motion, settled reset poses, or both side by side.

  Args:
    motion_file: Compiled recovery NPZ under ``artifacts/g1_recovery/clips``.
    frame_range: Optional half-open target-frame range ``(start, end)``.
    speed: Playback speed multiplier; 0.25 gives four-times slow motion.
    loop: Whether to restart the selected frames after the last frame.
    check_only: Validate and describe the selection without opening a window.
    pose_source: ``motion``, ``physical``, or side-by-side ``compare``.
    physical_init_file: Simulator-validated reset-state bank.
    compare_separation_m: Lateral spacing in side-by-side comparison mode.
  """
  if speed <= 0.0:
    raise ValueError(f"speed must be positive, got {speed}.")
  if compare_separation_m <= 0.0:
    raise ValueError("compare_separation_m must be positive.")
  motion, start, end = _load_motion(motion_file, frame_range)
  print(
    f"Loaded {motion_file}: target frames [{start}, {end}), "
    f"{len(motion.frame) / G1_RECOVERY_TARGET_FPS:.3f} s at "
    f"{G1_RECOVERY_TARGET_FPS:g} Hz"
  )

  sequences: dict[str, _PoseSequence]
  offsets: dict[str, float]
  if pose_source == "motion":
    sequences = {"motion": motion}
    offsets = {"motion": 0.0}
  else:
    physical, audit = _load_physical(physical_init_file, motion_file.stem, start, end)
    _print_physical_audit(motion_file.stem, physical, audit)
    if pose_source == "physical":
      sequences = {"physical": physical}
      offsets = {"physical": 0.0}
    else:
      paired_motion = _select_motion_frames(motion, physical.frame)
      half_separation = compare_separation_m / 2.0
      sequences = {"motion": paired_motion, "physical": physical}
      offsets = {"motion": -half_separation, "physical": half_separation}
      print("Compare layout: motion at -Y, physical reset at +Y.")

  if check_only and pose_source == "motion":
    return

  model = _build_model(tuple(sequences))
  data = mujoco.MjData(model)
  addresses = {prefix: _qpos_addresses(model, prefix) for prefix in sequences}

  def apply_frame(frame_index: int) -> None:
    for prefix, sequence in sequences.items():
      root_qpos_adr, joint_qpos_adrs, _ = addresses[prefix]
      root_position = sequence.root_position[frame_index].copy()
      root_position[1] += offsets[prefix]
      data.qpos[root_qpos_adr : root_qpos_adr + 3] = root_position
      data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = sequence.root_quaternion_xyzw[
        frame_index, [3, 0, 1, 2]
      ]
      data.qpos[joint_qpos_adrs] = sequence.joint_position[frame_index]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

  apply_frame(0)
  if "physical" in sequences:
    physical = sequences["physical"]
    print("Physical reset ground contacts:")
    for frame_index, frame in enumerate(physical.frame):
      apply_frame(frame_index)
      contacts = _ground_contact_names(model, data, "physical")
      label = ", ".join(contacts) if contacts else "NONE"
      print(f"  frame {int(frame)}: {label}")
    apply_frame(0)
  if check_only:
    return
  with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
  ) as viewer:
    if len(sequences) == 1:
      root_body_id = next(iter(addresses.values()))[2]
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      viewer.cam.trackbodyid = root_body_id
      viewer.cam.distance = 2.5
    else:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
      viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
      viewer.cam.distance = 3.5
    viewer.cam.elevation = -10.0
    viewer.cam.azimuth = 135.0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    viewer.sync()

    frame_index = 0
    frame_count = len(next(iter(sequences.values())).frame)
    frame_period = 1.0 / (G1_RECOVERY_TARGET_FPS * speed)
    deadline = time.monotonic()
    while viewer.is_running():
      apply_frame(frame_index)
      viewer.sync()
      frame_index += 1
      if frame_index == frame_count:
        if not loop:
          break
        frame_index = 0
      deadline += frame_period
      remaining = deadline - time.monotonic()
      if remaining > 0.0:
        time.sleep(remaining)
      else:
        deadline = time.monotonic()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
