import argparse
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

ASSET_ZOO_PATH = Path(__file__).resolve().parents[1] / "asset_zoo" / "robots"
QLMINI2_SOURCE_XML = ASSET_ZOO_PATH / "qlmini2" / "urdf" / "qlmini2.xml"
QLMINI2_DEPLOY_XML = ASSET_ZOO_PATH / "qlmini2" / "urdf" / "qlmini2_deploy.xml"
RLBOY_XML = ASSET_ZOO_PATH / "RL_BOY" / "RLBOY2sim.xml"

QLMINI2_JOINT_NAMES = (
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_foot_pitch_joint",
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_foot_pitch_joint",
  "waist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
)


def load_qlmini2_model(xml_path: str) -> mujoco.MjModel:
  """Load qlmini2 and inject its motor actuators when absent from the XML."""
  model = mujoco.MjModel.from_xml_path(xml_path)
  if model.nu:
    return model

  effort_limits = (
    (11, 6, 5, 6, 6) * 2
    + (6, 5, 1.8, 1.8, 1.8)
    + (
      5,
      1.8,
      1.8,
      1.8,
    )
  )
  armatures = (
    (0.005, 0.003, 0.002, 0.003, 0.003) * 2
    + (0.003,)
    + (
      0.002,
      0.0005,
      0.0005,
      0.0005,
    )
    + (
      0.002,
      0.0005,
      0.0005,
      0.0005,
    )
  )

  spec = mujoco.MjSpec.from_file(xml_path)
  for joint_name, effort_limit, armature in zip(
    QLMINI2_JOINT_NAMES, effort_limits, armatures, strict=True
  ):
    actuator = spec.add_actuator(name=joint_name, target=joint_name)
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
    actuator.gear[0] = 1.0
    actuator.forcelimited = True
    actuator.forcerange[:] = np.array((-effort_limit, effort_limit))
    actuator.ctrllimited = True
    actuator.ctrlrange[:] = np.array((-effort_limit, effort_limit))
    spec.joint(joint_name).armature = armature
  return spec.compile()


def quat_rotate_inverse(quat, world_vec):
  w, x, y, z = quat
  q_vec = np.array([x, y, z])
  t = np.cross(q_vec, world_vec) * 2.0
  return world_vec - w * t + np.cross(q_vec, t)


def projected_gravity(quat):
  world_gravity = np.array([0.0, 0.0, -1.0])
  return quat_rotate_inverse(quat, world_gravity)


def get_sensor_data(model: mujoco.MjModel, data: mujoco.MjData, name: str):
  """Return one MuJoCo sensor value by name."""
  sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
  if sensor_id < 0:
    raise ValueError(f"Model does not define required sensor {name!r}.")
  start = model.sensor_adr[sensor_id]
  stop = start + model.sensor_dim[sensor_id]
  return data.sensordata[start:stop].copy()


def build_velocity_observation(
  base_ang_vel: np.ndarray,
  gravity_orientation: np.ndarray,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
  action: np.ndarray,
  command: np.ndarray,
) -> np.ndarray:
  """Build an actor observation in velocity-task term order."""
  return np.concatenate(
    (
      base_ang_vel,
      gravity_orientation,
      joint_pos,
      joint_vel,
      action,
      command,
    )
  ).astype(np.float32, copy=False)


def configure_tracking_camera(model: mujoco.MjModel, viewer) -> None:
  """Make the passive viewer follow the body attached to the free joint."""
  free_joint_type = mujoco.mjtJoint.mjJNT_FREE.value
  free_joint_ids = np.flatnonzero(model.jnt_type == free_joint_type)
  if len(free_joint_ids) != 1:
    raise ValueError(
      f"Camera tracking requires exactly one free joint, found {len(free_joint_ids)}."
    )

  root_body_id = model.jnt_bodyid[free_joint_ids[0]]
  viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
  viewer.cam.trackbodyid = root_body_id
  viewer.cam.fixedcamid = -1
  viewer.cam.distance = 3.0
  viewer.cam.elevation = -20.0
  viewer.cam.azimuth = 135.0


def pd_control(target_q, q, kp, target_dq, dq, kd):
  return (target_q - q) * kp + (target_dq - dq) * kd


class KeyboardControlWindow:
  """Receive velocity commands in a dedicated Tk window."""

  def __init__(
    self,
    lin_vel_x_limit: float = 1.0,
    lin_vel_y_limit: float = 1.0,
    ang_vel_z_limit: float = 0.5,
  ):
    self.lin_vel_x = 0.0
    self.lin_vel_y = 0.0
    self.ang_vel_z = 0.0
    self.lin_vel_x_limit = lin_vel_x_limit
    self.lin_vel_y_limit = lin_vel_y_limit
    self.ang_vel_z_limit = ang_vel_z_limit
    self.heading_target = 0.0
    self.heading_mode = False
    self._lock = threading.Lock()
    self._root = None
    self._status = None
    self._thread = None
    self._started = threading.Event()
    self._startup_error = None
    self._running = True

  def start(self):
    """Open the controller window on a dedicated UI thread."""
    if self._thread is not None:
      return
    self._thread = threading.Thread(
      target=self._run_window,
      name="mjlab-keyboard-controller",
      daemon=True,
    )
    self._thread.start()
    self._started.wait(timeout=3.0)
    if self._startup_error is not None:
      raise RuntimeError(
        "Failed to open the keyboard control window."
      ) from self._startup_error
    if not self._started.is_set():
      raise RuntimeError("Timed out opening the keyboard control window.")

  def _run_window(self):
    try:
      import tkinter as tk

      root = tk.Tk()
      self._root = root
      root.title("mjlab Velocity Controller")
      root.resizable(False, False)
      root.protocol("WM_DELETE_WINDOW", self.close)

      tk.Label(
        root,
        text="Click this window, then use the keyboard",
        font=("Sans", 12, "bold"),
        padx=12,
        pady=8,
      ).grid(row=0, column=0, columnspan=3)
      tk.Label(
        root,
        text="W/S: forward   A/D: lateral   Q/E: yaw\n"
        "H: heading mode   F/G: heading target\n"
        "Space: zero velocity   Esc: quit",
        justify="left",
        padx=12,
        pady=6,
      ).grid(row=1, column=0, columnspan=3)

      for column, (label, key) in enumerate(
        (("Zero velocity", "SPACE"), ("Toggle heading", "H"), ("Quit", "ESCAPE"))
      ):
        tk.Button(
          root,
          text=label,
          command=lambda value=key: self._handle_key(value),
          width=16,
        ).grid(row=2, column=column, padx=4, pady=4)

      self._status = tk.StringVar()
      tk.Label(
        root,
        textvariable=self._status,
        font=("Monospace", 11),
        padx=12,
        pady=8,
      ).grid(row=3, column=0, columnspan=3)
      self._refresh_status()
      root.bind("<KeyPress>", self._on_key_press)
      root.focus_force()
      self._started.set()
      root.mainloop()
    except Exception as error:
      self._startup_error = error
      self._started.set()
      with self._lock:
        self._running = False

  def _on_key_press(self, event):
    self._handle_key(event.keysym)

  def _handle_key(self, key):
    key = str(key).upper()
    with self._lock:
      if key == "W":
        self.lin_vel_x = np.clip(
          self.lin_vel_x + 0.1, -self.lin_vel_x_limit, self.lin_vel_x_limit
        )
      elif key == "S":
        self.lin_vel_x = np.clip(
          self.lin_vel_x - 0.1, -self.lin_vel_x_limit, self.lin_vel_x_limit
        )
      elif key == "A":
        self.lin_vel_y = np.clip(
          self.lin_vel_y + 0.1, -self.lin_vel_y_limit, self.lin_vel_y_limit
        )
      elif key == "D":
        self.lin_vel_y = np.clip(
          self.lin_vel_y - 0.1, -self.lin_vel_y_limit, self.lin_vel_y_limit
        )
      elif key == "Q":
        self.ang_vel_z = np.clip(
          self.ang_vel_z + 0.1, -self.ang_vel_z_limit, self.ang_vel_z_limit
        )
      elif key == "E":
        self.ang_vel_z = np.clip(
          self.ang_vel_z - 0.1, -self.ang_vel_z_limit, self.ang_vel_z_limit
        )
      elif key == "H":
        self.heading_mode = not self.heading_mode
      elif key == "F":
        self.heading_target = wrap_to_pi(self.heading_target + 0.2)
      elif key == "G":
        self.heading_target = wrap_to_pi(self.heading_target - 0.2)
      elif key == "SPACE":
        self.lin_vel_x = self.lin_vel_y = self.ang_vel_z = 0.0
      elif key == "ESCAPE":
        self._running = False
        if self._root is not None:
          self._root.after(0, self._root.destroy)
    self._refresh_status()

  def _refresh_status(self):
    if self._status is None:
      return
    with self._lock:
      mode = "heading" if self.heading_mode else "direct"
      status = (
        f"command: x={self.lin_vel_x:+.1f}  y={self.lin_vel_y:+.1f}  "
        f"yaw={self.ang_vel_z:+.1f}  mode={mode}  "
        f"heading={self.heading_target:+.1f}"
      )
    self._status.set(status)

  def get_command(self):
    with self._lock:
      return (
        self.lin_vel_x,
        self.lin_vel_y,
        self.ang_vel_z,
        self.heading_target,
        self.heading_mode,
      )

  def is_running(self):
    with self._lock:
      return self._running

  def close(self):
    with self._lock:
      self._running = False
    if self._root is not None:
      try:
        self._root.after(0, self._root.destroy)
      except RuntimeError:
        pass


def wrap_to_pi(angle):
  return np.arctan2(np.sin(angle), np.cos(angle))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--robot", choices=("qlmini2", "rlboy"), default="qlmini2")
  parser.add_argument("--xml-path", type=str, default=None)
  parser.add_argument("--policy-path", type=Path, required=True)
  parser.add_argument(
    "--track-camera",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Follow the robot root body in the MuJoCo viewer (default: enabled).",
  )
  args = parser.parse_args()

  simulation_dt = 0.005
  control_decimation = 4

  STIFFNESS_LIST = [
    16.34,
    16.34,
    16.34,
    16.34,
    8.241,
    16.34,
    16.34,
    16.34,
    16.34,
    8.241,
    8.241,
    8.241,
    1.683,
    1.683,
    1.683,
    1.683,
    1.683,
    1.683,
    1.683,
    1.683,
  ]
  kps = np.array(STIFFNESS_LIST, dtype=np.float32)
  DAMPING_LIST = [
    1.045,
    1.045,
    1.045,
    1.045,
    0.5256,
    1.045,
    1.045,
    1.045,
    1.045,
    0.5256,
    0.5256,
    0.5256,
    0.1069,
    0.1069,
    0.1069,
    0.1069,
    0.1069,
    0.1069,
    0.1069,
    0.1069,
  ]
  kds = np.array(DAMPING_LIST, dtype=np.float32)
  JOINT_POS_LIST = [
    0,
    0,
    -0.2,
    0.4,
    -0.2,
    0,
    0,
    -0.2,
    0.4,
    -0.2,
    0,
    0,
    0.15,
    0.3,
    0,
    0.9,
    0.15,
    -0.3,
    0,
    0.9,
  ]
  ACTION_SCALE_LIST = [
    0.306,
    0.306,
    0.306,
    0.306,
    0.334,
    0.306,
    0.306,
    0.306,
    0.306,
    0.334,
    0.334,
    0.334,
    0.445,
    0.445,
    0.445,
    0.445,
    0.445,
    0.445,
    0.445,
    0.445,
  ]
  action_scales = np.array(ACTION_SCALE_LIST, dtype=np.float32)
  default_angles = np.array(JOINT_POS_LIST, dtype=np.float32)

  if args.robot == "qlmini2":
    kps = np.array(
      [16, 16, 8, 16, 16] * 2 + [16, 8, 5, 5, 5, 8, 5, 5, 5],
      dtype=np.float32,
    )
    kds = np.array(
      [0.7, 0.7, 0.5, 0.7, 0.7] * 2 + [0.7, 0.5, 0.3, 0.3, 0.3, 0.5, 0.3, 0.3, 0.3],
      dtype=np.float32,
    )
    default_angles = np.array(
      [-0.15, 0, 0, 0.3, -0.15] * 2 + [0, 0.15, -0.2, 0, 0.7, 0.15, 0.2, 0, 0.7],
      dtype=np.float32,
    )
    action_scales = np.array(
      [0.171875, 0.09375, 0.15625, 0.09375, 0.09375] * 2
      + [0.09375, 0.15625, 0.09, 0.09, 0.09, 0.15625, 0.09, 0.09, 0.09],
      dtype=np.float32,
    )
    xml_path = args.xml_path or str(QLMINI2_DEPLOY_XML)
  else:
    xml_path = args.xml_path or str(RLBOY_XML)

  num_actions = len(default_angles)
  num_obs = 3 + 3 + num_actions * 3 + 3
  cmd = np.array([0, 0, 0], dtype=np.float32)

  action = np.zeros(num_actions, dtype=np.float32)
  target_dof_pos = default_angles.copy()
  obs = np.zeros(num_obs, dtype=np.float32)
  counter = 0

  policy_path = args.policy_path
  if policy_path.suffix.lower() != ".onnx":
    raise ValueError(
      "--policy-path must be an exported .onnx policy. A training checkpoint "
      "(.pt) is neither needed nor supported by this deployment script."
    )

  m = (
    load_qlmini2_model(xml_path)
    if args.robot == "qlmini2"
    else mujoco.MjModel.from_xml_path(xml_path)
  )
  if m.nu != num_actions:
    raise ValueError(
      f"{args.robot} has {m.nu} actuators in {xml_path}, expected {num_actions}"
    )
  d = mujoco.MjData(m)
  d.qpos[7:] = default_angles
  mujoco.mj_forward(m, d)
  m.opt.timestep = simulation_dt

  policy = ort.InferenceSession(str(policy_path))
  input_name = policy.get_inputs()[0].name
  output_name = policy.get_outputs()[0].name
  print(f"Loaded ONNX policy from {policy_path}")

  velocity_limits = (1.0, 0.8, 1.0) if args.robot == "qlmini2" else (1.0, 1.0, 0.5)
  kbd = KeyboardControlWindow(*velocity_limits)
  kbd.start()

  def get_heading():
    quat = d.qpos[3:7]
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

  heading_control_stiffness = 0.5

  try:
    with mujoco.viewer.launch_passive(m, d) as viewer:
      if args.track_camera:
        configure_tracking_camera(m, viewer)
      start = time.time()
      while viewer.is_running() and kbd.is_running():
        step_start = time.time()
        tau = pd_control(
          target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds
        )
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        counter += 1
        if counter % control_decimation == 0:
          lin_vel_x, lin_vel_y, ang_vel_z, heading_target, heading_mode = (
            kbd.get_command()
          )

          if heading_mode:
            heading_error = wrap_to_pi(heading_target - get_heading())
            ang_vel_z = np.clip(
              heading_control_stiffness * heading_error,
              -velocity_limits[2],
              velocity_limits[2],
            )

          cmd[0] = lin_vel_x
          cmd[1] = lin_vel_y
          cmd[2] = ang_vel_z

          if counter % 400 == 0:
            mode_str = "heading" if heading_mode else "direct "
            print(
              f"cmd: [{lin_vel_x:+.2f}, {lin_vel_y:+.2f}, {ang_vel_z:+.2f}] "
              f"mode: {mode_str} | heading_target: {heading_target:+.2f}"
            )

          quat = d.qpos[3:7]
          # Match the velocity task exactly: IMU sensors are body-frame values.
          base_ang_vel = get_sensor_data(m, d, "imu_ang_vel")
          gravity_orientation = projected_gravity(quat)
          qj = d.qpos[7:]
          joint_pos = qj - default_angles
          dqj = d.qvel[6:]
          joint_vel = dqj

          obs[:] = build_velocity_observation(
            base_ang_vel,
            gravity_orientation,
            joint_pos,
            joint_vel,
            action,
            cmd,
          )

          # runner.export_policy_as_onnx() embeds obs_normalizer in the ONNX
          # graph. Passing normalized data here would normalize it twice.
          action = policy.run([output_name], {input_name: obs[np.newaxis, :]})[0]
          action = np.asarray(action, dtype=np.float32).reshape(-1)

          target_dof_pos = default_angles + action * action_scales

        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)
  finally:
    kbd.close()
