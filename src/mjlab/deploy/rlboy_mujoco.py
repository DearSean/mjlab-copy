import threading
import time
import argparse

import mujoco
import mujoco.viewer
import numpy as np
import torch
import onnxruntime as ort


def quat_rotate_inverse(quat, world_vec):
    w, x, y, z = quat
    q_vec = np.array([x, y, z])
    t = np.cross(q_vec, world_vec) * 2.0
    return world_vec + w * t + np.cross(q_vec, t)


def projected_gravity(quat):
    world_gravity = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, world_gravity)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


class PynputKeyboardCommander:
    """Keyboard commander using pynput (works in uv run, conda, venv, etc.)"""

    def __init__(self):
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_z = 0.0
        self.heading_target = 0.0
        self.heading_mode = False
        self._lock = threading.Lock()
        self._listener = None

    def _on_press(self, key):
        try:
            if key.char in ('w', 'W'):
                self.update(lin_vel_x=self.lin_vel_x + 0.1)
            elif key.char in ('s', 'S'):
                self.update(lin_vel_x=self.lin_vel_x - 0.1)
            elif key.char in ('a', 'A'):
                self.update(lin_vel_y=self.lin_vel_y + 0.1)
            elif key.char in ('d', 'D'):
                self.update(lin_vel_y=self.lin_vel_y - 0.1)
            elif key.char in ('q', 'Q'):
                self.update(ang_vel_z=self.ang_vel_z + 0.1)
            elif key.char in ('e', 'E'):
                self.update(ang_vel_z=self.ang_vel_z - 0.1)
            elif key.char in ('h', 'H'):
                self.update(heading_mode=not self.heading_mode)
            elif key.char in ('f', 'F'):
                self.update(heading_target=self.heading_target + 0.2)
            elif key.char in ('g', 'G'):
                self.update(heading_target=self.heading_target - 0.2)
            elif key.char == ' ':
                self.update(lin_vel_x=0.0, lin_vel_y=0.0, ang_vel_z=0.0)
        except AttributeError:
            pass

    def start(self):
        from pynput import keyboard
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()
        print("\n=== Keyboard Commander (pynput) ===")
        print("W/S: Forward/Backward")
        print("A/D: Strafe Left/Right")
        print("Q/E: Rotate CCW/CW")
        print("H: Toggle heading mode")
        print("F/G: Adjust heading target")
        print("Space: Reset all to zero")
        print("=========================\n")

    def stop(self):
        if self._listener:
            self._listener.stop()

    def update(self, lin_vel_x=None, lin_vel_y=None, ang_vel_z=None,
               heading_target=None, heading_mode=None):
        with self._lock:
            if lin_vel_x is not None:
                self.lin_vel_x = np.clip(lin_vel_x, -1.0, 1.0)
            if lin_vel_y is not None:
                self.lin_vel_y = np.clip(lin_vel_y, -1.0, 1.0)
            if ang_vel_z is not None:
                self.ang_vel_z = np.clip(ang_vel_z, -0.5, 0.5)
            if heading_target is not None:
                self.heading_target = wrap_to_pi(heading_target)
            if heading_mode is not None:
                self.heading_mode = heading_mode

    def get_command(self):
        with self._lock:
            return (self.lin_vel_x, self.lin_vel_y, self.ang_vel_z,
                    self.heading_target, self.heading_mode)


def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def load_obs_normalizer_stats(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_sd = ckpt.get("actor_state_dict", ckpt)
    mean = actor_sd["obs_normalizer._mean"].squeeze().numpy()
    std = actor_sd["obs_normalizer._std"].squeeze().numpy()
    return mean.astype(np.float32), std.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-path", type=str,
                        default="/home/rx03116/GithubItems/mjlab/src/mjlab/asset_zoo/robots/RL_BOY/RLBOY2sim.xml")
    parser.add_argument("--policy-path", type=str,
                        default="/home/rx03116/GithubItems/mjlab/logs/rsl_rl/rlboy_velocity/2026-06-16_15-43-59/model_3100.pt")
    parser.add_argument("--normalizer-checkpoint", type=str, default=None)
    args = parser.parse_args()

    simulation_dt = 0.005
    control_decimation = 4

    STIFFNESS_LIST = [
        16.34, 16.34, 16.34, 16.34, 8.241,
        16.34, 16.34, 16.34, 16.34, 8.241,
        8.241, 8.241,
        1.683, 1.683, 1.683, 1.683,
        1.683, 1.683, 1.683, 1.683,
    ]
    kps = np.array(STIFFNESS_LIST, dtype=np.float32)
    DAMPING_LIST = [
        1.045, 1.045, 1.045, 1.045, 0.5256,
        1.045, 1.045, 1.045, 1.045, 0.5256,
        0.5256, 0.5256,
        0.1069, 0.1069, 0.1069, 0.1069,
        0.1069, 0.1069, 0.1069, 0.1069,
    ]
    kds = np.array(DAMPING_LIST, dtype=np.float32)
    JOINT_POS_LIST = [
        0, 0, -0.2, 0.4, -0.2,
        0, 0, -0.2, 0.4, -0.2,
        0, 0,
        0.15, 0.3, 0, 0.9,
        0.15, -0.3, 0, 0.9,
    ]
    ACTION_SCALE_LIST = [
        0.306, 0.306, 0.306, 0.306, 0.334,
        0.306, 0.306, 0.306, 0.306, 0.334,
        0.334, 0.334,
        0.445, 0.445, 0.445, 0.445,
        0.445, 0.445, 0.445, 0.445,
    ]
    action_scales = np.array(ACTION_SCALE_LIST, dtype=np.float32)
    default_angles = np.array(JOINT_POS_LIST, dtype=np.float32)

    num_actions = 20
    num_obs = 72
    cmd = np.array([0, 0, 0], dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    xml_path = args.xml_path
    policy_path = args.policy_path

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    use_onnx = policy_path.lower().endswith(".onnx")
    if use_onnx:
        policy = ort.InferenceSession(policy_path)
        input_name = policy.get_inputs()[0].name
        output_name = policy.get_outputs()[0].name
        print(f"Loaded ONNX policy from {policy_path}")
        import onnx
        model = onnx.load(policy_path)
        for prop in model.metadata_props:
            print(f"{prop.key}: {prop.value}")
    else:
        policy = torch.jit.load(policy_path)
        print(f"Loaded TorchScript policy from {policy_path}")

    normalizer_ckpt = args.normalizer_checkpoint or policy_path
    try:
        obs_mean, obs_std = load_obs_normalizer_stats(normalizer_ckpt)
        print(f"Loaded obs normalizer stats from {normalizer_ckpt}")
    except Exception as e:
        print(f"Warning: could not load obs normalizer stats: {e}")
        obs_mean = np.zeros(num_obs, dtype=np.float32)
        obs_std = np.ones(num_obs, dtype=np.float32)

    # 启动 pynput 键盘监听
    kbd = PynputKeyboardCommander()
    kbd.start()

    def get_heading():
        quat = d.qpos[3:7]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    heading_control_stiffness = 0.5

    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            start = time.time()
            while viewer.is_running():
                step_start = time.time()
                tau = pd_control(
                    target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds
                )
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)

                counter += 1
                if counter % control_decimation == 0:
                    lin_vel_x, lin_vel_y, ang_vel_z, heading_target, heading_mode = kbd.get_command()

                    if heading_mode:
                        heading_error = wrap_to_pi(heading_target - get_heading())
                        ang_vel_z = np.clip(
                            heading_control_stiffness * heading_error,
                            -0.5, 0.5
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
                    world_lin_vel = d.qvel[0:3]
                    base_lin_vel = quat_rotate_inverse(quat, world_lin_vel)
                    world_ang_vel = d.qvel[3:6]
                    base_ang_vel = quat_rotate_inverse(quat, world_ang_vel)
                    gravity_orientation = projected_gravity(quat)
                    qj = d.qpos[7:]
                    joint_pos = qj - default_angles
                    dqj = d.qvel[6:]
                    joint_vel = dqj

                    idx = 0
                    obs[idx : idx + 3] = base_lin_vel
                    idx += 3
                    obs[idx : idx + 3] = base_ang_vel
                    idx += 3
                    obs[idx : idx + 3] = gravity_orientation
                    idx += 3
                    obs[idx : idx + num_actions] = joint_pos
                    idx += num_actions
                    obs[idx : idx + num_actions] = joint_vel
                    idx += num_actions
                    obs[idx : idx + num_actions] = action
                    idx += num_actions
                    obs[idx : idx + 3] = cmd
                    idx += 3

                    obs_normalized = (obs - obs_mean) / obs_std
                    obs_tensor = torch.from_numpy(obs_normalized).unsqueeze(0)

                    if use_onnx:
                        action = policy.run([output_name], {input_name: obs_tensor.numpy()})[0]
                        action = np.asarray(action).reshape(-1)
                    else:
                        action = policy(obs_tensor).detach().numpy().squeeze()

                    target_dof_pos = default_angles + action * action_scales

                viewer.sync()

                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    finally:
        kbd.stop()