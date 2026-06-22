import argparse
import time

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


class RandomCommandSampler:
    """Random velocity command sampler matching Mjlab-Velocity-Flat-RL_BOY play mode."""

    def __init__(self):
        # Ranges matching rlboy_flat_env_cfg(play=True)
        self.lin_vel_x_range = (-0.5, 0.5)
        self.lin_vel_y_range = (-0.5, 0.5)
        self.ang_vel_z_range = (-0.5, 0.5)
        # Resampling time range (seconds)
        self.resampling_time_range = (3.0, 8.0)
        self._next_resample_time = 0.0
        self._current_cmd = np.zeros(3, dtype=np.float32)

    def start(self):
        print("\n=== Random Command Sampler ===")
        print(f"lin_vel_x: {self.lin_vel_x_range}")
        print(f"lin_vel_y: {self.lin_vel_y_range}")
        print(f"ang_vel_z: {self.ang_vel_z_range}")
        print(f"resampling every {self.resampling_time_range[0]}-{self.resampling_time_range[1]}s")
        print("=========================\n")

    def _resample(self):
        self._current_cmd[0] = np.random.uniform(*self.lin_vel_x_range)
        self._current_cmd[1] = np.random.uniform(*self.lin_vel_y_range)
        self._current_cmd[2] = np.random.uniform(*self.ang_vel_z_range)

    def update(self, simulation_time):
        if simulation_time >= self._next_resample_time:
            self._resample()
            self._next_resample_time = simulation_time + np.random.uniform(
                *self.resampling_time_range
            )

    def get_command(self):
        return self._current_cmd.copy()


def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-path", type=str,
                        default="/home/rx03116/GithubItems/mjlab/src/mjlab/asset_zoo/robots/RL_BOY/RLBOY2sim.xml")
    parser.add_argument("--policy-path", type=str,
                        default="/home/rx03116/GithubItems/mjlab/logs/rsl_rl/rlboy_velocity/2026-06-16_15-43-59/model_3100.pt")
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
        0.334, 0,
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

    # 根据 policy 文件后缀自动选择推理后端：.onnx → ONNX Runtime, .pt → TorchScript JIT
    suffix = policy_path.lower().rsplit(".", 1)[-1]
    if suffix == "onnx":
      policy = ort.InferenceSession(policy_path)
      input_name = policy.get_inputs()[0].name
      output_name = policy.get_outputs()[0].name
      print(f"Loaded ONNX policy from {policy_path}")
      import onnx
      model = onnx.load(policy_path)
      for prop in model.metadata_props:
        print(f"{prop.key}: {prop.value}")
    elif suffix == "pt":
      policy = torch.jit.load(policy_path)
      policy.eval()
      print(f"Loaded TorchScript policy from {policy_path}")
    else:
      raise ValueError(
        f"Unsupported policy format '.{suffix}' at {policy_path}. "
        "Expected '.onnx' (ONNX Runtime) or '.pt' (TorchScript JIT)."
      )

    # 启动随机命令采样器
    cmd_sampler = RandomCommandSampler()
    cmd_sampler.start()

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
                    simulation_time = time.time() - start
                    cmd_sampler.update(simulation_time)
                    cmd[:] = cmd_sampler.get_command()

                    if counter % 400 == 0:
                        print(
                            f"cmd: [{cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}]"
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

                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)

                    if suffix == "onnx":
                        action = policy.run([output_name], {input_name: obs_tensor.numpy()})[0]
                        action = np.asarray(action).reshape(-1)
                    else:  # "pt" → TorchScript
                        action = policy(obs_tensor).detach().numpy().squeeze()

                    target_dof_pos = default_angles + action * action_scales

                viewer.sync()

                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)