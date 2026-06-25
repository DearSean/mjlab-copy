"""Event terms for tracking tasks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.string import resolve_expr

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Hamilton product of two quaternions (w, x, y, z).

  Supports broadcasting: both inputs can be (4,) or (N, 4).
  """
  if q1.dim() == 1 and q2.dim() == 2:
    q1 = q1.unsqueeze(0).expand(q2.shape[0], -1)
  if q2.dim() == 1 and q1.dim() == 2:
    q2 = q2.unsqueeze(0).expand(q1.shape[0], -1)

  w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
  w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

  return torch.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dim=-1,
  )


def randomize_initial_pose(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  poses: list[dict[str, tuple[float, ...]]],
  probability: float,
  min_step_count: int,
  z_rotation_range: tuple[float, float] = (0.0, 2.0 * math.pi),
) -> None:
  """Randomly override the initial pose for a subset of environments.

  After ``min_step_count`` environment steps, with probability ``probability``
  for each environment in ``env_ids``, sample one of the provided ``poses`` and
  apply it to the robot. A random rotation around the world z-axis is composed
  on top of the sampled pose.

  Args:
      env: The RL environment.
      env_ids: Environments being reset. If None, all environments are considered.
      asset_cfg: Scene entity configuration for the robot.
      poses: List of pose dictionaries. Each dictionary must contain:
          - ``"pos"``: tuple of (x, y, z) root position.
          - ``"quat"``: tuple of (w, x, y, z) root quaternion.
          - ``"joint_pos"`` (optional): dict mapping joint name patterns to
            joint angles, e.g. ``{"left_hip_pitch_joint": -0.2, ".*_knee_pitch_joint": 0.4}``.
      probability: Probability in [0, 1] of applying the randomization to each env.
      min_step_count: Only apply after this many environment steps have elapsed.
      z_rotation_range: Range (min, max) in radians for the random yaw rotation.
  """
  if env.common_step_counter < min_step_count or probability <= 0.0 or not poses:
    return

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  # Select environments that will receive a randomized pose.
  mask = torch.rand(len(env_ids), device=env.device) < probability
  selected = env_ids[mask]
  n = len(selected)
  if n == 0:
    return

  asset = env.scene[asset_cfg.name]
  device = env.device

  # Precompute pose tensors.
  root_pos = torch.tensor(
    [pose["pos"] for pose in poses], device=device, dtype=torch.float32
  )
  root_quat = torch.tensor(
    [pose["quat"] for pose in poses], device=device, dtype=torch.float32
  )
  joint_pos_list = []
  for pose in poses:
    joint_values = resolve_expr(
      pose.get("joint_pos", {".*": 0.0}), asset.joint_names, 0.0
    )
    joint_pos_list.append(joint_values)
  joint_pos = torch.tensor(joint_pos_list, device=device, dtype=torch.float32)

  # Sample pose index for each selected environment.
  pose_indices = torch.randint(len(poses), (n,), device=device)

  selected_root_pos = root_pos[pose_indices]
  selected_root_quat = root_quat[pose_indices]
  selected_joint_pos = joint_pos[pose_indices]

  # Add environment origin offsets so the robot is placed at the correct world
  # location for each parallel environment.
  selected_root_pos += env.scene.env_origins[selected]

  # Add random yaw rotation around world z-axis.
  angles = torch.rand(n, device=device) * (z_rotation_range[1] - z_rotation_range[0])
  angles += z_rotation_range[0]
  half_angles = angles * 0.5
  z_quats = torch.stack(
    [
      torch.cos(half_angles),
      torch.zeros_like(angles),
      torch.zeros_like(angles),
      torch.sin(half_angles),
    ],
    dim=-1,
  )
  selected_root_quat = _quat_mul(z_quats, selected_root_quat)

  # Assemble full root state: pos(3), quat(4), lin_vel(3), ang_vel(3).
  root_state = torch.zeros(n, 13, device=device, dtype=torch.float32)
  root_state[:, 0:3] = selected_root_pos
  root_state[:, 3:7] = selected_root_quat
  # Linear and angular velocities remain zero.

  asset.write_root_state_to_sim(root_state, env_ids=selected)


def randomize_initial_root_pose(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  poses: list[dict[str, tuple[float, ...]]],
  probability: float,
  min_step_count: int,
  z_rotation_range: tuple[float, float] = (0.0, 2.0 * math.pi),
) -> None:
  """Randomize only the root pose for selected envs on reset.

  This is like :func:`randomize_initial_pose` but preserves the current joint
  configuration instead of overriding joint angles.
  """
  if env.common_step_counter < min_step_count or probability <= 0.0 or not poses:
    return

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  mask = torch.rand(len(env_ids), device=env.device) < probability
  selected = env_ids[mask]
  n = len(selected)
  if n == 0:
    return

  asset = env.scene[asset_cfg.name]
  device = env.device

  root_pos = torch.tensor(
    [pose["pos"] for pose in poses], device=device, dtype=torch.float32
  )
  root_quat = torch.tensor(
    [pose["quat"] for pose in poses], device=device, dtype=torch.float32
  )

  pose_indices = torch.randint(len(poses), (n,), device=device)

  selected_root_pos = root_pos[pose_indices]
  selected_root_quat = root_quat[pose_indices]

  selected_root_pos += env.scene.env_origins[selected]

  angles = torch.rand(n, device=device) * (z_rotation_range[1] - z_rotation_range[0])
  angles += z_rotation_range[0]
  half_angles = angles * 0.5
  z_quats = torch.stack(
    [
      torch.cos(half_angles),
      torch.zeros_like(angles),
      torch.zeros_like(angles),
      torch.sin(half_angles),
    ],
    dim=-1,
  )
  selected_root_quat = _quat_mul(z_quats, selected_root_quat)

  root_state = torch.zeros(n, 13, device=device, dtype=torch.float32)
  root_state[:, 0:3] = selected_root_pos
  root_state[:, 3:7] = selected_root_quat

  asset.write_root_state_to_sim(root_state, env_ids=selected)
