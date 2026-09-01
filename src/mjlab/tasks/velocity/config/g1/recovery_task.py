"""Tensor-only standing criterion and rewards shared by the G1 recovery task."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, kw_only=True)
class G1StandingCfg:
  nominal_height_m: float = 0.76
  min_upright: float = 0.90
  max_linear_velocity_mps: float = 0.20
  max_angular_velocity_radps: float = 0.40
  max_joint_velocity_rms_radps: float = 0.50
  max_normalized_pose_rms: float = 1.0
  hold_steps: int = 25


class G1StandingTracker:
  """Track the 0.5-second standing hold requirement independently per env."""

  def __init__(self, num_envs: int, device: torch.device, cfg: G1StandingCfg) -> None:
    self.cfg = cfg
    self.hold_count = torch.zeros(num_envs, dtype=torch.long, device=device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.hold_count[env_ids] = 0

  def update(
    self,
    *,
    pelvis_height_m: torch.Tensor,
    projected_gravity: torch.Tensor,
    base_linear_velocity: torch.Tensor,
    base_angular_velocity: torch.Tensor,
    joint_velocity: torch.Tensor,
    normalized_pose_error: torch.Tensor,
    left_foot_contact: torch.Tensor,
    right_foot_contact: torch.Tensor,
    nonfoot_support: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    upright = torch.clamp(-projected_gravity[:, 2], 0.0, 1.0)
    standing = (
      (pelvis_height_m >= 0.85 * self.cfg.nominal_height_m)
      & (upright >= self.cfg.min_upright)
      & (
        torch.linalg.vector_norm(base_linear_velocity, dim=1)
        <= self.cfg.max_linear_velocity_mps
      )
      & (
        torch.linalg.vector_norm(base_angular_velocity, dim=1)
        <= self.cfg.max_angular_velocity_radps
      )
      & (
        torch.sqrt(torch.mean(torch.square(joint_velocity), dim=1))
        <= self.cfg.max_joint_velocity_rms_radps
      )
      & (
        torch.sqrt(torch.mean(torch.square(normalized_pose_error), dim=1))
        <= self.cfg.max_normalized_pose_rms
      )
      & left_foot_contact
      & right_foot_contact
      & ~nonfoot_support
    )
    self.hold_count = torch.where(
      standing, self.hold_count + 1, torch.zeros_like(self.hold_count)
    )
    return standing, self.hold_count >= self.cfg.hold_steps


def standing_potential(
  pelvis_height_m: torch.Tensor,
  projected_gravity: torch.Tensor,
  normalized_pose_error: torch.Tensor,
  cfg: G1StandingCfg,
) -> torch.Tensor:
  """Bounded standing potential for fallen, reference, and stand resets."""
  height = torch.clamp(pelvis_height_m / cfg.nominal_height_m, 0.0, 1.0)
  upright = torch.clamp(-projected_gravity[:, 2], 0.0, 1.0)
  pose = torch.exp(-torch.mean(torch.square(normalized_pose_error), dim=1))
  return (height + upright + pose) / 3.0


def potential_progress(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
  """Reward only net standing progress, not absolute height every step."""
  return current - previous
