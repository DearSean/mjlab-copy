"""Robot-independent posture phase estimation for velocity tasks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@dataclass(frozen=True, kw_only=True)
class PosturePhaseEstimatorCfg:
  """Dimensionless posture phase boundaries shared across robot morphologies.

  The estimator derives the robot's characteristic height from its configured
  default root pose. Consequently these boundaries describe confidence, not
  robot-specific distances or angles.
  """

  recovery_confidence: float = 0.60
  ready_confidence: float = 0.85

  def __post_init__(self) -> None:
    if not 0.0 < self.recovery_confidence < self.ready_confidence <= 1.0:
      raise ValueError(
        "Posture confidence boundaries must satisfy "
        "0 < recovery_confidence < ready_confidence <= 1."
      )


@dataclass(frozen=True)
class PosturePhaseState:
  """Continuous posture features and fuzzy phase memberships."""

  height: torch.Tensor
  uprightness: torch.Tensor
  progress: torch.Tensor
  motion_stability: torch.Tensor
  fallen: torch.Tensor
  recovering: torch.Tensor
  upright_unstable: torch.Tensor
  ready: torch.Tensor
  walk_gate: torch.Tensor
  needs_recovery: torch.Tensor
  is_ready: torch.Tensor


class PosturePhaseEstimator:
  """Estimate morphology-normalized fallen-to-walking posture state.

  Height is normalized by the asset's default standing root height. Uprightness
  is the positive part of the body's world-up projection, so any roll or pitch
  tilt of 90 degrees or more has zero uprightness. Motion uses gravitational
  time scaling and only separates stable from unstable upright states; it does
  not gate locomotion, because walking is intentionally dynamic.
  """

  def __init__(self, cfg: PosturePhaseEstimatorCfg | None = None) -> None:
    self.cfg = cfg or PosturePhaseEstimatorCfg()

  def estimate(
    self,
    env: object,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> PosturePhaseState:
    scene = env.scene  # type: ignore[attr-defined]
    asset: Entity = scene[asset_cfg.name]
    root_height = asset.data.root_link_pos_w[:, 2]

    env_origins = getattr(scene, "env_origins", None)
    if env_origins is None:
      ground_height = torch.zeros_like(root_height)
    else:
      ground_height = env_origins[:, 2]

    default_root_state = asset.data.default_root_state
    nominal_height = default_root_state[:, 2].clamp_min(1e-6)
    height = ((root_height - ground_height) / nominal_height).clamp(0.0, 1.0)

    uprightness = (-asset.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
    progress = height * uprightness
    motion_stability = self._motion_stability(asset, nominal_height)

    # The degree-two Bernstein basis provides non-negative memberships that
    # sum to one. Split its upright component using motion stability.
    fallen = torch.square(1.0 - progress)
    recovering = 2.0 * progress * (1.0 - progress)
    upright = torch.square(progress)
    upright_unstable = upright * (1.0 - motion_stability)
    ready = upright * motion_stability

    walk_gate = _smoothstep(
      progress,
      self.cfg.recovery_confidence,
      self.cfg.ready_confidence,
    )
    return PosturePhaseState(
      height=height,
      uprightness=uprightness,
      progress=progress,
      motion_stability=motion_stability,
      fallen=fallen,
      recovering=recovering,
      upright_unstable=upright_unstable,
      ready=ready,
      walk_gate=walk_gate,
      needs_recovery=progress < self.cfg.recovery_confidence,
      is_ready=progress >= self.cfg.ready_confidence,
    )

  @staticmethod
  def _motion_stability(asset: Entity, nominal_height: torch.Tensor) -> torch.Tensor:
    lin_vel = getattr(asset.data, "root_link_lin_vel_w", None)
    ang_vel = getattr(asset.data, "root_link_ang_vel_b", None)
    gravity = getattr(asset.data, "gravity_vec_w", None)
    if lin_vel is None or ang_vel is None or gravity is None:
      return torch.ones_like(nominal_height)

    gravity_magnitude = torch.linalg.vector_norm(gravity).clamp_min(1e-6)
    velocity_scale = torch.sqrt(gravity_magnitude * nominal_height)
    time_scale = torch.sqrt(nominal_height / gravity_magnitude)
    vertical_speed = torch.abs(lin_vel[:, 2]) / velocity_scale
    tilt_rate = torch.linalg.vector_norm(ang_vel[:, :2], dim=1) * time_scale
    return torch.exp(-0.5 * (torch.square(vertical_speed) + torch.square(tilt_rate)))


def _smoothstep(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
  ratio = ((value - low) / (high - low)).clamp(0.0, 1.0)
  return ratio * ratio * (3.0 - 2.0 * ratio)
