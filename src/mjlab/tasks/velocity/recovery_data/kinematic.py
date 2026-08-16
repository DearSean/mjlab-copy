"""Contact-label-free recovery representation for learned motion priors."""

from __future__ import annotations

from mjlab.tasks.velocity.recovery_data.schema import (
  RECOVERY_KINEMATIC_DIM,
  RECOVERY_SEMANTIC_DIM,
  CanonicalMotionClip,
  KinematicMotion,
)
from mjlab.tasks.velocity.recovery_data.semantic import (
  CONTACT_NAMES,
  RecoverySemanticEncoder,
  RecoverySemanticEncoderCfg,
  recovery_semantic_feature_names,
)

_BINARY_CONTACT_START = 75
_BINARY_CONTACT_END = _BINARY_CONTACT_START + len(CONTACT_NAMES)


class RecoveryKinematicEncoder:
  """Encode motion without inferred binary human contact labels.

  The legacy semantic encoder remains the single implementation of geometry,
  normalization, and recovery progress. Only its threshold-derived binary
  contact columns are removed. Continuous ground clearances remain available
  so MLD can learn ground relationships directly from trajectories.
  """

  def __init__(self, cfg: RecoverySemanticEncoderCfg | None = None) -> None:
    self._semantic_encoder = RecoverySemanticEncoder(cfg)

  def encode(self, clip: CanonicalMotionClip) -> KinematicMotion:
    semantic = self._semantic_encoder.encode(clip)
    keep = (
      *range(_BINARY_CONTACT_START),
      *range(_BINARY_CONTACT_END, RECOVERY_SEMANTIC_DIM),
    )
    features = semantic.features[:, keep]
    names = tuple(semantic.feature_names[index] for index in keep)
    if features.shape[1] != RECOVERY_KINEMATIC_DIM:
      raise RuntimeError(
        f"Kinematic encoder produced {features.shape[1]} features, "
        f"expected {RECOVERY_KINEMATIC_DIM}."
      )
    if any(name.endswith("_contact") for name in names):
      raise RuntimeError("Binary contact labels leaked into kinematic features.")
    return KinematicMotion(
      features=features.copy(),
      floor_height_m=semantic.floor_height_m,
      nominal_height_m=semantic.nominal_height_m,
      feature_names=names,
    )


def recovery_kinematic_feature_names() -> tuple[str, ...]:
  """Return the stable ordered feature names without loading a motion clip."""
  legacy = recovery_semantic_feature_names()
  return legacy[:_BINARY_CONTACT_START] + legacy[_BINARY_CONTACT_END:]
