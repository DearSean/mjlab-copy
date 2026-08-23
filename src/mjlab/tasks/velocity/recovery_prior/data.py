"""Deterministic variable-length datasets for recovery motion priors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from mjlab.tasks.velocity.recovery_data.dataset import (
  RECOVERY_DATASET_SCHEMA_VERSION,
)
from mjlab.tasks.velocity.recovery_data.schema import RECOVERY_KINEMATIC_DIM


@dataclass
class KinematicBatch:
  """One padded batch and the mask required by every model and loss."""

  motion: torch.Tensor
  valid_mask: torch.Tensor
  lengths: torch.Tensor
  nominal_height_m: torch.Tensor
  complete_recovery: torch.Tensor
  metadata: list[dict[str, Any]]

  def to(
    self, device: torch.device | str, *, non_blocking: bool = False
  ) -> KinematicBatch:
    return KinematicBatch(
      motion=self.motion.to(device, non_blocking=non_blocking),
      valid_mask=self.valid_mask.to(device, non_blocking=non_blocking),
      lengths=self.lengths.to(device, non_blocking=non_blocking),
      nominal_height_m=self.nominal_height_m.to(device, non_blocking=non_blocking),
      complete_recovery=self.complete_recovery.to(device, non_blocking=non_blocking),
      metadata=self.metadata,
    )


class KinematicNormalizer:
  """Immutable Train statistics shared by all splits and evaluation code."""

  def __init__(
    self,
    mean: NDArray[np.float32],
    std: NDArray[np.float32],
    feature_names: tuple[str, ...],
  ) -> None:
    if mean.shape != (RECOVERY_KINEMATIC_DIM,) or std.shape != (
      RECOVERY_KINEMATIC_DIM,
    ):
      raise ValueError("Normalizer mean/std do not match the 85D schema.")
    if len(feature_names) != RECOVERY_KINEMATIC_DIM:
      raise ValueError("Normalizer feature names do not match the 85D schema.")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
      raise ValueError("Normalizer contains NaN or Inf.")
    if np.any(std <= 0.0):
      raise ValueError("Normalizer standard deviations must be positive.")
    self.mean = mean.astype(np.float32, copy=True)
    self.std = std.astype(np.float32, copy=True)
    self.feature_names = feature_names

  @classmethod
  def from_dataset_dir(cls, dataset_dir: str | Path) -> KinematicNormalizer:
    root = Path(dataset_dir)
    manifest = _load_dataset_manifest(root)
    with np.load(root / "normalizer.npz", allow_pickle=False) as archive:
      mean = archive["mean"].astype(np.float32)
      std = archive["std"].astype(np.float32)
      feature_names = tuple(str(value) for value in archive["feature_names"])
    if list(feature_names) != manifest["feature_names"]:
      raise ValueError("Normalizer and dataset manifest feature schemas differ.")
    return cls(mean, std, feature_names)

  def normalize(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
    if value.shape[-1] != RECOVERY_KINEMATIC_DIM:
      raise ValueError("Motion does not match the 85D schema.")
    return ((value - self.mean) / self.std).astype(np.float32)

  def torch_statistics(
    self, device: torch.device | str | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return (
      torch.as_tensor(self.mean, device=device),
      torch.as_tensor(self.std, device=device),
    )


class MotionWindowDataset(Dataset[dict[str, Any]]):
  """Deterministic random windows that never cross a source-clip boundary."""

  def __init__(
    self,
    dataset_dir: str | Path,
    *,
    split: str,
    samples_per_epoch: int,
    min_frames: int = 40,
    max_frames: int = 160,
    seed: int = 42,
    random_length: bool = True,
    normalizer: KinematicNormalizer | None = None,
  ) -> None:
    if samples_per_epoch <= 0:
      raise ValueError("samples_per_epoch must be positive.")
    if not 1 <= min_frames <= max_frames:
      raise ValueError("Window lengths must satisfy 1 <= min <= max.")
    root = Path(dataset_dir)
    _load_dataset_manifest(root)
    arrays = _load_shard(root, split)
    self.features = arrays["motion_features"].astype(np.float32)
    self.offsets = arrays["motion_offsets"].astype(np.int64)
    self.clip_ids = tuple(str(value) for value in arrays["motion_clip_ids"])
    self.nominal_heights_m = arrays["motion_nominal_heights_m"].astype(np.float32)
    _validate_offsets(self.offsets, self.features.shape[0], len(self.clip_ids))
    if self.nominal_heights_m.shape != (len(self.clip_ids),):
      raise ValueError("Motion nominal heights do not match clip count.")
    lengths = np.diff(self.offsets)
    if np.any(lengths <= 0):
      raise ValueError("Every source clip must contain at least one frame.")
    self.clip_probabilities = lengths.astype(np.float64) / lengths.sum()
    self.normalizer = normalizer or KinematicNormalizer.from_dataset_dir(root)
    self.samples_per_epoch = samples_per_epoch
    self.min_frames = min_frames
    self.max_frames = max_frames
    self.seed = seed
    self.random_length = random_length
    self.epoch = 0

  def __len__(self) -> int:
    return self.samples_per_epoch

  def set_epoch(self, epoch: int) -> None:
    if epoch < 0:
      raise ValueError("epoch must be non-negative.")
    self.epoch = epoch

  def __getitem__(self, index: int) -> dict[str, Any]:
    if not 0 <= index < len(self):
      raise IndexError(index)
    rng = np.random.default_rng(np.random.SeedSequence((self.seed, self.epoch, index)))
    clip_index = int(rng.choice(len(self.clip_ids), p=self.clip_probabilities))
    clip_start = int(self.offsets[clip_index])
    clip_end = int(self.offsets[clip_index + 1])
    available = clip_end - clip_start
    maximum = min(self.max_frames, available)
    minimum = min(self.min_frames, maximum)
    length = (
      int(rng.integers(minimum, maximum + 1))
      if self.random_length and minimum < maximum
      else maximum
    )
    local_start = int(rng.integers(0, available - length + 1))
    start = clip_start + local_start
    end = start + length
    motion = self.normalizer.normalize(self.features[start:end])
    return {
      "motion": torch.from_numpy(motion.copy()),
      "nominal_height_m": float(self.nominal_heights_m[clip_index]),
      "complete_recovery": False,
      "metadata": {
        "kind": "motion",
        "clip_id": self.clip_ids[clip_index],
        "clip_index": clip_index,
        "start": local_start,
        "end": local_start + length,
        "clip_length": available,
      },
    }


class RecoverySequenceDataset(Dataset[dict[str, Any]]):
  """All accepted recovery intervals with human-reviewed provenance."""

  def __init__(
    self,
    dataset_dir: str | Path,
    *,
    split: str,
    normalizer: KinematicNormalizer | None = None,
  ) -> None:
    root = Path(dataset_dir)
    _load_dataset_manifest(root)
    arrays = _load_shard(root, split)
    self.features = arrays["recovery_features"].astype(np.float32)
    self.offsets = arrays["recovery_offsets"].astype(np.int64)
    self.candidate_ids = tuple(str(value) for value in arrays["recovery_candidate_ids"])
    self.source_paths = tuple(str(value) for value in arrays["recovery_source_paths"])
    self.initial_postures = tuple(
      str(value) for value in arrays["recovery_initial_postures"]
    )
    self.terminal_modes = tuple(
      str(value) for value in arrays["recovery_terminal_modes"]
    )
    self.outcomes = tuple(str(value) for value in arrays["recovery_outcomes"])
    self.complete = arrays["recovery_complete"].astype(np.bool_)
    self.nominal_heights_m = arrays["recovery_nominal_heights_m"].astype(np.float32)
    self.support_start = arrays["recovery_support_start_indices"].astype(np.int64)
    self.support_complete = arrays["recovery_support_complete_indices"].astype(np.int64)
    self.locomotion_takeover = arrays["recovery_locomotion_takeover_indices"].astype(
      np.int64
    )
    _validate_offsets(self.offsets, self.features.shape[0], len(self.candidate_ids))
    expected = (len(self.candidate_ids),)
    for name, value in (
      ("source paths", self.source_paths),
      ("initial postures", self.initial_postures),
      ("terminal modes", self.terminal_modes),
      ("outcomes", self.outcomes),
      ("complete flags", self.complete),
      ("nominal heights", self.nominal_heights_m),
      ("support starts", self.support_start),
      ("support completions", self.support_complete),
      ("locomotion takeovers", self.locomotion_takeover),
    ):
      if len(value) != expected[0]:
        raise ValueError(f"Recovery {name} do not match segment count.")
    self.normalizer = normalizer or KinematicNormalizer.from_dataset_dir(root)

  def __len__(self) -> int:
    return len(self.candidate_ids)

  def __getitem__(self, index: int) -> dict[str, Any]:
    if not 0 <= index < len(self):
      raise IndexError(index)
    start = int(self.offsets[index])
    end = int(self.offsets[index + 1])
    motion = self.normalizer.normalize(self.features[start:end])
    return {
      "motion": torch.from_numpy(motion.copy()),
      "nominal_height_m": float(self.nominal_heights_m[index]),
      "complete_recovery": bool(self.complete[index]),
      "metadata": {
        "kind": "recovery",
        "candidate_id": self.candidate_ids[index],
        "source_path": self.source_paths[index],
        "initial_posture": self.initial_postures[index],
        "terminal_mode": self.terminal_modes[index],
        "outcome": self.outcomes[index],
        "support_start": int(self.support_start[index]),
        "support_complete": int(self.support_complete[index]),
        "locomotion_takeover": int(self.locomotion_takeover[index]),
      },
    }


def collate_kinematic_sequences(samples: list[dict[str, Any]]) -> KinematicBatch:
  """Pad variable-length sequences and make padding impossible to score."""
  if not samples:
    raise ValueError("Cannot collate an empty batch.")
  lengths = torch.as_tensor(
    [int(sample["motion"].shape[0]) for sample in samples], dtype=torch.long
  )
  maximum = int(lengths.max())
  motion = torch.zeros(
    (len(samples), maximum, RECOVERY_KINEMATIC_DIM), dtype=torch.float32
  )
  valid_mask = torch.zeros((len(samples), maximum), dtype=torch.bool)
  for index, sample in enumerate(samples):
    value = sample["motion"]
    length = int(value.shape[0])
    if value.shape != (length, RECOVERY_KINEMATIC_DIM):
      raise ValueError("A batch sample does not match the 85D schema.")
    motion[index, :length] = value
    valid_mask[index, :length] = True
  return KinematicBatch(
    motion=motion,
    valid_mask=valid_mask,
    lengths=lengths,
    nominal_height_m=torch.as_tensor(
      [sample["nominal_height_m"] for sample in samples], dtype=torch.float32
    ),
    complete_recovery=torch.as_tensor(
      [sample["complete_recovery"] for sample in samples], dtype=torch.bool
    ),
    metadata=[sample["metadata"] for sample in samples],
  )


def _load_dataset_manifest(root: Path) -> dict[str, Any]:
  path = root / "dataset_manifest.json"
  if not path.is_file():
    raise FileNotFoundError(f"Dataset manifest does not exist: {path}")
  with path.open(encoding="utf-8") as stream:
    manifest = json.load(stream)
  if manifest.get("schema_version") != RECOVERY_DATASET_SCHEMA_VERSION:
    raise ValueError(f"Unsupported recovery dataset schema in {path}.")
  if int(manifest.get("feature_dim", -1)) != RECOVERY_KINEMATIC_DIM:
    raise ValueError("Dataset manifest does not describe 85D motion.")
  return manifest


def _load_shard(root: Path, split: str) -> dict[str, NDArray[Any]]:
  if split not in {"train", "validation", "test"}:
    raise ValueError(f"Unknown split {split!r}.")
  path = root / f"{split}.npz"
  if not path.is_file():
    raise FileNotFoundError(f"Dataset shard does not exist: {path}")
  with np.load(path, allow_pickle=False) as archive:
    return {name: archive[name].copy() for name in archive.files}


def _validate_offsets(offsets: NDArray[np.int64], frames: int, count: int) -> None:
  if offsets.shape != (count + 1,):
    raise ValueError("Sequence offsets do not match sequence count.")
  if offsets[0] != 0 or offsets[-1] != frames or np.any(np.diff(offsets) <= 0):
    raise ValueError("Sequence offsets are invalid or contain empty intervals.")
