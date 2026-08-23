from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mjlab.tasks.velocity.recovery_data.dataset import (
  RECOVERY_DATASET_SCHEMA_VERSION,
)
from mjlab.tasks.velocity.recovery_data.kinematic import (
  recovery_kinematic_feature_names,
)
from mjlab.tasks.velocity.recovery_data.schema import RECOVERY_KINEMATIC_DIM
from mjlab.tasks.velocity.recovery_prior.data import (
  KinematicNormalizer,
  MotionWindowDataset,
  RecoverySequenceDataset,
  collate_kinematic_sequences,
)
from mjlab.tasks.velocity.recovery_prior.train import (
  RecoveryVaeTrainCfg,
  train_recovery_vae,
)
from mjlab.tasks.velocity.recovery_prior.vae import (
  RecoveryMotionVae,
  RecoveryVaeCfg,
  RecoveryVaeLoss,
)


def test_motion_windows_are_deterministic_and_never_cross_clips(tmp_path: Path):
  root = _write_dataset(tmp_path)
  dataset = MotionWindowDataset(
    root,
    split="train",
    samples_per_epoch=1000,
    min_frames=4,
    max_frames=12,
    seed=7,
  )

  first = dataset[25]
  again = dataset[25]
  torch.testing.assert_close(first["motion"], again["motion"])
  assert first["metadata"] == again["metadata"]
  for index in range(len(dataset)):
    metadata = dataset[index]["metadata"]
    assert 0 <= metadata["start"] < metadata["end"] <= metadata["clip_length"]


def test_recovery_sequences_collate_with_explicit_padding_mask(tmp_path: Path):
  root = _write_dataset(tmp_path)
  dataset = RecoverySequenceDataset(root, split="train")

  batch = collate_kinematic_sequences([dataset[0], dataset[1]])

  assert batch.motion.shape == (2, 9, RECOVERY_KINEMATIC_DIM)
  assert batch.lengths.tolist() == [5, 9]
  assert batch.valid_mask[0, :5].all()
  assert not batch.valid_mask[0, 5:].any()
  assert torch.count_nonzero(batch.motion[0, 5:]) == 0
  assert batch.complete_recovery.tolist() == [True, False]
  assert batch.metadata[0]["support_complete"] == 3


def test_vae_forward_and_loss_ignore_padding(tmp_path: Path):
  root = _write_dataset(tmp_path)
  normalizer = KinematicNormalizer.from_dataset_dir(root)
  dataset = RecoverySequenceDataset(root, split="train", normalizer=normalizer)
  batch = collate_kinematic_sequences([dataset[0], dataset[1]])
  model = RecoveryMotionVae(
    RecoveryVaeCfg(
      latent_dim=16,
      num_latent_tokens=4,
      model_dim=32,
      num_heads=4,
      num_layers=1,
      feedforward_dim=64,
      dropout=0.0,
      max_sequence_length=12,
    )
  )
  prediction, mean, logvar = model(batch.motion, batch.valid_mask, sample=False)
  loss_fn = RecoveryVaeLoss(normalizer)

  loss, metrics = loss_fn(
    prediction,
    batch.motion,
    batch,
    kl_mean=mean,
    kl_logvar=logvar,
    beta=1e-4,
  )
  changed_padding = prediction.clone()
  changed_padding[~batch.valid_mask] = 1e6
  changed_loss, _ = loss_fn(
    changed_padding,
    batch.motion,
    batch,
    kl_mean=mean,
    kl_logvar=logvar,
    beta=1e-4,
  )

  assert prediction.shape == batch.motion.shape
  assert mean.shape == (2, 4, 16)
  assert logvar.shape == (2, 4, 16)
  assert torch.count_nonzero(prediction[~batch.valid_mask]) == 0
  assert torch.isfinite(loss)
  torch.testing.assert_close(loss, changed_loss)
  assert all(torch.isfinite(value) for value in metrics.values())


def test_training_smoke_writes_resumable_checkpoint(tmp_path: Path):
  root = _write_dataset(tmp_path)
  output = tmp_path / "model"

  summary = train_recovery_vae(
    RecoveryVaeTrainCfg(
      dataset_dir=root,
      output_dir=output,
      epochs=1,
      batch_size=2,
      gradient_accumulation=1,
      samples_per_epoch=4,
      validation_samples=2,
      min_window_frames=4,
      max_window_frames=8,
      latent_dim=8,
      num_latent_tokens=4,
      model_dim=16,
      num_heads=4,
      num_layers=1,
      feedforward_dim=32,
      dropout=0.0,
      kl_warmup_steps=1,
      device="cpu",
      mixed_precision=False,
      max_steps=1,
    )
  )

  assert (output / "latest.pt").is_file()
  assert (output / "best.pt").is_file()
  assert (output / "config.json").is_file()
  assert (output / "metrics.jsonl").is_file()
  checkpoint = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
  assert checkpoint["schema_version"] == "recovery-vae-checkpoint-v1"
  assert checkpoint["global_step"] == 1
  assert checkpoint["model_cfg"]["num_latent_tokens"] == 4
  assert summary["validation"]["loss"] >= 0.0


def _write_dataset(tmp_path: Path) -> Path:
  root = tmp_path / "dataset"
  root.mkdir()
  names = recovery_kinematic_feature_names()
  manifest = {
    "schema_version": RECOVERY_DATASET_SCHEMA_VERSION,
    "feature_dim": RECOVERY_KINEMATIC_DIM,
    "feature_names": list(names),
  }
  (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
  mean = np.zeros(RECOVERY_KINEMATIC_DIM, dtype=np.float32)
  std = np.ones(RECOVERY_KINEMATIC_DIM, dtype=np.float32)
  np.savez(
    root / "normalizer.npz",
    mean=mean,
    std=std,
    feature_names=np.asarray(names),
    frame_count=np.asarray([90]),
  )
  rng = np.random.default_rng(3)
  motion = rng.normal(size=(90, RECOVERY_KINEMATIC_DIM)).astype(np.float32)
  recovery = rng.normal(size=(14, RECOVERY_KINEMATIC_DIM)).astype(np.float32)
  arrays = {
    "motion_features": motion,
    "motion_offsets": np.asarray([0, 20, 50, 90], dtype=np.int64),
    "motion_clip_ids": np.asarray(["a", "b", "c"]),
    "motion_source_fps": np.full(3, 30.0),
    "motion_nominal_heights_m": np.asarray([1.7, 1.8, 1.9], dtype=np.float32),
    "recovery_features": recovery,
    "recovery_offsets": np.asarray([0, 5, 14], dtype=np.int64),
    "recovery_candidate_ids": np.asarray(["first", "second"]),
    "recovery_source_paths": np.asarray(["a.bvh", "b.bvh"]),
    "recovery_source_start_frames": np.asarray([0, 10]),
    "recovery_source_end_frames": np.asarray([5, 19]),
    "recovery_support_start_frames": np.asarray([0, 10]),
    "recovery_support_complete_frames": np.asarray([3, 17]),
    "recovery_locomotion_takeover_frames": np.asarray([-1, -1]),
    "recovery_support_start_indices": np.asarray([0, 0]),
    "recovery_support_complete_indices": np.asarray([3, 7]),
    "recovery_locomotion_takeover_indices": np.asarray([-1, -1]),
    "recovery_initial_postures": np.asarray(["supine", "prone"]),
    "recovery_terminal_modes": np.asarray(["stationary", "support_only"]),
    "recovery_outcomes": np.asarray(["success", "partial"]),
    "recovery_complete": np.asarray([True, False]),
    "recovery_nominal_heights_m": np.asarray([1.7, 1.8], dtype=np.float32),
  }
  for split in ("train", "validation", "test"):
    np.savez(root / f"{split}.npz", **arrays)
  return root
