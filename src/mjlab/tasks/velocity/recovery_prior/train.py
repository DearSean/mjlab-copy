"""Native PyTorch training loop for the recovery motion VAE."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from mjlab.tasks.velocity.recovery_prior.data import (
  KinematicNormalizer,
  MotionWindowDataset,
  RecoverySequenceDataset,
  collate_kinematic_sequences,
)
from mjlab.tasks.velocity.recovery_prior.vae import (
  RecoveryMotionVae,
  RecoveryVaeCfg,
  RecoveryVaeLoss,
)


@dataclass(frozen=True, kw_only=True)
class RecoveryVaeTrainCfg:
  """Reproducible pretraining or recovery-finetuning configuration."""

  dataset_dir: Path = Path("artifacts/recovery/dataset")
  output_dir: Path = Path("artifacts/recovery/models/vae")
  stage: Literal["pretrain", "recovery"] = "pretrain"
  epochs: int = 100
  batch_size: int = 12
  gradient_accumulation: int = 4
  learning_rate: float = 3e-4
  weight_decay: float = 1e-4
  max_gradient_norm: float = 1.0
  samples_per_epoch: int = 4096
  validation_samples: int = 512
  min_window_frames: int = 40
  max_window_frames: int = 160
  latent_dim: int = 256
  num_latent_tokens: int = 1
  model_dim: int = 256
  num_heads: int = 4
  num_layers: int = 4
  feedforward_dim: int = 1024
  dropout: float = 0.1
  beta_max: float = 1e-4
  kl_warmup_steps: int = 2000
  seed: int = 42
  num_workers: int = 0
  device: str = "cuda"
  mixed_precision: bool = True
  max_steps: int | None = None
  resume: Path | None = None
  init_checkpoint: Path | None = None

  def __post_init__(self) -> None:
    positive = (
      self.epochs,
      self.batch_size,
      self.gradient_accumulation,
      self.samples_per_epoch,
      self.validation_samples,
      self.min_window_frames,
      self.max_window_frames,
      self.kl_warmup_steps,
      self.num_latent_tokens,
    )
    if any(value <= 0 for value in positive):
      raise ValueError(
        "Epoch, batch, sample, window, and warmup values must be positive."
      )
    if self.min_window_frames > self.max_window_frames:
      raise ValueError("min_window_frames must not exceed max_window_frames.")
    if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
      raise ValueError("Optimizer configuration is invalid.")
    if self.max_gradient_norm <= 0.0 or self.beta_max < 0.0:
      raise ValueError("Gradient norm and beta configuration are invalid.")
    if self.max_steps is not None and self.max_steps <= 0:
      raise ValueError("max_steps must be positive when set.")
    if self.num_workers < 0:
      raise ValueError("num_workers must be non-negative.")
    if self.resume is not None and self.init_checkpoint is not None:
      raise ValueError("resume and init_checkpoint are mutually exclusive.")


def train_recovery_vae(cfg: RecoveryVaeTrainCfg) -> dict[str, Any]:
  """Run training and return the best validation summary."""
  _seed_everything(cfg.seed)
  device = torch.device(cfg.device)
  if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA training requested but CUDA is unavailable.")
  if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats(device)
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  normalizer = KinematicNormalizer.from_dataset_dir(cfg.dataset_dir)
  train_dataset, validation_dataset = _make_datasets(cfg, normalizer)
  train_loader = _make_loader(train_dataset, cfg, train=True)
  validation_loader = _make_loader(validation_dataset, cfg, train=False)

  model_cfg = RecoveryVaeCfg(
    latent_dim=cfg.latent_dim,
    num_latent_tokens=cfg.num_latent_tokens,
    model_dim=cfg.model_dim,
    num_heads=cfg.num_heads,
    num_layers=cfg.num_layers,
    feedforward_dim=cfg.feedforward_dim,
    dropout=cfg.dropout,
    max_sequence_length=cfg.max_window_frames,
  )
  model = RecoveryMotionVae(model_cfg).to(device)
  loss_fn = RecoveryVaeLoss(normalizer).to(device)
  optimizer = AdamW(
    model.parameters(),
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
  )
  scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
  amp_enabled = cfg.mixed_precision and device.type == "cuda"
  scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
  start_epoch = 0
  global_step = 0
  best_validation = float("inf")
  if cfg.init_checkpoint is not None:
    checkpoint = torch.load(cfg.init_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
  if cfg.resume is not None:
    checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=False)
    _validate_checkpoint_dataset(checkpoint, cfg.dataset_dir)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = int(checkpoint["epoch"]) + 1
    global_step = int(checkpoint["global_step"])
    best_validation = float(checkpoint["best_validation_loss"])
    _restore_rng_state(checkpoint["rng_state"])

  parameter_count = sum(parameter.numel() for parameter in model.parameters())
  print(
    f"Recovery VAE: stage={cfg.stage}, parameters={parameter_count:,}, "
    f"device={device}, amp={amp_enabled}, train={len(train_dataset)}, "
    f"validation={len(validation_dataset)}"
  )
  _write_json_atomic(
    cfg.output_dir / "config.json",
    {
      "train": _jsonable(asdict(cfg)),
      "model": asdict(model_cfg),
      "dataset_hashes": _dataset_hashes(cfg.dataset_dir),
      "git_commit": _git_commit(),
    },
  )

  best_summary: dict[str, Any] = {}
  stop = False
  for epoch in range(start_epoch, cfg.epochs):
    if isinstance(train_dataset, MotionWindowDataset):
      train_dataset.set_epoch(epoch)
    train_metrics, global_step, stop = _train_epoch(
      model,
      loss_fn,
      train_loader,
      optimizer,
      scaler,
      device,
      cfg,
      global_step,
    )
    validation_metrics = _evaluate(
      model,
      loss_fn,
      validation_loader,
      device,
      beta=_kl_beta(global_step, cfg),
      amp_enabled=amp_enabled,
    )
    scheduler.step()
    summary = {
      "epoch": epoch,
      "global_step": global_step,
      "learning_rate": optimizer.param_groups[0]["lr"],
      "train": train_metrics,
      "validation": validation_metrics,
    }
    if device.type == "cuda":
      summary["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    print(json.dumps(summary, sort_keys=True))
    _append_json_line(cfg.output_dir / "metrics.jsonl", summary)
    improved = validation_metrics["loss"] < best_validation
    if improved:
      best_validation = validation_metrics["loss"]
      best_summary = summary
    checkpoint = _checkpoint(
      model=model,
      optimizer=optimizer,
      scheduler=scheduler,
      scaler=scaler,
      cfg=cfg,
      model_cfg=model_cfg,
      epoch=epoch,
      global_step=global_step,
      best_validation=best_validation,
    )
    _save_checkpoint_atomic(cfg.output_dir / "latest.pt", checkpoint)
    if improved:
      _save_checkpoint_atomic(cfg.output_dir / "best.pt", checkpoint)
    if stop:
      break
  return best_summary


def _make_datasets(
  cfg: RecoveryVaeTrainCfg,
  normalizer: KinematicNormalizer,
) -> tuple[
  MotionWindowDataset | RecoverySequenceDataset,
  MotionWindowDataset | RecoverySequenceDataset,
]:
  if cfg.stage == "recovery":
    return (
      RecoverySequenceDataset(cfg.dataset_dir, split="train", normalizer=normalizer),
      RecoverySequenceDataset(
        cfg.dataset_dir, split="validation", normalizer=normalizer
      ),
    )
  return (
    MotionWindowDataset(
      cfg.dataset_dir,
      split="train",
      samples_per_epoch=cfg.samples_per_epoch,
      min_frames=cfg.min_window_frames,
      max_frames=cfg.max_window_frames,
      seed=cfg.seed,
      random_length=True,
      normalizer=normalizer,
    ),
    MotionWindowDataset(
      cfg.dataset_dir,
      split="validation",
      samples_per_epoch=cfg.validation_samples,
      min_frames=cfg.min_window_frames,
      max_frames=cfg.max_window_frames,
      seed=cfg.seed + 1,
      random_length=False,
      normalizer=normalizer,
    ),
  )


def _make_loader(
  dataset: MotionWindowDataset | RecoverySequenceDataset,
  cfg: RecoveryVaeTrainCfg,
  *,
  train: bool,
) -> DataLoader[dict[str, Any]]:
  generator = torch.Generator().manual_seed(cfg.seed + (0 if train else 1))
  return DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=train and isinstance(dataset, RecoverySequenceDataset),
    num_workers=cfg.num_workers,
    pin_memory=cfg.device.startswith("cuda"),
    persistent_workers=cfg.num_workers > 0,
    collate_fn=collate_kinematic_sequences,
    generator=generator,
  )


def _train_epoch(
  model: RecoveryMotionVae,
  loss_fn: RecoveryVaeLoss,
  loader: DataLoader[dict[str, Any]],
  optimizer: AdamW,
  scaler: torch.amp.GradScaler,
  device: torch.device,
  cfg: RecoveryVaeTrainCfg,
  global_step: int,
) -> tuple[dict[str, float], int, bool]:
  model.train()
  optimizer.zero_grad(set_to_none=True)
  accumulator: dict[str, float] = {}
  batch_count = 0
  stop = False
  amp_enabled = cfg.mixed_precision and device.type == "cuda"
  for batch_index, cpu_batch in enumerate(loader):
    batch = cpu_batch.to(device, non_blocking=True)
    beta = _kl_beta(global_step, cfg)
    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
      prediction, mean, logvar = model(
        batch.motion,
        batch.valid_mask,
        sample=True,
      )
      loss, metrics = loss_fn(
        prediction,
        batch.motion,
        batch,
        kl_mean=mean,
        kl_logvar=logvar,
        beta=beta,
      )
      scaled_loss = loss / cfg.gradient_accumulation
    scaler.scale(scaled_loss).backward()
    update = (batch_index + 1) % cfg.gradient_accumulation == 0 or (
      batch_index + 1 == len(loader)
    )
    if update:
      scaler.unscale_(optimizer)
      clip_grad_norm_(model.parameters(), cfg.max_gradient_norm)
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad(set_to_none=True)
      global_step += 1
      if cfg.max_steps is not None and global_step >= cfg.max_steps:
        stop = True
    _accumulate(accumulator, metrics)
    batch_count += 1
    if stop:
      break
  return _averages(accumulator, batch_count), global_step, stop


@torch.no_grad()
def _evaluate(
  model: RecoveryMotionVae,
  loss_fn: RecoveryVaeLoss,
  loader: DataLoader[dict[str, Any]],
  device: torch.device,
  *,
  beta: float,
  amp_enabled: bool,
) -> dict[str, float]:
  model.eval()
  accumulator: dict[str, float] = {}
  batch_count = 0
  for cpu_batch in loader:
    batch = cpu_batch.to(device, non_blocking=True)
    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
      prediction, mean, logvar = model(
        batch.motion,
        batch.valid_mask,
        sample=False,
      )
      _, metrics = loss_fn(
        prediction,
        batch.motion,
        batch,
        kl_mean=mean,
        kl_logvar=logvar,
        beta=beta,
      )
    _accumulate(accumulator, metrics)
    batch_count += 1
  return _averages(accumulator, batch_count)


def _kl_beta(global_step: int, cfg: RecoveryVaeTrainCfg) -> float:
  return cfg.beta_max * min(1.0, global_step / cfg.kl_warmup_steps)


def _accumulate(target: dict[str, float], metrics: dict[str, torch.Tensor]) -> None:
  for name, value in metrics.items():
    target[name] = target.get(name, 0.0) + float(value.detach().cpu())


def _averages(total: dict[str, float], count: int) -> dict[str, float]:
  if count <= 0:
    raise ValueError("A training or validation loader produced no batches.")
  return {name: value / count for name, value in sorted(total.items())}


def _checkpoint(
  *,
  model: RecoveryMotionVae,
  optimizer: AdamW,
  scheduler: CosineAnnealingLR,
  scaler: torch.amp.GradScaler,
  cfg: RecoveryVaeTrainCfg,
  model_cfg: RecoveryVaeCfg,
  epoch: int,
  global_step: int,
  best_validation: float,
) -> dict[str, Any]:
  return {
    "schema_version": "recovery-vae-checkpoint-v1",
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict(),
    "train_cfg": asdict(cfg),
    "model_cfg": asdict(model_cfg),
    "epoch": epoch,
    "global_step": global_step,
    "best_validation_loss": best_validation,
    "dataset_hashes": _dataset_hashes(cfg.dataset_dir),
    "git_commit": _git_commit(),
    "rng_state": _rng_state(),
  }


def _save_checkpoint_atomic(path: Path, checkpoint: dict[str, Any]) -> None:
  descriptor, temporary_name = tempfile.mkstemp(
    dir=path.parent, prefix=f".{path.name}."
  )
  os.close(descriptor)
  temporary = Path(temporary_name)
  try:
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def _dataset_hashes(dataset_dir: Path) -> dict[str, str]:
  return {
    name: _sha256(dataset_dir / name)
    for name in ("dataset_manifest.json", "normalizer.npz")
  }


def _validate_checkpoint_dataset(checkpoint: dict[str, Any], dataset_dir: Path) -> None:
  if checkpoint.get("dataset_hashes") != _dataset_hashes(dataset_dir):
    raise ValueError("Checkpoint belongs to a different dataset or normalizer.")


def _rng_state() -> dict[str, Any]:
  return {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch": torch.get_rng_state(),
    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
  }


def _restore_rng_state(state: dict[str, Any]) -> None:
  random.setstate(state["python"])
  np.random.set_state(state["numpy"])
  torch.set_rng_state(state["torch"])
  if torch.cuda.is_available() and state["cuda"]:
    torch.cuda.set_rng_state_all(state["cuda"])


def _seed_everything(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
  descriptor, temporary_name = tempfile.mkstemp(
    dir=path.parent, prefix=f".{path.name}."
  )
  temporary = Path(temporary_name)
  try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
      json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
      stream.write("\n")
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def _append_json_line(path: Path, value: dict[str, Any]) -> None:
  with path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(item) for item in value]
  return value


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while block := stream.read(1024 * 1024):
      digest.update(block)
  return digest.hexdigest()


def _git_commit() -> str | None:
  result = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    check=False,
    capture_output=True,
    text=True,
  )
  return result.stdout.strip() if result.returncode == 0 else None
