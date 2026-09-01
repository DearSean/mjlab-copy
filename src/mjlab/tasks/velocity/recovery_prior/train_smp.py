"""Offline DDPM training for the 51-D G1 score-matching motion prior."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mjlab.tasks.velocity.recovery_prior.smp_model import SmpDenoiser, SmpDenoiserCfg


@dataclass(frozen=True, kw_only=True)
class G1SmpTrainCfg:
  feature_file: Path = Path("artifacts/g1_recovery/smp_features.npz")
  normalizer_file: Path = Path("artifacts/g1_recovery/smp_normalizer.npz")
  output_dir: Path = Path("artifacts/g1_recovery/smp")
  epochs: int = 300
  batch_size: int = 128
  learning_rate: float = 3e-4
  ema_decay: float = 0.999
  seed: int = 0
  device: str = "cuda"
  mixed_precision: bool = True
  model_dim: int = 128
  num_heads: int = 4
  num_layers: int = 2
  feedforward_dim: int = 512


class _WindowDataset(Dataset[torch.Tensor]):
  def __init__(self, feature_file: Path, normalizer_file: Path, split: str) -> None:
    arrays = np.load(feature_file)
    normalizer = np.load(normalizer_file)
    features = (arrays["features"] - normalizer["mean"]) / normalizer["std"]
    self.windows: list[np.ndarray] = []
    for begin, end, item_split in zip(
      arrays["offsets"][:-1], arrays["offsets"][1:], arrays["splits"], strict=True
    ):
      if str(item_split) != split:
        continue
      for start in range(int(begin), int(end) - 10 + 1):
        self.windows.append(features[start : start + 10].astype(np.float32))
    if not self.windows:
      raise ValueError(f"No legal 10-step windows for split {split!r}.")

  def __len__(self) -> int:
    return len(self.windows)

  def __getitem__(self, index: int) -> torch.Tensor:
    return torch.from_numpy(self.windows[index])


def train_g1_smp(cfg: G1SmpTrainCfg) -> dict[str, float]:
  """Train a DDPM noise predictor and save its EMA weights for PPO."""
  torch.manual_seed(cfg.seed)
  device = torch.device(cfg.device)
  if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA training was requested but no CUDA device is available.")
  train = _WindowDataset(cfg.feature_file, cfg.normalizer_file, "train")
  validation = _WindowDataset(cfg.feature_file, cfg.normalizer_file, "validation")
  train_loader = DataLoader(
    train, batch_size=cfg.batch_size, shuffle=True, drop_last=False
  )
  validation_loader = DataLoader(validation, batch_size=cfg.batch_size, shuffle=False)
  model = SmpDenoiser(
    SmpDenoiserCfg(
      model_dim=cfg.model_dim,
      num_heads=cfg.num_heads,
      num_layers=cfg.num_layers,
      feedforward_dim=cfg.feedforward_dim,
    )
  ).to(device)
  ema = copy.deepcopy(model).eval()
  for parameter in ema.parameters():
    parameter.requires_grad_(False)
  optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
  beta = torch.linspace(1e-4, 0.02, model.cfg.diffusion_steps, device=device)
  alpha_bar = torch.cumprod(1.0 - beta, dim=0)
  scaler = torch.amp.GradScaler(
    device.type, enabled=cfg.mixed_precision and device.type == "cuda"
  )
  best_loss = float("inf")
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  for epoch in range(cfg.epochs):
    model.train()
    for clean in train_loader:
      clean = clean.to(device)
      timestep = torch.randint(
        0, model.cfg.diffusion_steps, (len(clean),), device=device
      )
      noise = torch.randn_like(clean)
      scale = alpha_bar[timestep].sqrt()[:, None, None]
      noised = scale * clean + (1.0 - alpha_bar[timestep]).sqrt()[:, None, None] * noise
      optimizer.zero_grad(set_to_none=True)
      with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
        loss = torch.mean((model(noised, timestep) - noise) ** 2)
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      with torch.no_grad():
        for ema_parameter, parameter in zip(
          ema.parameters(), model.parameters(), strict=True
        ):
          ema_parameter.lerp_(parameter, 1.0 - cfg.ema_decay)
    validation_loss = _validation_loss(ema, validation_loader, alpha_bar, device)
    checkpoint = _checkpoint(ema, cfg, epoch + 1, validation_loss)
    torch.save(checkpoint, cfg.output_dir / "latest.pt")
    if validation_loss < best_loss:
      best_loss = validation_loss
      torch.save(checkpoint, cfg.output_dir / "best.pt")
  (cfg.output_dir / "config.json").write_text(
    json.dumps(asdict(cfg), default=str, indent=2)
  )
  return {
    "validation_loss": best_loss,
    "train_windows": float(len(train)),
    "parameter_count": float(
      sum(parameter.numel() for parameter in model.parameters())
    ),
  }


@torch.no_grad()
def _validation_loss(
  model: SmpDenoiser,
  loader: DataLoader[torch.Tensor],
  alpha_bar: torch.Tensor,
  device: torch.device,
) -> float:
  model.eval()
  losses = []
  for clean in loader:
    clean = clean.to(device)
    timestep = torch.randint(0, model.cfg.diffusion_steps, (len(clean),), device=device)
    noise = torch.randn_like(clean)
    noised = (
      alpha_bar[timestep].sqrt()[:, None, None] * clean
      + (1.0 - alpha_bar[timestep]).sqrt()[:, None, None] * noise
    )
    losses.append(torch.mean((model(noised, timestep) - noise) ** 2).item())
  return float(np.mean(losses))


def _checkpoint(
  model: SmpDenoiser, cfg: G1SmpTrainCfg, epoch: int, validation_loss: float
) -> dict[str, object]:
  return {
    "schema_version": "g1-smp-checkpoint-v1",
    "model_cfg": asdict(model.cfg),
    "ema_state_dict": model.state_dict(),
    "train_cfg": asdict(cfg),
    "epoch": epoch,
    "validation_loss": validation_loss,
  }
