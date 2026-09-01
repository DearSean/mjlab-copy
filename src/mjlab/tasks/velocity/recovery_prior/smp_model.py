"""Small Transformer denoiser used by the frozen G1 SMP reward."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from mjlab.tasks.velocity.recovery_prior.g1_smp_data import G1_SMP_FEATURE_DIM


@dataclass(frozen=True, kw_only=True)
class SmpDenoiserCfg:
  feature_dim: int = G1_SMP_FEATURE_DIM
  window_size: int = 10
  diffusion_steps: int = 50
  model_dim: int = 128
  num_heads: int = 4
  num_layers: int = 2
  feedforward_dim: int = 512
  dropout: float = 0.0


@dataclass(frozen=True, kw_only=True)
class FrozenG1SmpPriorCfg:
  """Selected EMA prior, kept separate from deployable actor checkpoints."""

  checkpoint: Path = Path(
    "artifacts/g1_recovery/smp_benchmark/transformer_2x128/seed_1/best.pt"
  )
  checkpoint_schema_version: str = "g1-smp-checkpoint-v1"
  normalizer_file: Path = Path("artifacts/g1_recovery/smp_normalizer.npz")
  esm_timesteps: tuple[int, int, int] = (22, 15, 8)
  esm_error_means: tuple[float, float, float] = (
    0.2257426679,
    0.2678150833,
    0.3945539594,
  )
  """Held-out fixed-noise means for timesteps 22, 15, and 8."""
  smp_scale: float = 1.0
  reward_weight: float = 10.0
  terminal_reward_weight: float = 2.5
  handoff_progress: tuple[float, float] = (0.65, 0.85)
  """Progress handoff from the full motion prior to the standing objective."""


class SmpDenoiser(nn.Module):
  """Timestep-conditioned Transformer noise predictor."""

  def __init__(self, cfg: SmpDenoiserCfg | None = None) -> None:
    super().__init__()
    self.cfg = cfg or SmpDenoiserCfg()
    self.input_projection = nn.Linear(self.cfg.feature_dim, self.cfg.model_dim)
    self.position = nn.Parameter(
      torch.empty(1, self.cfg.window_size, self.cfg.model_dim)
    )
    self.timestep = nn.Embedding(self.cfg.diffusion_steps, self.cfg.model_dim)
    layer = nn.TransformerEncoderLayer(
      d_model=self.cfg.model_dim,
      nhead=self.cfg.num_heads,
      dim_feedforward=self.cfg.feedforward_dim,
      dropout=self.cfg.dropout,
      activation="gelu",
      batch_first=True,
      norm_first=True,
    )
    self.encoder = nn.TransformerEncoder(layer, num_layers=self.cfg.num_layers)
    self.output_projection = nn.Linear(self.cfg.model_dim, self.cfg.feature_dim)
    nn.init.normal_(self.position, std=0.02)

  def forward(self, window: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    if window.ndim != 3 or window.shape[1:] != (
      self.cfg.window_size,
      self.cfg.feature_dim,
    ):
      raise ValueError("window must have shape [B, 10, 51].")
    if timestep.shape != (window.shape[0],):
      raise ValueError("timestep must have shape [B].")
    tokens = (
      self.input_projection(window) + self.position + self.timestep(timestep)[:, None]
    )
    return self.output_projection(self.encoder(tokens))
