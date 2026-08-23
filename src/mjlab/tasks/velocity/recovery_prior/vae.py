"""Lightweight MLD-style variational autoencoder for 85D motion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from mjlab.tasks.velocity.recovery_data.schema import RECOVERY_KINEMATIC_DIM
from mjlab.tasks.velocity.recovery_prior.data import (
  KinematicBatch,
  KinematicNormalizer,
)


@dataclass(frozen=True, kw_only=True)
class RecoveryVaeCfg:
  """Small multi-token VAE sized for an approximately 8 GiB GPU."""

  feature_dim: int = RECOVERY_KINEMATIC_DIM
  latent_dim: int = 256
  num_latent_tokens: int = 1
  model_dim: int = 256
  num_heads: int = 4
  num_layers: int = 4
  feedforward_dim: int = 1024
  dropout: float = 0.1
  max_sequence_length: int = 160

  def __post_init__(self) -> None:
    if self.feature_dim != RECOVERY_KINEMATIC_DIM:
      raise ValueError("Recovery VAE requires the 85D kinematic schema.")
    if self.latent_dim <= 0 or self.model_dim <= 0:
      raise ValueError("VAE dimensions must be positive.")
    if self.num_latent_tokens <= 0:
      raise ValueError("num_latent_tokens must be positive.")
    if self.model_dim % self.num_heads != 0:
      raise ValueError("model_dim must be divisible by num_heads.")
    if self.num_layers <= 0 or self.feedforward_dim <= 0:
      raise ValueError("Transformer depth and feedforward size must be positive.")
    if not 0.0 <= self.dropout < 1.0:
      raise ValueError("dropout must lie in [0, 1).")
    if self.max_sequence_length <= 0:
      raise ValueError("max_sequence_length must be positive.")


class RecoveryMotionVae(nn.Module):
  """Encode a variable-length motion into configurable Gaussian latent tokens."""

  def __init__(self, cfg: RecoveryVaeCfg | None = None) -> None:
    super().__init__()
    self.cfg = cfg or RecoveryVaeCfg()
    self.input_projection = nn.Linear(self.cfg.feature_dim, self.cfg.model_dim)
    distribution_token_count = 2 * self.cfg.num_latent_tokens
    self.encoder_distribution_tokens = nn.Parameter(
      torch.empty(1, distribution_token_count, self.cfg.model_dim)
    )
    self.encoder_position = nn.Parameter(
      torch.empty(
        1,
        self.cfg.max_sequence_length + distribution_token_count,
        self.cfg.model_dim,
      )
    )
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=self.cfg.model_dim,
      nhead=self.cfg.num_heads,
      dim_feedforward=self.cfg.feedforward_dim,
      dropout=self.cfg.dropout,
      activation="gelu",
      batch_first=True,
      norm_first=True,
    )
    self.encoder = nn.TransformerEncoder(
      encoder_layer,
      num_layers=self.cfg.num_layers,
      norm=nn.LayerNorm(self.cfg.model_dim),
    )
    self.mean_projection = nn.Linear(self.cfg.model_dim, self.cfg.latent_dim)
    self.logvar_projection = nn.Linear(self.cfg.model_dim, self.cfg.latent_dim)

    self.latent_projection = nn.Linear(self.cfg.latent_dim, self.cfg.model_dim)
    self.decoder_queries = nn.Parameter(
      torch.empty(1, self.cfg.max_sequence_length, self.cfg.model_dim)
    )
    decoder_layer = nn.TransformerDecoderLayer(
      d_model=self.cfg.model_dim,
      nhead=self.cfg.num_heads,
      dim_feedforward=self.cfg.feedforward_dim,
      dropout=self.cfg.dropout,
      activation="gelu",
      batch_first=True,
      norm_first=True,
    )
    self.decoder = nn.TransformerDecoder(
      decoder_layer,
      num_layers=self.cfg.num_layers,
      norm=nn.LayerNorm(self.cfg.model_dim),
    )
    self.output_projection = nn.Linear(self.cfg.model_dim, self.cfg.feature_dim)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    nn.init.normal_(self.encoder_distribution_tokens, std=0.02)
    nn.init.normal_(self.encoder_position, std=0.02)
    nn.init.normal_(self.decoder_queries, std=0.02)
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)

  def encode(
    self,
    motion: torch.Tensor,
    valid_mask: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    self._validate_inputs(motion, valid_mask)
    batch_size, sequence_length, _ = motion.shape
    latent_count = self.cfg.num_latent_tokens
    distribution_token_count = 2 * latent_count
    tokens = self.input_projection(motion)
    distribution_tokens = self.encoder_distribution_tokens.expand(batch_size, -1, -1)
    tokens = torch.cat([distribution_tokens, tokens], dim=1)
    tokens = tokens + self.encoder_position[
      :, : sequence_length + distribution_token_count
    ]
    token_mask = torch.cat(
      [
        torch.ones(
          (batch_size, distribution_token_count),
          dtype=torch.bool,
          device=motion.device,
        ),
        valid_mask,
      ],
      dim=1,
    )
    encoded = self.encoder(tokens, src_key_padding_mask=~token_mask)
    mean = self.mean_projection(encoded[:, :latent_count])
    logvar = self.logvar_projection(
      encoded[:, latent_count:distribution_token_count]
    ).clamp(-12.0, 8.0)
    return mean, logvar

  @staticmethod
  def reparameterize(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    *,
    sample: bool,
  ) -> torch.Tensor:
    if not sample:
      return mean
    return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

  def decode(self, latent: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    expected_shape = (self.cfg.num_latent_tokens, self.cfg.latent_dim)
    if latent.ndim != 3 or latent.shape[1:] != expected_shape:
      raise ValueError(
        f"latent must have shape [B, {self.cfg.num_latent_tokens}, "
        f"{self.cfg.latent_dim}], "
        f"got {tuple(latent.shape)}."
      )
    if valid_mask.ndim != 2 or valid_mask.shape[0] != latent.shape[0]:
      raise ValueError("valid_mask must have shape [B, T].")
    sequence_length = valid_mask.shape[1]
    if sequence_length > self.cfg.max_sequence_length:
      raise ValueError("Sequence exceeds configured VAE maximum length.")
    memory = self.latent_projection(latent)
    queries = self.decoder_queries[:, :sequence_length].expand(latent.shape[0], -1, -1)
    decoded = self.decoder(
      queries,
      memory,
      tgt_key_padding_mask=~valid_mask,
    )
    output = self.output_projection(decoded)
    return output.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

  def forward(
    self,
    motion: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    sample: bool = True,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, logvar = self.encode(motion, valid_mask)
    latent = self.reparameterize(mean, logvar, sample=sample)
    return self.decode(latent, valid_mask), mean, logvar

  def _validate_inputs(
    self,
    motion: torch.Tensor,
    valid_mask: torch.Tensor,
  ) -> None:
    if motion.ndim != 3 or motion.shape[-1] != self.cfg.feature_dim:
      raise ValueError(f"motion must have shape [B, T, {self.cfg.feature_dim}].")
    if valid_mask.shape != motion.shape[:2] or valid_mask.dtype != torch.bool:
      raise ValueError("valid_mask must be boolean with shape [B, T].")
    if motion.shape[1] > self.cfg.max_sequence_length:
      raise ValueError("Sequence exceeds configured VAE maximum length.")
    if not torch.all(valid_mask.any(dim=1)):
      raise ValueError("Every batch item must contain at least one valid frame.")


@dataclass(frozen=True, kw_only=True)
class RecoveryVaeLossCfg:
  pose_weight: float = 1.0
  velocity_weight: float = 0.5
  clearance_weight: float = 0.5
  progress_weight: float = 0.1
  rotation_weight: float = 0.1
  terminal_weight: float = 0.25


class RecoveryVaeLoss(nn.Module):
  """Masked grouped reconstruction, geometry, and KL objectives."""

  feature_mean: torch.Tensor
  feature_std: torch.Tensor

  def __init__(
    self,
    normalizer: KinematicNormalizer,
    cfg: RecoveryVaeLossCfg | None = None,
  ) -> None:
    super().__init__()
    self.cfg = cfg or RecoveryVaeLossCfg()
    self.register_buffer("feature_mean", torch.from_numpy(normalizer.mean.copy()))
    self.register_buffer("feature_std", torch.from_numpy(normalizer.std.copy()))

  def forward(
    self,
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: KinematicBatch,
    *,
    kl_mean: torch.Tensor,
    kl_logvar: torch.Tensor,
    beta: float,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape or prediction.shape != batch.motion.shape:
      raise ValueError("Prediction, target, and batch motion shapes must match.")
    mask = batch.valid_mask
    pose = (
      _masked_smooth_l1(prediction[..., 0:1], target[..., 0:1], mask)
      + _masked_smooth_l1(prediction[..., 1:7], target[..., 1:7], mask)
      + _masked_smooth_l1(prediction[..., 13:40], target[..., 13:40], mask)
    ) / 3.0
    velocity = (
      _masked_smooth_l1(prediction[..., 7:13], target[..., 7:13], mask)
      + _masked_smooth_l1(prediction[..., 40:67], target[..., 40:67], mask)
    ) / 2.0
    clearance = _masked_smooth_l1(prediction[..., 67:75], target[..., 67:75], mask)
    progress = _masked_smooth_l1(prediction[..., 84:85], target[..., 84:85], mask)
    raw_rotation = prediction[..., 1:7] * self.feature_std[1:7] + self.feature_mean[1:7]
    rotation = _rotation_6d_regularization(raw_rotation, mask)
    terminal = _terminal_reconstruction(prediction, target, batch)
    kl = -0.5 * torch.mean(1.0 + kl_logvar - kl_mean.square() - kl_logvar.exp())
    total = (
      self.cfg.pose_weight * pose
      + self.cfg.velocity_weight * velocity
      + self.cfg.clearance_weight * clearance
      + self.cfg.progress_weight * progress
      + self.cfg.rotation_weight * rotation
      + self.cfg.terminal_weight * terminal
      + float(beta) * kl
    )
    metrics = self._physical_metrics(prediction, target, batch)
    metrics.update(
      {
        "loss": total.detach(),
        "pose_loss": pose.detach(),
        "velocity_loss": velocity.detach(),
        "clearance_loss": clearance.detach(),
        "progress_loss": progress.detach(),
        "rotation_loss": rotation.detach(),
        "terminal_loss": terminal.detach(),
        "kl_loss": kl.detach(),
        "beta": total.new_tensor(beta),
      }
    )
    return total, metrics

  def _physical_metrics(
    self,
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: KinematicBatch,
  ) -> dict[str, torch.Tensor]:
    raw_prediction = prediction * self.feature_std + self.feature_mean
    raw_target = target * self.feature_std + self.feature_mean
    height = batch.nominal_height_m[:, None]
    valid = batch.valid_mask
    position_error = (raw_prediction[..., 13:40] - raw_target[..., 13:40]).reshape(
      *prediction.shape[:2], 9, 3
    )
    position_error = torch.linalg.vector_norm(position_error, dim=-1)
    mpjpe_m = _masked_mean(position_error * height.unsqueeze(-1), valid)
    root_height_mae_m = _masked_mean(
      torch.abs(raw_prediction[..., 0] - raw_target[..., 0]) * height,
      valid,
    )
    clearance_height_ratio_mae = _masked_mean(
      torch.abs(raw_prediction[..., 67:75] - raw_target[..., 67:75]),
      valid,
    )
    return {
      "mpjpe_mm": mpjpe_m.detach() * 1000.0,
      "root_height_mae_mm": root_height_mae_m.detach() * 1000.0,
      "clearance_height_ratio_mae": clearance_height_ratio_mae.detach(),
    }


def _masked_smooth_l1(
  prediction: torch.Tensor,
  target: torch.Tensor,
  valid_mask: torch.Tensor,
) -> torch.Tensor:
  value = F.smooth_l1_loss(prediction, target, reduction="none")
  return _masked_mean(value, valid_mask)


def _masked_mean(value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
  mask = valid_mask
  while mask.ndim < value.ndim:
    mask = mask.unsqueeze(-1)
  expanded = mask.expand_as(value)
  return (value * expanded).sum() / expanded.sum().clamp_min(1)


def _rotation_6d_regularization(
  rotation: torch.Tensor,
  valid_mask: torch.Tensor,
) -> torch.Tensor:
  first, second = rotation[..., :3], rotation[..., 3:]
  error = (
    (torch.linalg.vector_norm(first, dim=-1) - 1.0).square()
    + (torch.linalg.vector_norm(second, dim=-1) - 1.0).square()
    + torch.sum(first * second, dim=-1).square()
  )
  return _masked_mean(error, valid_mask)


def _terminal_reconstruction(
  prediction: torch.Tensor,
  target: torch.Tensor,
  batch: KinematicBatch,
) -> torch.Tensor:
  selected = torch.nonzero(batch.complete_recovery, as_tuple=False).flatten()
  if selected.numel() == 0:
    return prediction.sum() * 0.0
  frames = batch.lengths[selected] - 1
  terminal_prediction = prediction[selected, frames]
  terminal_target = target[selected, frames]
  return F.smooth_l1_loss(
    terminal_prediction[..., [0, 1, 2, 3, 4, 5, 6, 84]],
    terminal_target[..., [0, 1, 2, 3, 4, 5, 6, 84]],
  )
