"""Compare G1 SMP scores on real and deliberately corrupted windows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.tasks.velocity.recovery_prior.smp_model import SmpDenoiser, SmpDenoiserCfg
from mjlab.tasks.velocity.recovery_prior.train_smp import _WindowDataset


@dataclass(frozen=True, kw_only=True)
class G1SmpEvaluationCfg:
  checkpoint: Path = Path("artifacts/g1_recovery/smp/best.pt")
  feature_file: Path = Path("artifacts/g1_recovery/smp_features.npz")
  normalizer_file: Path = Path("artifacts/g1_recovery/smp_normalizer.npz")
  output_file: Path = Path("artifacts/g1_recovery/smp/evaluation.json")
  batch_size: int = 128
  seed: int = 0
  device: str = "cuda"


@torch.no_grad()
def evaluate_g1_smp(cfg: G1SmpEvaluationCfg) -> dict[str, float]:
  """Measure whether the frozen denoiser assigns higher scores to real windows."""
  generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
  checkpoint = torch.load(cfg.checkpoint, map_location=cfg.device, weights_only=False)
  model = SmpDenoiser(SmpDenoiserCfg(**checkpoint["model_cfg"])).to(cfg.device)
  model.load_state_dict(checkpoint["ema_state_dict"])
  model.eval()
  dataset = _WindowDataset(cfg.feature_file, cfg.normalizer_file, "validation")
  real = torch.stack([dataset[index] for index in range(len(dataset))]).to(cfg.device)
  variants = {
    "real": real,
    "time_shuffled": real[
      :, torch.randperm(10, generator=generator, device=cfg.device)
    ],
    "joint_shuffled": _joint_shuffled(real, generator),
    "random_noise": torch.randn(real.shape, generator=generator, device=cfg.device),
  }
  results = {
    name: _smp_rewards(model, windows, cfg.batch_size, generator).median().item()
    for name, windows in variants.items()
  }
  results["passed"] = float(
    results["real"]
    > max(results["time_shuffled"], results["joint_shuffled"], results["random_noise"])
  )
  cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
  cfg.output_file.write_text(
    json.dumps(
      {"config": asdict(cfg), "median_rewards": results}, default=str, indent=2
    )
    + "\n",
    encoding="utf-8",
  )
  return results


def _joint_shuffled(window: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
  result = window.clone()
  result[:, :, 10:39] = result[:, :, 10:39][
    :, :, torch.randperm(29, generator=generator, device=window.device)
  ]
  return result


@torch.no_grad()
def _smp_rewards(
  model: SmpDenoiser, windows: torch.Tensor, batch_size: int, generator: torch.Generator
) -> torch.Tensor:
  beta = torch.linspace(1e-4, 0.02, model.cfg.diffusion_steps, device=windows.device)
  alpha_bar = torch.cumprod(1.0 - beta, dim=0)
  rewards = []
  for begin in range(0, len(windows), batch_size):
    window = windows[begin : begin + batch_size]
    errors = []
    for timestep_value in (22, 15, 8):
      timestep = torch.full(
        (len(window),), timestep_value, device=windows.device, dtype=torch.long
      )
      noise = torch.randn(window.shape, generator=generator, device=windows.device)
      noised = (
        alpha_bar[timestep_value].sqrt() * window
        + (1.0 - alpha_bar[timestep_value]).sqrt() * noise
      )
      errors.append(torch.mean((model(noised, timestep) - noise) ** 2, dim=(1, 2)))
    rewards.append(torch.exp(-torch.stack(errors).mean(dim=0)))
  return torch.cat(rewards)


def main() -> None:
  results = evaluate_g1_smp(tyro.cli(G1SmpEvaluationCfg))
  print(
    "G1 SMP median rewards: "
    + ", ".join(f"{name}={value:.4f}" for name, value in results.items())
  )
  if not results["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
