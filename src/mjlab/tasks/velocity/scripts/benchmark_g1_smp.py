"""Run deterministic G1 SMP architecture and seed ablations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro

from mjlab.tasks.velocity.recovery_prior.train_smp import G1SmpTrainCfg, train_g1_smp
from mjlab.tasks.velocity.scripts.evaluate_g1_smp import (
  G1SmpEvaluationCfg,
  evaluate_g1_smp,
)


@dataclass(frozen=True, kw_only=True)
class G1SmpBenchmarkCfg:
  feature_file: Path = Path("artifacts/g1_recovery/smp_features.npz")
  normalizer_file: Path = Path("artifacts/g1_recovery/smp_normalizer.npz")
  output_dir: Path = Path("artifacts/g1_recovery/smp_benchmark")
  epochs: int = 300
  batch_size: int = 128
  seeds: tuple[int, ...] = (0, 1, 2)
  device: str = "cuda"


_ARCHITECTURES: dict[str, dict[str, int]] = {
  "transformer_1x128": {
    "model_dim": 128,
    "num_heads": 4,
    "num_layers": 1,
    "feedforward_dim": 512,
  },
  "transformer_2x128": {
    "model_dim": 128,
    "num_heads": 4,
    "num_layers": 2,
    "feedforward_dim": 512,
  },
  "transformer_2x256": {
    "model_dim": 256,
    "num_heads": 8,
    "num_layers": 2,
    "feedforward_dim": 1024,
  },
}


def benchmark_g1_smp(cfg: G1SmpBenchmarkCfg) -> list[dict[str, float | int | str]]:
  """Train each architecture/seed pair and rank held-out SMP discrimination."""
  rows: list[dict[str, float | int | str]] = []
  for architecture, model_cfg in _ARCHITECTURES.items():
    for seed in cfg.seeds:
      run_dir = cfg.output_dir / architecture / f"seed_{seed}"
      train_result = train_g1_smp(
        G1SmpTrainCfg(
          feature_file=cfg.feature_file,
          normalizer_file=cfg.normalizer_file,
          output_dir=run_dir,
          epochs=cfg.epochs,
          batch_size=cfg.batch_size,
          seed=seed,
          device=cfg.device,
          model_dim=model_cfg["model_dim"],
          num_heads=model_cfg["num_heads"],
          num_layers=model_cfg["num_layers"],
          feedforward_dim=model_cfg["feedforward_dim"],
        )
      )
      evaluation = evaluate_g1_smp(
        G1SmpEvaluationCfg(
          checkpoint=run_dir / "best.pt",
          feature_file=cfg.feature_file,
          normalizer_file=cfg.normalizer_file,
          output_file=run_dir / "evaluation.json",
          batch_size=cfg.batch_size,
          seed=seed,
          device=cfg.device,
        )
      )
      best_corrupted = max(
        evaluation["time_shuffled"],
        evaluation["joint_shuffled"],
        evaluation["random_noise"],
      )
      rows.append(
        {
          "architecture": architecture,
          "seed": seed,
          "validation_loss": train_result["validation_loss"],
          "parameter_count": train_result["parameter_count"],
          "real_reward": evaluation["real"],
          "best_corrupted_reward": best_corrupted,
          "reward_margin": evaluation["real"] - best_corrupted,
        }
      )
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  (cfg.output_dir / "summary.json").write_text(
    json.dumps({"config": asdict(cfg), "runs": rows}, default=str, indent=2) + "\n",
    encoding="utf-8",
  )
  return rows


def main() -> None:
  rows = benchmark_g1_smp(tyro.cli(G1SmpBenchmarkCfg))
  for row in rows:
    print(
      f"{row['architecture']} seed={row['seed']}: "
      f"loss={row['validation_loss']:.4f}, margin={row['reward_margin']:.4f}"
    )


if __name__ == "__main__":
  main()
