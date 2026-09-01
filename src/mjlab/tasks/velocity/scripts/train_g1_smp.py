"""Train the G1 frozen score-matching motion prior."""

from __future__ import annotations

import tyro

from mjlab.tasks.velocity.recovery_prior.train_smp import G1SmpTrainCfg, train_g1_smp


def main() -> None:
  result = train_g1_smp(tyro.cli(G1SmpTrainCfg))
  print(f"Finished G1 SMP training: validation_loss={result['validation_loss']:.6f}")


if __name__ == "__main__":
  main()
