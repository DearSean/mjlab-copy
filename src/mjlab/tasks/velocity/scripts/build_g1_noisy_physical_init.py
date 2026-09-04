"""Build collision-checked noisy variants of the G1 recovery reset bank."""

from __future__ import annotations

import tyro

import mjlab
from mjlab.tasks.velocity.recovery_data.g1_physical_init import (
  G1NoisyPhysicalInitCfg,
  build_g1_noisy_physical_init,
)


def main() -> None:
  cfg = tyro.cli(G1NoisyPhysicalInitCfg, config=mjlab.TYRO_FLAGS)
  report = build_g1_noisy_physical_init(cfg)
  print(
    "Built G1 noisy physical reset bank: "
    f"accepted={report['accepted_variant_count']}, "
    f"frame_coverage={100.0 * report['frame_coverage']:.1f}%"
  )
  print(f"States: {report['output_file']}")
  print(f"Report: {cfg.report_file.resolve()}")


if __name__ == "__main__":
  main()
