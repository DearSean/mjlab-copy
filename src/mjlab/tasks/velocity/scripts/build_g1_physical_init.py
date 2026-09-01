"""Build the simulator-validated G1 recovery reset-state bank."""

from __future__ import annotations

import tyro

import mjlab
from mjlab.tasks.velocity.recovery_data.g1_physical_init import (
  G1PhysicalInitCfg,
  build_g1_physical_init,
)


def main() -> None:
  cfg = tyro.cli(G1PhysicalInitCfg, config=mjlab.TYRO_FLAGS)
  report = build_g1_physical_init(cfg)
  print(
    "Built G1 physical reset bank: "
    f"accepted={report['accepted_frame_count']}/"
    f"{report['source_frame_count']} "
    f"({100.0 * report['acceptance_rate']:.1f}%)"
  )
  print(f"States: {report['output_file']}")
  print(f"Report: {cfg.report_file.resolve()}")


if __name__ == "__main__":
  main()
