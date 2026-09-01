"""Compile explicitly-versioned G1 get-up CSV clips at 50 Hz."""

from __future__ import annotations

import tyro

from mjlab.tasks.velocity.recovery_data.g1_compiler import (
  G1RecoveryCompilerCfg,
  compile_g1_recovery_dataset,
)


def main() -> None:
  cfg = tyro.cli(G1RecoveryCompilerCfg)
  manifest = compile_g1_recovery_dataset(cfg)
  print(
    f"Compiled {len(manifest['clips'])} G1 recovery clips to {cfg.output_dir.resolve()}"
  )


if __name__ == "__main__":
  main()
