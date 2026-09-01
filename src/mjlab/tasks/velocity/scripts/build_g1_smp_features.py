"""Build 51-D SMP features from compiled G1 get-up clips."""

from __future__ import annotations

import tyro

from mjlab.tasks.velocity.recovery_prior.g1_smp_data import (
  G1SmpFeatureCompilerCfg,
  compile_g1_smp_features,
)


def main() -> None:
  cfg = tyro.cli(G1SmpFeatureCompilerCfg)
  result = compile_g1_smp_features(cfg)
  print(f"Built {result['frame_count']} SMP frames from {result['clip_count']} clips")


if __name__ == "__main__":
  main()
