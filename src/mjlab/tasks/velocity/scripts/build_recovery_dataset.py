"""Build contact-label-free LaFAN shards for MLD and SMP.

Example:
  uv run python -m mjlab.tasks.velocity.scripts.build_recovery_dataset \
    --dataset-root /path/to/full/lafan1
"""

from __future__ import annotations

import tyro

from mjlab.tasks.velocity.recovery_data.dataset import (
  RecoveryDatasetCompilerCfg,
  compile_recovery_dataset,
)


def main() -> None:
  cfg = tyro.cli(RecoveryDatasetCompilerCfg)
  manifest = compile_recovery_dataset(cfg)
  splits = manifest["splits"]
  print(
    "Compiled recovery kinematic dataset: "
    f"train={splits['train']['recovery_segment_count']}, "
    f"validation={splits['validation']['recovery_segment_count']}, "
    f"test={splits['test']['recovery_segment_count']}, "
    f"output={cfg.output_dir.resolve()}"
  )


if __name__ == "__main__":
  main()
