"""Build the versioned LaFAN recovery manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from mjlab.tasks.velocity.recovery_data import (
  build_recovery_manifest,
  write_recovery_manifest,
)


@dataclass(kw_only=True)
class BuildRecoveryManifestCfg:
  """Command-line configuration for LaFAN preprocessing."""

  dataset_root: Path
  """Directory containing the original LaFAN ``.bvh`` files."""

  output: Path = Path("artifacts/recovery/lafan_manifest.json")
  """Destination for the deterministic JSON manifest."""

  strict: bool = True
  """Fail on the first invalid BVH instead of recording rejected files."""


def main(cfg: BuildRecoveryManifestCfg) -> None:
  manifest = build_recovery_manifest(cfg.dataset_root, strict=cfg.strict)
  write_recovery_manifest(manifest, cfg.output)
  print(
    "Built LaFAN recovery manifest: "
    f"clips={len(manifest.clips)}, frames={manifest.frame_count}, "
    f"segments={manifest.segment_count}, failures={len(manifest.failures)}, "
    f"output={cfg.output}"
  )


if __name__ == "__main__":
  main(tyro.cli(BuildRecoveryManifestCfg))
