"""Train the contact-label-free MLD-style recovery VAE."""

from __future__ import annotations

import tyro

from mjlab.tasks.velocity.recovery_prior.train import (
  RecoveryVaeTrainCfg,
  train_recovery_vae,
)


def main() -> None:
  cfg = tyro.cli(RecoveryVaeTrainCfg)
  train_recovery_vae(cfg)


if __name__ == "__main__":
  main()
