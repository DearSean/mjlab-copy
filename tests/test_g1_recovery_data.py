"""Tests for the explicit G1 get-up CSV compiler."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mjlab.tasks.velocity.recovery_data.g1_compiler import (
  G1RecoveryCompilerCfg,
  compile_g1_recovery_dataset,
)
from mjlab.tasks.velocity.recovery_data.g1_schema import G1RecoveryClip
from mjlab.tasks.velocity.recovery_data.g1_windows import (
  g1_reference_commands,
  padded_reference_window,
  valid_smp_window_starts,
)


def test_g1_csv_compiler_resamples_preserves_endpoints_and_groups_sources(
  tmp_path: Path,
):
  source = tmp_path / "source"
  source.mkdir()
  _write_csv(source / "first.csv")
  _write_csv(source / "second.csv")
  index = {
    "segments": [
      {
        "file": "first.csv",
        "source_csv": "recording_subject1.csv",
        "annotation": {"outcome": "success", "terminal_mode": "stationary"},
      },
      {"file": "second.csv", "source_csv": "recording_subject1.csv"},
    ]
  }
  (source / "index.json").write_text(json.dumps(index), encoding="utf-8")

  output = tmp_path / "output"
  manifest = compile_g1_recovery_dataset(
    G1RecoveryCompilerCfg(
      source_dir=source,
      source_fps=25.0,
      quaternion_order="xyzw",
      output_dir=output,
      source_index=source / "index.json",
    )
  )

  assert len(manifest["clips"]) == 2
  assert len({clip["split"] for clip in manifest["clips"]}) == 1
  assert manifest["clips"][0]["valid_length"] == 5
  arrays = np.load(output / manifest["clips"][0]["file"])
  np.testing.assert_allclose(arrays["root_position"][-1], (1.0, 0.0, 0.0))
  np.testing.assert_allclose(arrays["joint_velocity"][2, 0], 12.5)
  commands = g1_reference_commands(_clip_from_arrays(arrays))
  window, mask = padded_reference_window(commands, 0)
  assert window.shape == (21, 38)
  assert mask[:10].sum() == 0
  assert mask[10]
  assert valid_smp_window_starts(11).tolist() == [0, 1]


def _write_csv(path: Path) -> None:
  values = np.zeros((3, 36), dtype=np.float64)
  values[:, 0] = (0.0, 0.5, 1.0)
  values[:, 6] = 1.0
  values[:, 7] = (0.0, 0.5, 1.0)
  np.savetxt(path, values, delimiter=",")


def _clip_from_arrays(arrays: np.lib.npyio.NpzFile) -> G1RecoveryClip:
  return G1RecoveryClip(**{name: arrays[name] for name in arrays.files})
