"""Tests for raw and physical G1 recovery pose selection."""

from pathlib import Path

import numpy as np
import pytest

from mjlab.tasks.velocity.recovery_data.g1_schema import (
  G1_JOINT_DIM,
  G1_PHYSICAL_INIT_SCHEMA_VERSION,
)
from mjlab.tasks.velocity.scripts.visualize_g1_recovery_clip import (
  _load_motion,
  _load_physical,
  _select_motion_frames,
)


def _write_motion(path: Path) -> None:
  np.savez_compressed(
    path,
    root_position=np.arange(12, dtype=np.float32).reshape(4, 3),
    root_quaternion_xyzw=np.tile(
      np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32), (4, 1)
    ),
    joint_position=np.arange(4 * G1_JOINT_DIM, dtype=np.float32).reshape(
      4, G1_JOINT_DIM
    ),
  )


def _write_physical(path: Path) -> None:
  np.savez_compressed(
    path,
    schema_version=np.asarray(G1_PHYSICAL_INIT_SCHEMA_VERSION),
    clip_id=np.asarray(("other", "clip", "clip")),
    frame=np.asarray((0, 3, 1), dtype=np.int32),
    root_position=np.asarray(
      ((0.0, 0.0, 0.1), (3.0, 3.0, 3.0), (1.0, 1.0, 1.0)),
      dtype=np.float32,
    ),
    root_quaternion_xyzw=np.tile(
      np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32), (3, 1)
    ),
    joint_position=np.zeros((3, G1_JOINT_DIM), dtype=np.float32),
    support_fraction=np.asarray((1.0, 0.9, 0.8), dtype=np.float32),
  )


def test_physical_viewer_pairs_only_validated_frames(tmp_path: Path):
  motion_file = tmp_path / "clip.npz"
  physical_file = tmp_path / "physical_init.npz"
  _write_motion(motion_file)
  _write_physical(physical_file)

  motion, start, end = _load_motion(motion_file, (1, 4))
  physical, audit = _load_physical(physical_file, "clip", start, end)
  paired = _select_motion_frames(motion, physical.frame)

  np.testing.assert_array_equal(physical.frame, (1, 3))
  np.testing.assert_array_equal(paired.frame, physical.frame)
  np.testing.assert_array_equal(paired.root_position[:, 0], (3.0, 9.0))
  np.testing.assert_allclose(audit["support_fraction"], (0.8, 0.9))


def test_physical_viewer_rejects_range_without_validated_frame(tmp_path: Path):
  motion_file = tmp_path / "clip.npz"
  physical_file = tmp_path / "physical_init.npz"
  _write_motion(motion_file)
  _write_physical(physical_file)

  motion, start, end = _load_motion(motion_file, (0, 1))
  assert motion.frame.tolist() == [0]
  with pytest.raises(ValueError, match="No simulator-validated reset frame"):
    _load_physical(physical_file, "clip", start, end)
