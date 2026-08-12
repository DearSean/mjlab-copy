from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from mjlab.tasks.velocity.recovery_data import (
  RECOVERY_SEMANTIC_DIM,
  CanonicalMotionClip,
  RecoverySemanticEncoder,
  Skeleton,
  build_recovery_manifest,
  load_lafan_bvh,
  write_recovery_manifest,
)
from mjlab.tasks.velocity.recovery_data.manifest import (
  RecoverySegmenterCfg,
  extract_recovery_segments,
)

_LAFAN_ROOT = os.environ.get("LAFAN1_ROOT")
_HOLOSOMA_LAFAN_ROOT = os.environ.get("HOLOSOMA_LAFAN_ROOT")


def test_bvh_reader_ignores_root_offset_and_preserves_handedness(tmp_path: Path):
  path = tmp_path / "walk1_subject1.bvh"
  _write_minimal_bvh(path)

  clip = load_lafan_bvh(path)

  assert clip.frame_count == 2
  assert clip.fps == pytest.approx(30.0)
  assert clip.skeleton.joint_names == ("Hips", "Head")
  # Source root (100, 200, 300) cm maps to (1, -3, 2) m. The hierarchy
  # root offset is deliberately not added a second time.
  np.testing.assert_allclose(clip.global_positions_m[0, 0], [1.0, -3.0, 2.0])
  np.testing.assert_allclose(clip.global_positions_m[0, 1], [1.0, -3.0, 3.0])
  assert np.linalg.det(_quat_to_matrix(clip.global_quat_wxyz[0, 0])) == pytest.approx(
    1.0, abs=1e-6
  )


def test_semantic_encoder_has_fixed_shape_and_is_similarity_invariant():
  clip = _standing_lafan_clip()
  transformed = _transform_clip(clip, yaw=0.73, scale=1.4, translation=(3.0, -2.0, 0.4))

  original_semantic = RecoverySemanticEncoder().encode(clip)
  transformed_semantic = RecoverySemanticEncoder().encode(transformed)

  assert original_semantic.features.shape == (clip.frame_count, RECOVERY_SEMANTIC_DIM)
  assert len(original_semantic.feature_names) == RECOVERY_SEMANTIC_DIM
  np.testing.assert_allclose(
    original_semantic.features,
    transformed_semantic.features,
    atol=2e-5,
  )
  assert np.all(original_semantic.progress > 0.8)


def test_contact_encoder_preserves_fast_horizontal_sliding_support():
  clip = _standing_lafan_clip(frame_count=6, fps=30.0)
  left_knee = clip.skeleton.index("LeftLeg")
  clip.global_positions_m[:, left_knee, 0] = np.linspace(-0.1, 0.5, clip.frame_count)
  clip.global_positions_m[:, left_knee, 2] = 0.03

  semantic = RecoverySemanticEncoder().encode(clip)

  assert np.all(semantic.contacts[:, 4])


def test_foot_contact_requires_temporal_confirmation():
  clip = _standing_lafan_clip(frame_count=6, fps=30.0)
  left_foot = clip.skeleton.index("LeftToe")
  clip.global_positions_m[:2, left_foot, 2] = 0.2
  clip.global_positions_m[2:, left_foot, 2] = 0.02

  contacts = RecoverySemanticEncoder().encode(clip).contacts[:, 6]

  assert not contacts[2]
  assert contacts[3]


def test_recovery_segmenter_finds_stable_terminal_interval():
  clip = _standing_lafan_clip(frame_count=80, fps=20.0)
  progress = np.concatenate(
    [
      np.ones(5),
      np.zeros(20),
      np.linspace(0.0, 1.0, 25),
      np.ones(30),
    ]
  )

  segments = extract_recovery_segments(
    clip,
    progress,
    RecoverySegmenterCfg(ready_hold_s=0.5, max_duration_s=4.0),
  )

  assert len(segments) == 1
  assert segments[0].start_frame <= 25
  assert segments[0].end_frame > 50
  assert segments[0].terminal_progress >= 0.85


def test_manifest_groups_synchronized_subjects_and_writes_deterministically(
  tmp_path: Path,
):
  dataset = tmp_path / "lafan"
  dataset.mkdir()
  _write_minimal_bvh(dataset / "walk1_subject1.bvh")
  _write_minimal_bvh(dataset / "walk1_subject4.bvh")

  manifest = build_recovery_manifest(dataset)
  first = tmp_path / "first.json"
  second = tmp_path / "second.json"
  write_recovery_manifest(manifest, first)
  write_recovery_manifest(manifest, second)

  assert len(manifest.clips) == 2
  assert manifest.frame_count == 4
  assert manifest.clips[0].split == manifest.clips[1].split
  assert first.read_bytes() == second.read_bytes()


@pytest.mark.slow
@pytest.mark.skipif(
  _LAFAN_ROOT is None or _HOLOSOMA_LAFAN_ROOT is None,
  reason="Set LAFAN1_ROOT and HOLOSOMA_LAFAN_ROOT to run this integration test.",
)
def test_lafan_fk_matches_reference_global_positions():
  assert _LAFAN_ROOT is not None
  assert _HOLOSOMA_LAFAN_ROOT is not None
  clip = load_lafan_bvh(Path(_LAFAN_ROOT) / "fallAndGetUp1_subject1.bvh")
  reference_y_up = np.load(Path(_HOLOSOMA_LAFAN_ROOT) / "fallAndGetUp1_subject1.npy")
  y_up_to_z_up = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
  reference_z_up = np.einsum("ij,tkj->tki", y_up_to_z_up, reference_y_up)

  np.testing.assert_allclose(
    clip.global_positions_m,
    reference_z_up,
    atol=1e-6,
  )


@pytest.mark.slow
@pytest.mark.skipif(
  _LAFAN_ROOT is None,
  reason="Set LAFAN1_ROOT to run this contact-label regression test.",
)
def test_fall_and_get_up_global_frame_267_support_contacts():
  assert _LAFAN_ROOT is not None
  clip = load_lafan_bvh(Path(_LAFAN_ROOT) / "fallAndGetUp1_subject1.bvh")

  all_contacts = RecoverySemanticEncoder().encode(clip).contacts
  contacts = all_contacts[267]

  # Hands, left knee, and both feet visibly support the dynamic recovery.
  np.testing.assert_array_equal(
    contacts,
    [False, False, True, True, True, False, True, True],
  )

  assert not np.any(all_contacts[285:290, 6])
  assert all_contacts[290, 6]


def _write_minimal_bvh(path: Path) -> None:
  path.write_text(
    """HIERARCHY
ROOT Hips
{
  OFFSET 100 200 300
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Head
  {
    OFFSET 0 100 0
    CHANNELS 3 Zrotation Yrotation Xrotation
    End Site
    {
      OFFSET 0 10 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.03333333333333333
100 200 300 0 0 0 0 0 0
100 200 300 90 0 0 0 0 0
""",
    encoding="utf-8",
  )


def _standing_lafan_clip(
  *,
  frame_count: int = 4,
  fps: float = 30.0,
) -> CanonicalMotionClip:
  names = (
    "Hips",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToe",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToe",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
  )
  parents = np.asarray(
    [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20],
    dtype=np.int64,
  )
  pose = {
    "Hips": (0.0, 0.0, 0.95),
    "LeftUpLeg": (-0.10, 0.0, 0.90),
    "LeftLeg": (-0.10, 0.0, 0.50),
    "LeftFoot": (-0.10, 0.0, 0.08),
    "LeftToe": (-0.10, 0.15, 0.0),
    "RightUpLeg": (0.10, 0.0, 0.90),
    "RightLeg": (0.10, 0.0, 0.50),
    "RightFoot": (0.10, 0.0, 0.08),
    "RightToe": (0.10, 0.15, 0.0),
    "Spine": (0.0, 0.0, 1.08),
    "Spine1": (0.0, 0.0, 1.22),
    "Spine2": (0.0, 0.0, 1.38),
    "Neck": (0.0, 0.0, 1.55),
    "Head": (0.0, 0.0, 1.72),
    "LeftShoulder": (-0.18, 0.0, 1.40),
    "LeftArm": (-0.34, 0.0, 1.36),
    "LeftForeArm": (-0.52, 0.0, 1.28),
    "LeftHand": (-0.70, 0.0, 1.20),
    "RightShoulder": (0.18, 0.0, 1.40),
    "RightArm": (0.34, 0.0, 1.36),
    "RightForeArm": (0.52, 0.0, 1.28),
    "RightHand": (0.70, 0.0, 1.20),
  }
  global_position = np.asarray([pose[name] for name in names], dtype=np.float32)
  offsets = np.zeros_like(global_position)
  for index in range(1, len(names)):
    offsets[index] = global_position[index] - global_position[parents[index]]
  global_positions = np.broadcast_to(
    global_position[None], (frame_count, len(names), 3)
  ).copy()
  local_positions = np.broadcast_to(offsets[None], (frame_count, len(names), 3)).copy()
  local_positions[:, 0] = global_positions[:, 0]
  quaternion = np.zeros((frame_count, len(names), 4), dtype=np.float32)
  quaternion[..., 0] = 1.0
  skeleton = Skeleton(joint_names=names, parents=parents, offsets_m=offsets)
  return CanonicalMotionClip(
    skeleton=skeleton,
    fps=fps,
    local_positions_m=local_positions,
    local_quat_wxyz=quaternion,
    global_positions_m=global_positions,
    global_quat_wxyz=quaternion.copy(),
  )


def _transform_clip(
  clip: CanonicalMotionClip,
  *,
  yaw: float,
  scale: float,
  translation: tuple[float, float, float],
) -> CanonicalMotionClip:
  rotation = np.asarray(
    [
      [np.cos(yaw), -np.sin(yaw), 0.0],
      [np.sin(yaw), np.cos(yaw), 0.0],
      [0.0, 0.0, 1.0],
    ]
  )
  translation_array = np.asarray(translation)
  global_positions = (
    scale * np.einsum("ij,tkj->tki", rotation, clip.global_positions_m)
    + translation_array
  )
  offsets = scale * np.einsum("ij,kj->ki", rotation, clip.skeleton.offsets_m)
  local_positions = scale * np.einsum("ij,tkj->tki", rotation, clip.local_positions_m)
  local_positions[:, 0] += translation_array
  yaw_quaternion = np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
  global_quaternion = np.broadcast_to(
    yaw_quaternion,
    clip.global_quat_wxyz.shape,
  ).astype(np.float32, copy=True)
  local_quaternion = clip.local_quat_wxyz.copy()
  local_quaternion[:, 0] = yaw_quaternion
  return CanonicalMotionClip(
    skeleton=Skeleton(
      joint_names=clip.skeleton.joint_names,
      parents=clip.skeleton.parents.copy(),
      offsets_m=offsets.astype(np.float32),
    ),
    fps=clip.fps,
    local_positions_m=local_positions.astype(np.float32),
    local_quat_wxyz=local_quaternion,
    global_positions_m=global_positions.astype(np.float32),
    global_quat_wxyz=global_quaternion,
  )


def _quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
  w, x, y, z = quaternion
  return np.asarray(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
  )
