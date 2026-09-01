"""Compile G1 recovery references into the 51-D SMP feature schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML, KNEES_BENT_KEYFRAME
from mjlab.tasks.velocity.recovery_data.g1_schema import G1_JOINT_NAMES
from mjlab.utils.string import resolve_expr

G1_SMP_FEATURE_DIM = 51
G1_SMP_FEATURE_SCHEMA_VERSION = "g1-smp-features-v1"


@dataclass(frozen=True, kw_only=True)
class G1SmpFeatureCompilerCfg:
  dataset_dir: Path = Path("artifacts/g1_recovery")
  output_file: Path = Path("artifacts/g1_recovery/smp_features.npz")
  normalizer_file: Path = Path("artifacts/g1_recovery/smp_normalizer.npz")
  nominal_pelvis_height_m: float = 0.76


def compile_g1_smp_features(cfg: G1SmpFeatureCompilerCfg) -> dict[str, Any]:
  """Build every clip's unnormalized feature sequence and train statistics."""
  dataset_dir = cfg.dataset_dir.resolve()
  manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  data = mujoco.MjData(model)
  joint_ids = _joint_ids(model)
  joint_qpos = model.jnt_qposadr[joint_ids]
  pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  endpoint_ids = np.asarray(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"),
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"),
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot"),
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot"),
    ]
  )
  assert KNEES_BENT_KEYFRAME.joint_pos is not None
  q0 = np.asarray(resolve_expr(KNEES_BENT_KEYFRAME.joint_pos, G1_JOINT_NAMES, 0.0))
  pose_scale = np.maximum(0.5 * np.diff(model.jnt_range[joint_ids], axis=1)[:, 0], 0.1)
  sequences: list[NDArray[np.float32]] = []
  offsets = [0]
  clip_ids: list[str] = []
  splits: list[str] = []
  for clip_meta in manifest["clips"]:
    arrays = np.load(dataset_dir / clip_meta["file"])
    features = _encode_clip(
      model, data, arrays, joint_qpos, pelvis_id, endpoint_ids, q0, pose_scale, cfg
    )
    sequences.append(features)
    offsets.append(offsets[-1] + len(features))
    clip_ids.append(clip_meta["clip_id"])
    splits.append(clip_meta["split"])
  all_features = np.concatenate(sequences, axis=0)
  cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    cfg.output_file,
    features=all_features,
    offsets=np.asarray(offsets, dtype=np.int64),
    clip_ids=np.asarray(clip_ids),
    splits=np.asarray(splits),
  )
  train_chunks = [
    sequence
    for sequence, split in zip(sequences, splits, strict=True)
    if split == "train"
  ]
  train_features = np.concatenate(train_chunks, axis=0)
  np.savez_compressed(
    cfg.normalizer_file,
    mean=train_features.mean(axis=0, dtype=np.float64).astype(np.float32),
    std=np.maximum(train_features.std(axis=0, dtype=np.float64), 1e-6).astype(
      np.float32
    ),
    feature_schema_version=np.asarray(G1_SMP_FEATURE_SCHEMA_VERSION),
  )
  return {"clip_count": len(sequences), "frame_count": len(all_features)}


def _encode_clip(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  arrays: np.lib.npyio.NpzFile,
  joint_qpos: NDArray[np.integer],
  pelvis_id: int,
  endpoint_ids: NDArray[np.integer],
  q0: NDArray[np.float64],
  pose_scale: NDArray[np.float64],
  cfg: G1SmpFeatureCompilerCfg,
) -> NDArray[np.float32]:
  output = np.empty(
    (len(arrays["joint_position"]), G1_SMP_FEATURE_DIM), dtype=np.float32
  )
  for frame in range(len(output)):
    data.qpos[:] = model.qpos0
    data.qpos[:3] = arrays["root_position"][frame]
    data.qpos[3:7] = arrays["root_quaternion_xyzw"][frame, (3, 0, 1, 2)]
    data.qpos[joint_qpos] = arrays["joint_position"][frame]
    mujoco.mj_forward(model, data)
    rotation = data.xmat[pelvis_id].reshape(3, 3)
    inverse_rotation = rotation.T
    pelvis_position = data.xpos[pelvis_id]
    endpoints = np.concatenate(
      (data.xpos[endpoint_ids[:2]], data.site_xpos[endpoint_ids[2:]]), axis=0
    )
    output[frame] = np.concatenate(
      (
        np.asarray((pelvis_position[2] / cfg.nominal_pelvis_height_m,)),
        inverse_rotation @ np.asarray((0.0, 0.0, -1.0)),
        inverse_rotation @ arrays["root_linear_velocity"][frame],
        inverse_rotation @ arrays["root_angular_velocity"][frame],
        (arrays["joint_position"][frame] - q0) / pose_scale,
        ((endpoints - pelvis_position) @ inverse_rotation.T).reshape(-1),
      )
    )
  return output


def _joint_ids(model: mujoco.MjModel) -> NDArray[np.int32]:
  return np.asarray(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      for name in G1_JOINT_NAMES
    ],
    dtype=np.int32,
  )
