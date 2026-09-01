"""Compile 36-column G1 retarget CSV files into auditable 50 Hz clips."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mjlab.tasks.velocity.recovery_data.g1_schema import (
  G1_JOINT_DIM,
  G1_JOINT_NAMES,
  G1_RECOVERY_FEATURE_SCHEMA_VERSION,
  G1_RECOVERY_SCHEMA_VERSION,
  G1_RECOVERY_TARGET_FPS,
  G1RecoveryClip,
  QuaternionOrder,
)


@dataclass(frozen=True, kw_only=True)
class G1RecoveryCompilerCfg:
  """Explicit source conventions and output location for G1 get-up clips."""

  source_dir: Path
  source_fps: float
  quaternion_order: QuaternionOrder
  output_dir: Path = Path("artifacts/g1_recovery")
  source_index: Path | None = None
  split_seed: int = 0
  target_fps: float = G1_RECOVERY_TARGET_FPS

  def __post_init__(self) -> None:
    if not np.isfinite(self.source_fps) or self.source_fps <= 0.0:
      raise ValueError("source_fps must be finite and positive.")
    if not np.isfinite(self.target_fps) or self.target_fps <= 0.0:
      raise ValueError("target_fps must be finite and positive.")


def compile_g1_recovery_dataset(cfg: G1RecoveryCompilerCfg) -> dict[str, Any]:
  """Compile every indexed (or discovered) CSV without mixing source recordings."""
  source_dir = cfg.source_dir.resolve()
  output_dir = cfg.output_dir.resolve()
  if not source_dir.is_dir():
    raise FileNotFoundError(f"G1 source directory does not exist: {source_dir}")
  index_path = (cfg.source_index or source_dir / "index.json").resolve()
  entries = _load_entries(source_dir, index_path if index_path.is_file() else None)
  if not entries:
    raise ValueError(f"No CSV clips found in {source_dir}.")

  clips_dir = output_dir / "clips"
  clips_dir.mkdir(parents=True, exist_ok=True)
  source_splits = _source_splits(entries, cfg.split_seed)
  compiled: list[dict[str, Any]] = []
  for entry in entries:
    relative_path = Path(entry["file"])
    csv_path = source_dir / relative_path
    raw = _read_csv(csv_path)
    clip = _resample_g1_clip(raw, cfg.source_fps, cfg.target_fps, cfg.quaternion_order)
    clip_id = relative_path.stem
    output_path = clips_dir / f"{clip_id}.npz"
    np.savez_compressed(
      output_path,
      root_position=clip.root_position,
      root_quaternion_xyzw=clip.root_quaternion_xyzw,
      joint_position=clip.joint_position,
      root_linear_velocity=clip.root_linear_velocity,
      root_angular_velocity=clip.root_angular_velocity,
      joint_velocity=clip.joint_velocity,
    )
    annotation = entry.get("annotation", {})
    source_recording_id = str(entry.get("source_csv", relative_path.stem))
    compiled.append(
      {
        "clip_id": clip_id,
        "source_recording_id": source_recording_id,
        "subject_id": _subject_id(source_recording_id),
        "split": source_splits[source_recording_id],
        "source_fps": cfg.source_fps,
        "target_fps": cfg.target_fps,
        "valid_length": clip.valid_length,
        "support_start": _target_event_index(
          annotation.get("support_start_frame"), entry, cfg.source_fps, cfg.target_fps
        ),
        "support_complete": _target_event_index(
          annotation.get("support_complete_frame"),
          entry,
          cfg.source_fps,
          cfg.target_fps,
        ),
        "source_support_start_frame": annotation.get("support_start_frame"),
        "source_support_complete_frame": annotation.get("support_complete_frame"),
        "success": annotation.get("outcome") == "success",
        "terminal_type": annotation.get("terminal_mode"),
        "source_file_sha256": _sha256(csv_path),
        "feature_schema_version": G1_RECOVERY_FEATURE_SCHEMA_VERSION,
        "file": output_path.relative_to(output_dir).as_posix(),
      }
    )
  manifest = {
    "schema_version": G1_RECOVERY_SCHEMA_VERSION,
    "feature_schema_version": G1_RECOVERY_FEATURE_SCHEMA_VERSION,
    "joint_names": list(G1_JOINT_NAMES),
    "source_quaternion_order": cfg.quaternion_order,
    "compiler": {
      "source_dir": str(source_dir),
      "source_fps": cfg.source_fps,
      "quaternion_order": cfg.quaternion_order,
      "output_dir": str(output_dir),
      "source_index": str(index_path) if index_path.is_file() else None,
      "split_seed": cfg.split_seed,
      "target_fps": cfg.target_fps,
    },
    "clips": compiled,
  }
  _write_json(output_dir / "manifest.json", manifest)
  return manifest


def _read_csv(path: Path) -> NDArray[np.float64]:
  if not path.is_file():
    raise FileNotFoundError(f"G1 CSV does not exist: {path}")
  values = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
  if values.shape[1] != 7 + G1_JOINT_DIM:
    raise ValueError(
      f"{path} must contain 36 columns (xyz + quaternion + {G1_JOINT_DIM} joints), "
      f"got {values.shape[1]}."
    )
  if not np.all(np.isfinite(values)):
    raise ValueError(f"{path} contains NaN or Inf.")
  return values


def _resample_g1_clip(
  values: NDArray[np.float64],
  source_fps: float,
  target_fps: float,
  order: QuaternionOrder,
) -> G1RecoveryClip:
  duration = (len(values) - 1) / source_fps
  frame_count = int(np.floor(duration * target_fps + 1e-9)) + 1
  target_times = np.arange(frame_count, dtype=np.float64) / target_fps
  coordinate = np.minimum(target_times * source_fps, len(values) - 1)
  before = np.floor(coordinate).astype(np.int64)
  after = np.minimum(before + 1, len(values) - 1)
  fraction = coordinate - before
  root_position = _lerp(values[:, :3], before, after, fraction)
  source_quat = values[:, 3:7]
  if order == "wxyz":
    source_quat = source_quat[:, (1, 2, 3, 0)]
  root_quaternion = _slerp(source_quat, before, after, fraction)
  joint_position = _lerp(values[:, 7:], before, after, fraction)
  return G1RecoveryClip(
    root_position=root_position.astype(np.float32),
    root_quaternion_xyzw=root_quaternion.astype(np.float32),
    joint_position=joint_position.astype(np.float32),
    root_linear_velocity=_differentiate(root_position, target_fps).astype(np.float32),
    root_angular_velocity=_angular_velocity(root_quaternion, target_fps).astype(
      np.float32
    ),
    joint_velocity=_differentiate(joint_position, target_fps).astype(np.float32),
  )


def _lerp(
  values: NDArray[np.float64],
  before: NDArray[np.int64],
  after: NDArray[np.int64],
  fraction: NDArray[np.float64],
) -> NDArray[np.float64]:
  return values[before] * (1.0 - fraction[:, None]) + values[after] * fraction[:, None]


def _slerp(
  values: NDArray[np.float64],
  before: NDArray[np.int64],
  after: NDArray[np.int64],
  fraction: NDArray[np.float64],
) -> NDArray[np.float64]:
  first = values[before] / np.maximum(
    np.linalg.norm(values[before], axis=-1, keepdims=True), 1e-12
  )
  second = values[after] / np.maximum(
    np.linalg.norm(values[after], axis=-1, keepdims=True), 1e-12
  )
  dot = np.sum(first * second, axis=-1, keepdims=True)
  second = np.where(dot < 0.0, -second, second)
  dot = np.clip(np.abs(dot), 0.0, 1.0)
  angle = np.arccos(dot)
  sine = np.sin(angle)
  linear = first * (1.0 - fraction[:, None]) + second * fraction[:, None]
  spherical = (
    first * np.sin((1.0 - fraction[:, None]) * angle)
    + second * np.sin(fraction[:, None] * angle)
  ) / np.where(sine > 1e-8, sine, 1.0)
  result = np.where(sine > 1e-8, spherical, linear)
  return result / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12)


def _differentiate(values: NDArray[np.float64], fps: float) -> NDArray[np.float64]:
  result = np.empty_like(values)
  if len(values) == 1:
    result[0] = 0.0
    return result
  result[0] = (values[1] - values[0]) * fps
  result[-1] = (values[-1] - values[-2]) * fps
  if len(values) > 2:
    result[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
  return result


def _angular_velocity(
  quaternion_xyzw: NDArray[np.float64], fps: float
) -> NDArray[np.float64]:
  if len(quaternion_xyzw) == 1:
    return np.zeros((1, 3), dtype=np.float64)
  result = np.empty((len(quaternion_xyzw), 3), dtype=np.float64)
  result[0] = (
    _rotation_log(_relative_rotation(quaternion_xyzw[0], quaternion_xyzw[1])) * fps
  )
  result[-1] = (
    _rotation_log(_relative_rotation(quaternion_xyzw[-2], quaternion_xyzw[-1])) * fps
  )
  for index in range(1, len(quaternion_xyzw) - 1):
    result[index] = _rotation_log(
      _relative_rotation(quaternion_xyzw[index - 1], quaternion_xyzw[index + 1])
    ) * (0.5 * fps)
  return result


def _relative_rotation(
  first: NDArray[np.float64], second: NDArray[np.float64]
) -> NDArray[np.float64]:
  x1, y1, z1, w1 = first
  x2, y2, z2, w2 = second
  return np.asarray(
    (
      w1 * x2 - x1 * w2 - y1 * z2 + z1 * y2,
      w1 * y2 + x1 * z2 - y1 * w2 - z1 * x2,
      w1 * z2 - x1 * y2 + y1 * x2 - z1 * w2,
      w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2,
    )
  )


def _rotation_log(quaternion_xyzw: NDArray[np.float64]) -> NDArray[np.float64]:
  quaternion = quaternion_xyzw / max(float(np.linalg.norm(quaternion_xyzw)), 1e-12)
  if quaternion[3] < 0.0:
    quaternion = -quaternion
  sine = float(np.linalg.norm(quaternion[:3]))
  if sine < 1e-8:
    return 2.0 * quaternion[:3]
  return quaternion[:3] * (2.0 * np.arctan2(sine, quaternion[3]) / sine)


def _load_entries(source_dir: Path, index_path: Path | None) -> list[dict[str, Any]]:
  if index_path is None:
    return [
      {"file": path.name, "source_csv": path.name}
      for path in sorted(source_dir.glob("*.csv"))
    ]
  raw = json.loads(index_path.read_text(encoding="utf-8"))
  entries = raw.get("segments")
  if not isinstance(entries, list):
    raise ValueError(f"{index_path} must contain a segments list.")
  return entries


def _source_splits(entries: list[dict[str, Any]], seed: int) -> dict[str, str]:
  """Split whole recordings, guaranteeing held-out sources for small corpora."""
  source_ids = sorted(
    {str(entry.get("source_csv", entry["file"])) for entry in entries}
  )
  ranked = sorted(
    source_ids,
    key=lambda source_id: hashlib.sha256(f"{seed}:{source_id}".encode()).digest(),
  )
  count = len(ranked)
  validation_count = 1 if count >= 3 else 0
  test_count = 1 if count >= 2 else 0
  train_count = count - validation_count - test_count
  return {
    source_id: (
      "train"
      if index < train_count
      else "validation"
      if index < train_count + validation_count
      else "test"
    )
    for index, source_id in enumerate(ranked)
  }


def _target_event_index(
  source_frame: object,
  entry: dict[str, Any],
  source_fps: float,
  target_fps: float,
) -> int | None:
  if not isinstance(source_frame, (int, float, str)):
    return None
  start_frame = entry.get("start_frame")
  if not isinstance(start_frame, (int, float, str)):
    return None
  return int(round((int(source_frame) - int(start_frame)) * target_fps / source_fps))


def _subject_id(source_recording_id: str) -> str | None:
  marker = "_subject"
  return (
    source_recording_id.split(marker, 1)[1].removesuffix(".csv")
    if marker in source_recording_id
    else None
  )


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
