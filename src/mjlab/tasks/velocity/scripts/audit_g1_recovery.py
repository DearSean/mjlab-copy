"""Audit compiled G1 get-up references against the MuJoCo G1 model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import tyro

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.tasks.velocity.recovery_data.g1_schema import G1_JOINT_NAMES


@dataclass(frozen=True, kw_only=True)
class G1RecoveryAuditCfg:
  """Limits for the S0 reference-state physical consistency audit."""

  dataset_dir: Path = Path("artifacts/g1_recovery")
  output_file: Path = Path("artifacts/g1_recovery/audit.json")
  max_joint_limit_violation_rad: float = 1e-4
  deep_penetration_m: float = 0.01
  fail_on_deep_penetration: bool = False
  """Treat reference self-penetration as a hard failure instead of a diagnostic."""
  max_root_linear_velocity_mps: float = 20.0
  max_root_angular_velocity_radps: float = 40.0
  max_joint_velocity_radps: float = 80.0


def audit_g1_recovery_dataset(cfg: G1RecoveryAuditCfg) -> dict[str, Any]:
  """Check every reference frame after loading it into MuJoCo qpos/qvel."""
  dataset_dir = cfg.dataset_dir.resolve()
  manifest_path = dataset_dir / "manifest.json"
  if not manifest_path.is_file():
    raise FileNotFoundError(f"Compiled manifest does not exist: {manifest_path}")
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  data = mujoco.MjData(model)
  joint_qpos = _joint_qpos_addresses(model)
  joint_dof = _joint_dof_addresses(model)
  report_clips: list[dict[str, Any]] = []
  for metadata in manifest["clips"]:
    arrays = np.load(dataset_dir / metadata["file"])
    report_clips.append(
      _audit_clip(model, data, arrays, joint_qpos, joint_dof, metadata["clip_id"], cfg)
    )
  failures = [clip for clip in report_clips if not clip["passed"]]
  report = {
    "dataset_manifest": str(manifest_path),
    "clip_count": len(report_clips),
    "failed_clip_count": len(failures),
    "passed": not failures,
    "thresholds": {
      "max_joint_limit_violation_rad": cfg.max_joint_limit_violation_rad,
      "deep_penetration_m": cfg.deep_penetration_m,
      "max_root_linear_velocity_mps": cfg.max_root_linear_velocity_mps,
      "max_root_angular_velocity_radps": cfg.max_root_angular_velocity_radps,
      "max_joint_velocity_radps": cfg.max_joint_velocity_radps,
      "fail_on_deep_penetration": cfg.fail_on_deep_penetration,
    },
    "clips": report_clips,
  }
  cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
  cfg.output_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  return report


def _audit_clip(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  arrays: np.lib.npyio.NpzFile,
  joint_qpos: np.ndarray,
  joint_dof: np.ndarray,
  clip_id: str,
  cfg: G1RecoveryAuditCfg,
) -> dict[str, Any]:
  max_penetration = 0.0
  nonfinite_frame_count = 0
  for frame in range(len(arrays["root_position"])):
    qpos = data.qpos
    qvel = data.qvel
    qpos[:] = model.qpos0
    qvel[:] = 0.0
    qpos[:3] = arrays["root_position"][frame]
    qpos[3:7] = arrays["root_quaternion_xyzw"][frame, (3, 0, 1, 2)]
    qpos[joint_qpos] = arrays["joint_position"][frame]
    qvel[:3] = arrays["root_linear_velocity"][frame]
    qvel[3:6] = arrays["root_angular_velocity"][frame]
    qvel[joint_dof] = arrays["joint_velocity"][frame]
    mujoco.mj_forward(model, data)
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
      nonfinite_frame_count += 1
    for contact in data.contact[: data.ncon]:
      max_penetration = max(max_penetration, float(max(0.0, -contact.dist)))
  joint_position = arrays["joint_position"]
  lower = model.jnt_range[_joint_ids(model), 0]
  upper = model.jnt_range[_joint_ids(model), 1]
  joint_limit_violation = float(
    max(np.max(lower - joint_position), np.max(joint_position - upper), 0.0)
  )
  max_root_linear_velocity = float(
    np.max(np.linalg.norm(arrays["root_linear_velocity"], axis=1))
  )
  max_root_angular_velocity = float(
    np.max(np.linalg.norm(arrays["root_angular_velocity"], axis=1))
  )
  max_joint_velocity = float(np.max(np.abs(arrays["joint_velocity"])))
  passed = (
    nonfinite_frame_count == 0
    and joint_limit_violation <= cfg.max_joint_limit_violation_rad
    and (not cfg.fail_on_deep_penetration or max_penetration <= cfg.deep_penetration_m)
    and max_root_linear_velocity <= cfg.max_root_linear_velocity_mps
    and max_root_angular_velocity <= cfg.max_root_angular_velocity_radps
    and max_joint_velocity <= cfg.max_joint_velocity_radps
  )
  return {
    "clip_id": clip_id,
    "frame_count": int(len(joint_position)),
    "max_penetration_m": max_penetration,
    "deep_penetration_detected": max_penetration > cfg.deep_penetration_m,
    "max_joint_limit_violation_rad": joint_limit_violation,
    "max_root_linear_velocity_mps": max_root_linear_velocity,
    "max_root_angular_velocity_radps": max_root_angular_velocity,
    "max_joint_velocity_radps": max_joint_velocity,
    "nonfinite_frame_count": nonfinite_frame_count,
    "passed": passed,
  }


def _joint_ids(model: mujoco.MjModel) -> np.ndarray:
  return np.asarray(
    [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      for name in G1_JOINT_NAMES
    ]
  )


def _joint_qpos_addresses(model: mujoco.MjModel) -> np.ndarray:
  return model.jnt_qposadr[_joint_ids(model)]


def _joint_dof_addresses(model: mujoco.MjModel) -> np.ndarray:
  return model.jnt_dofadr[_joint_ids(model)]


def main() -> None:
  cfg = tyro.cli(G1RecoveryAuditCfg)
  report = audit_g1_recovery_dataset(cfg)
  print(
    f"Audited {report['clip_count']} G1 recovery clips: "
    f"passed={report['passed']}, failed={report['failed_clip_count']}"
  )
  if not report["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
