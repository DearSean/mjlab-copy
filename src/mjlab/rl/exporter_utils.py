"""Shared utilities for ONNX policy export across RL tasks."""

import onnx
import torch

from mjlab.actuator import IdealPdActuator
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
  """Convert list to CSV string with specified decimal precision."""
  fmt = f"{{:.{decimals}f}}"
  return delimiter.join(
    fmt.format(x)
    if isinstance(x, (int, float))
    else str(x)  # numbers → format, strings → as-is
    for x in arr
  )


def get_base_metadata(
  env: ManagerBasedRlEnv, run_path: str
) -> dict[str, list | str | float]:
  """Get base metadata common to all RL policy exports.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.

  Returns:
    Dictionary of metadata fields that are common across all tasks.
  """
  robot: Entity = env.scene["robot"]
  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, JointPositionAction)
  joint_gains: dict[str, tuple[float, float]] = {}
  for actuator in robot.actuators:
    if isinstance(actuator, IdealPdActuator):
      assert actuator.default_stiffness is not None
      assert actuator.default_damping is not None
      stiffness = actuator.default_stiffness[0].cpu().tolist()
      damping = actuator.default_damping[0].cpu().tolist()
    else:
      global_ctrl_ids = actuator.global_ctrl_ids.cpu().tolist()
      stiffness = env.sim.mj_model.actuator_gainprm[global_ctrl_ids, 0].tolist()
      damping = (-env.sim.mj_model.actuator_biasprm[global_ctrl_ids, 2]).tolist()
    for name, kp, kd in zip(actuator.target_names, stiffness, damping, strict=True):
      joint_gains[name] = (kp, kd)

  actuated_joint_names = [name for name in robot.joint_names if name in joint_gains]
  joint_stiffness = [joint_gains[name][0] for name in actuated_joint_names]
  joint_damping = [joint_gains[name][1] for name in actuated_joint_names]
  return {
    "run_path": run_path,
    "joint_names": list(robot.joint_names),
    "joint_stiffness": joint_stiffness,
    "joint_damping": joint_damping,
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "command_names": list(env.command_manager.active_terms),
    "observation_names": env.observation_manager.active_terms["actor"],
    "action_scale": joint_action._scale[0].cpu().tolist()
    if isinstance(joint_action._scale, torch.Tensor)
    else joint_action._scale,
  }


def attach_metadata_to_onnx(
  onnx_path: str, metadata: dict[str, list | str | float]
) -> None:
  """Attach metadata to an ONNX model file.

  Args:
    onnx_path: Path to the ONNX model file.
    metadata: Dictionary of metadata key-value pairs to attach.
  """
  model = onnx.load(onnx_path)

  for k, v in metadata.items():
    entry = onnx.StringStringEntryProto()
    entry.key = k
    entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
    model.metadata_props.append(entry)

  onnx.save(model, onnx_path)
