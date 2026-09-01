from pathlib import Path

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      use_wandb = self.logger.logger_type == "wandb"
      if use_wandb:
        import wandb

      run_name: str = wandb.run.name or "local" if use_wandb and wandb.run else "local"
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if use_wandb and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")

    # 额外保存 TorchScript JIT 格式策略（policy.pt）
    jit_filename = Path(filename).with_suffix(".pt").name
    try:
      self.export_policy_to_jit(str(policy_dir), jit_filename)
      if self.logger.logger_type == "wandb" and self.cfg["upload_model"]:
        import wandb

        wandb.save(str(policy_dir / jit_filename), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] JIT export failed (training continues): {e}")
