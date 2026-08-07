from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import qlmini2_flat_env_cfg
from .rl_cfg import qlmini2_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-qlmini2",
  env_cfg=qlmini2_flat_env_cfg(),
  play_env_cfg=qlmini2_flat_env_cfg(play=True),
  rl_cfg=qlmini2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
