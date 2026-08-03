from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  rlboy_flat_env_cfg,
  rlboy_flat_recovery_env_cfg,
  rlboy_rough_env_cfg,
  rlboy_rough_recovery_env_cfg,
)
from .rl_cfg import rlboy_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough",
  env_cfg=rlboy_rough_env_cfg(),
  play_env_cfg=rlboy_rough_env_cfg(play=True),
  rl_cfg=rlboy_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat",
  env_cfg=rlboy_flat_env_cfg(),
  play_env_cfg=rlboy_flat_env_cfg(play=True),
  rl_cfg=rlboy_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Recovery",
  env_cfg=rlboy_rough_recovery_env_cfg(),
  play_env_cfg=rlboy_rough_recovery_env_cfg(play=True),
  rl_cfg=rlboy_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Recovery",
  env_cfg=rlboy_flat_recovery_env_cfg(),
  play_env_cfg=rlboy_flat_recovery_env_cfg(play=True),
  rl_cfg=rlboy_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
