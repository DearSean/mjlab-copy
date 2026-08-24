from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None
  payload_range: tuple[float, float] | None


class CommandModeStage(TypedDict, total=False):
  """A curriculum stage for categorical velocity-command probabilities."""

  step: int
  standing: float
  forward: float
  backward: float
  lateral: float
  yaw: float
  mixed: float
  min_linear_speed: float
  min_angular_speed: float


class PushStage(TypedDict):
  """A curriculum stage for interval velocity pushes."""

  step: int
  interval_range_s: tuple[float, float]
  velocity_range: dict[str, tuple[float, float]]


class AdaptiveVelocityStage(TypedDict):
  """One joint command, disturbance, and regularization difficulty level."""

  step: int
  success_threshold: float | None
  lin_vel_x: tuple[float, float]
  lin_vel_y: tuple[float, float]
  ang_vel_z: tuple[float, float]
  mode_probabilities: dict[str, float]
  interval_range_s: tuple[float, float]
  velocity_range: dict[str, tuple[float, float]]
  reward_weights: dict[str, float]


class adaptive_velocity_progression:
  """Advance qlmini2 difficulty only after a window of successful episodes.

  A successful velocity episode is one that reaches the configured time limit.
  The wall-clock training step is therefore an *earliest* stage boundary, while
  the success threshold prevents a sudden increase in command, push, and cost
  difficulty when the current policy is still falling frequently.
  """

  def __init__(self, cfg: Any, env: ManagerBasedRlEnv):
    params = cfg.params
    self._command_name = cast(str, params["command_name"])
    self._push_event_name = cast(str, params["push_event_name"])
    self._stages = cast(list[AdaptiveVelocityStage], params["stages"])
    self._window_size = cast(int, params.get("window_size", 4096))
    if not self._stages:
      raise ValueError("adaptive velocity progression requires at least one stage.")
    if self._window_size <= 0:
      raise ValueError("adaptive velocity progression window_size must be positive.")
    self._stage_index = 0
    self._attempts = torch.zeros((), dtype=torch.long, device=env.device)
    self._successes = torch.zeros((), dtype=torch.long, device=env.device)

  def _apply_stage(self, env: ManagerBasedRlEnv) -> None:
    stage = self._stages[self._stage_index]
    command_term = env.command_manager.get_term(self._command_name)
    assert command_term is not None
    command_cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
    command_cfg.ranges.lin_vel_x = stage["lin_vel_x"]
    command_cfg.ranges.lin_vel_y = stage["lin_vel_y"]
    command_cfg.ranges.ang_vel_z = stage["ang_vel_z"]
    command_cfg.mode_probabilities = UniformVelocityCommandCfg.ModeProbabilities(
      **stage["mode_probabilities"]
    )
    command_cfg.validate()

    push_cfg = env.event_manager.get_term_cfg(self._push_event_name)
    push_cfg.interval_range_s = stage["interval_range_s"]
    push_cfg.params["velocity_range"] = stage["velocity_range"]

    for reward_name, weight in stage["reward_weights"].items():
      env.reward_manager.get_term_cfg(reward_name).weight = weight

  def _record_outcomes(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice
  ) -> None:
    if isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=env.device)
    # The initial reset has no preceding episode and must not count as a failure.
    completed = env_ids[env.episode_length_buf[env_ids] > 0]
    if len(completed) == 0:
      return
    self._attempts += len(completed)
    self._successes += env.termination_manager.time_outs[completed].sum()

  def _try_advance(self, env: ManagerBasedRlEnv) -> None:
    if self._stage_index >= len(self._stages) - 1:
      return
    next_stage = self._stages[self._stage_index + 1]
    success_threshold = next_stage["success_threshold"]
    if env.common_step_counter < next_stage["step"]:
      return
    if self._attempts < self._window_size:
      return
    success_rate = self._successes.float() / self._attempts.float()
    if success_threshold is None or success_rate >= success_threshold:
      self._stage_index += 1
    # Start a fresh, same-difficulty evaluation window whether it passed or not.
    self._attempts.zero_()
    self._successes.zero_()

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice, **kwargs: Any
  ) -> dict[str, torch.Tensor]:
    del kwargs
    self._record_outcomes(env, env_ids)
    self._try_advance(env)
    self._apply_stage(env)
    stage = self._stages[self._stage_index]
    mode = stage["mode_probabilities"]
    max_push = max(
      max(abs(low), abs(high)) for low, high in stage["velocity_range"].values()
    )
    success_rate = self._successes.float() / self._attempts.clamp_min(1).float()
    return {
      "stage": torch.tensor(self._stage_index, device=env.device),
      "success_rate": success_rate,
      "attempts": self._attempts,
      "required_success_rate": torch.tensor(
        stage["success_threshold"] or 0.0, device=env.device
      ),
      "lin_vel_x_min": torch.tensor(stage["lin_vel_x"][0], device=env.device),
      "lin_vel_x_max": torch.tensor(stage["lin_vel_x"][1], device=env.device),
      "lin_vel_y_max": torch.tensor(stage["lin_vel_y"][1], device=env.device),
      "ang_vel_z_max": torch.tensor(stage["ang_vel_z"][1], device=env.device),
      "backward_prob": torch.tensor(mode["backward"], device=env.device),
      "lateral_prob": torch.tensor(mode["lateral"], device=env.device),
      "yaw_prob": torch.tensor(mode["yaw"], device=env.device),
      "mixed_prob": torch.tensor(mode["mixed"], device=env.device),
      "push_max_velocity": torch.tensor(max_push, device=env.device),
    }


def staged_push_by_setting_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> None:
  """Apply an interval push unless its active curriculum stage is disabled."""
  if all(low == 0.0 and high == 0.0 for low, high in velocity_range.values()):
    return
  envs_mdp.push_by_setting_velocity(
    env, env_ids, velocity_range=velocity_range, asset_cfg=asset_cfg
  )


def push_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  stages: list[PushStage],
) -> dict[str, torch.Tensor]:
  """Update the active interval-push stage and report its scale."""
  del env_ids
  event_cfg = env.event_manager.get_term_cfg(event_name)
  active_stage = stages[0]
  stage_index = 0
  for index, stage in enumerate(stages):
    if env.common_step_counter >= stage["step"]:
      active_stage = stage
      stage_index = index
  event_cfg.interval_range_s = active_stage["interval_range_s"]
  event_cfg.params["velocity_range"] = active_stage["velocity_range"]
  max_push = max(
    max(abs(low), abs(high)) for low, high in active_stage["velocity_range"].values()
  )
  return {
    "stage": torch.tensor(stage_index, device=env.device),
    "interval_min_s": torch.tensor(active_stage["interval_range_s"][0]),
    "interval_max_s": torch.tensor(active_stage["interval_range_s"][1]),
    "max_velocity": torch.tensor(max_push),
  }


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
  mode_stages: list[CommandModeStage] | None = None,
  payload_event_name: str | None = None,
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  payload_cfg = (
    env.event_manager.get_term_cfg(payload_event_name)
    if payload_event_name is not None
    else None
  )
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
      if (
        payload_cfg is not None
        and "payload_range" in stage
        and stage["payload_range"] is not None
      ):
        payload_cfg.params["ranges"] = stage["payload_range"]
  if mode_stages is not None:
    mode_probabilities = cfg.mode_probabilities
    if mode_probabilities is None:
      raise ValueError("mode_stages requires categorical velocity-command sampling.")
    for stage in mode_stages:
      if env.common_step_counter >= stage["step"]:
        mode_probabilities = UniformVelocityCommandCfg.ModeProbabilities(
          standing=stage.get("standing", mode_probabilities.standing),
          forward=stage.get("forward", mode_probabilities.forward),
          backward=stage.get("backward", mode_probabilities.backward),
          lateral=stage.get("lateral", mode_probabilities.lateral),
          yaw=stage.get("yaw", mode_probabilities.yaw),
          mixed=stage.get("mixed", mode_probabilities.mixed),
          min_linear_speed=stage.get(
            "min_linear_speed", mode_probabilities.min_linear_speed
          ),
          min_angular_speed=stage.get(
            "min_angular_speed", mode_probabilities.min_angular_speed
          ),
        )
    cfg.mode_probabilities = mode_probabilities
  cfg.validate()
  state = {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }
  if cfg.mode_probabilities is not None:
    for name, value in zip(
      ("standing", "forward", "backward", "lateral", "yaw", "mixed"),
      cfg.mode_probabilities.values(),
      strict=True,
    ):
      state[f"{name}_prob"] = torch.tensor(value)
    state["min_linear_speed"] = torch.tensor(cfg.mode_probabilities.min_linear_speed)
    state["min_angular_speed"] = torch.tensor(cfg.mode_probabilities.min_angular_speed)
  if payload_cfg is not None:
    payload_range = cast(tuple[float, float], payload_cfg.params["ranges"])
    state["payload_min"] = torch.tensor(payload_range[0])
    state["payload_max"] = torch.tensor(payload_range[1])
  return state
