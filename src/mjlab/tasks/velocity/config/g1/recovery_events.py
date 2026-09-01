"""Episode state, three-mode reset, and assistance for G1 recovery."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .recovery_reference import G1ReferenceLibrary

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


REFERENCE_MODE = 0
FALLEN_MODE = 1
STAND_MODE = 2


class G1RecoveryReset:
  """Own all state shared by recovery observations, rewards, and termination."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
    params = cfg.params
    self._env = env
    self._asset: Entity = env.scene[params["asset_cfg"].name]
    self._asset_cfg = params["asset_cfg"]
    self._force_ranges = torch.tensor(
      params["force_ranges"], device=env.device, dtype=torch.float32
    )
    self._assist_success_windows = tuple(
      int(value) for value in params["assist_success_windows"]
    )
    self._posture_mode_probabilities = tuple(
      tuple(float(value) for value in probabilities)
      for probabilities in params["posture_mode_probabilities"]
    )
    self._posture_reference_min_progress = tuple(
      float(value) for value in params["posture_reference_min_progress"]
    )
    self._posture_success_windows = tuple(
      int(value) for value in params["posture_success_windows"]
    )
    self._reference_frontier_probability = float(
      params["reference_frontier_probability"]
    )
    self._adaptive_bin_duration_s = float(params["adaptive_bin_duration_s"])
    self._adaptive_ema_rate = float(params["adaptive_ema_rate"])
    self._adaptive_uniform_probability = float(params["adaptive_uniform_probability"])
    self._adaptive_max_difficulty = float(params["adaptive_max_difficulty"])
    self._adaptive_max_probability_ratio = float(
      params["adaptive_max_probability_ratio"]
    )
    self._curriculum_probe_probability = float(params["curriculum_probe_probability"])
    self._hard_bin_report_count = int(params["hard_bin_report_count"])
    self._hard_bin_min_attempts = int(params["hard_bin_min_attempts"])
    self._hard_bin_report_interval = int(params["hard_bin_report_interval"])
    self._reference_max_progress = float(params["reference_max_progress"])
    self._fallen_max_progress = float(params["fallen_max_progress"])
    self.posture_level = int(params["initial_posture_level"])
    self.assist_level = int(params["initial_assist_level"])
    if self._force_ranges.ndim != 2 or self._force_ranges.shape[1] != 2:
      raise ValueError("force_ranges must contain (minimum, maximum) pairs.")
    if len(self._assist_success_windows) != len(self._force_ranges) - 1:
      raise ValueError("assist windows must define every force-level transition.")
    if len(self._posture_success_windows) != len(self._posture_mode_probabilities) - 1:
      raise ValueError("posture windows must define every posture-level transition.")
    success_windows = self._posture_success_windows + self._assist_success_windows
    if any(window <= 0 for window in success_windows):
      raise ValueError("success_windows must be positive.")
    if any(
      later <= earlier
      for earlier, later in zip(success_windows, success_windows[1:], strict=False)
    ):
      raise ValueError("success windows must grow across both curriculum stages.")
    if not 0 <= self.posture_level < len(self._posture_mode_probabilities):
      raise ValueError("initial_posture_level is outside the posture curriculum.")
    if not 0 <= self.assist_level < len(self._force_ranges):
      raise ValueError("initial_assist_level is outside force_ranges.")
    if len(self._posture_reference_min_progress) != len(
      self._posture_mode_probabilities
    ):
      raise ValueError("reference progress must define every posture level.")
    for probabilities in self._posture_mode_probabilities:
      if len(probabilities) != 3:
        raise ValueError("mode probabilities must contain reference, fallen, stand.")
      if any(probability < 0.0 for probability in probabilities):
        raise ValueError("mode probabilities must be non-negative.")
      if abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError("mode probabilities must sum to one.")
    if any(
      not 0.0 <= progress < self._reference_max_progress
      for progress in self._posture_reference_min_progress
    ):
      raise ValueError("reference progress curriculum bounds are invalid.")
    if any(
      later > earlier
      for earlier, later in zip(
        self._posture_reference_min_progress,
        self._posture_reference_min_progress[1:],
        strict=False,
      )
    ):
      raise ValueError("reference progress must not become easier at later levels.")
    if not 0.0 <= self._reference_frontier_probability <= 1.0:
      raise ValueError("reference_frontier_probability must be between zero and one.")
    if self._adaptive_bin_duration_s <= 0.0:
      raise ValueError("adaptive_bin_duration_s must be positive.")
    if not 0.0 < self._adaptive_ema_rate <= 1.0:
      raise ValueError("adaptive_ema_rate must be in (0, 1].")
    if not 0.0 <= self._adaptive_uniform_probability <= 1.0:
      raise ValueError("adaptive_uniform_probability must be between zero and one.")
    if not 0.0 < self._adaptive_max_difficulty <= 1.0:
      raise ValueError("adaptive_max_difficulty must be in (0, 1].")
    if self._adaptive_max_probability_ratio < 1.0:
      raise ValueError("adaptive_max_probability_ratio must be at least one.")
    if not 0.0 < self._curriculum_probe_probability <= 1.0:
      raise ValueError("curriculum_probe_probability must be in (0, 1].")
    if self._hard_bin_report_count <= 0:
      raise ValueError("hard_bin_report_count must be positive.")
    if self._hard_bin_min_attempts <= 0:
      raise ValueError("hard_bin_min_attempts must be positive.")
    if self._hard_bin_report_interval <= 0:
      raise ValueError("hard_bin_report_interval must be positive.")
    if not 0.0 < self._fallen_max_progress < self._reference_max_progress:
      raise ValueError("fallen progress must remain below the initial reference band.")
    if bool((self._force_ranges < 0.0).any()) or bool(
      (self._force_ranges[:, 0] > self._force_ranges[:, 1]).any()
    ):
      raise ValueError("force_ranges must be non-negative ordered pairs.")
    if any(
      abs(float(current[0]) - float(following[1])) > 1e-6
      for current, following in zip(
        params["force_ranges"][:-1],
        params["force_ranges"][1:],
        strict=True,
      )
    ):
      raise ValueError("Adjacent force_ranges must meet at the same boundary.")
    self._max_assist_force = float(self._force_ranges[:, 1].max().item())
    self._body_ids = self._asset_cfg.body_ids
    self.mode = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.reference = G1ReferenceLibrary(
      Path(params["dataset_dir"]),
      env.device,
      self._adaptive_bin_duration_s,
      physical_init_file=Path(params["physical_init_file"]),
      require_physical_init=bool(params["require_physical_init"]),
    )
    self.sampled_force = torch.zeros(env.num_envs, device=env.device)
    self.applied_force = torch.zeros_like(self.sampled_force)
    self.previous_progress = torch.zeros_like(self.sampled_force)
    self.current_progress = torch.zeros_like(self.sampled_force)
    self.reset_progress = torch.zeros_like(self.sampled_force)
    self.reset_clip = torch.full_like(self.mode, -1)
    self.reset_frame = torch.full_like(self.mode, -1)
    self.reset_temporal_bin = torch.full_like(self.mode, -1)
    self.curriculum_probe = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    self.hold_count = torch.zeros_like(self.mode)
    self.succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.just_succeeded = torch.zeros_like(self.succeeded)
    self.episode_started = torch.zeros_like(self.succeeded)
    self.attempts = torch.zeros((), dtype=torch.long, device=env.device)
    self.successes = torch.zeros_like(self.attempts)
    self.last_window_attempts = torch.zeros_like(self.attempts)
    self.last_success_rate = torch.zeros((), device=env.device)
    self.training_attempts = torch.zeros_like(self.attempts)
    self.training_successes = torch.zeros_like(self.attempts)
    self.last_training_success_rate = torch.zeros((), device=env.device)
    self.bin_failure_ema = torch.ones(
      self.reference.num_temporal_bins, device=env.device
    )
    self.bin_attempts = torch.zeros(
      self.reference.num_temporal_bins, dtype=torch.long, device=env.device
    )
    self.bin_successes = torch.zeros_like(self.bin_attempts)
    self._adaptive_total_attempts = 0
    self._adaptive_attempts_since_report = 0
    self._hard_report_version = 0
    self._hard_report_posture_level = -1
    self._reported_hard_bins = torch.full(
      (self._hard_bin_report_count,), -1, dtype=torch.long, device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    dataset_dir: str,
    physical_init_file: str,
    require_physical_init: bool,
    force_ranges: tuple[tuple[float, float], ...],
    assist_success_windows: tuple[int, ...],
    posture_mode_probabilities: tuple[tuple[float, float, float], ...],
    posture_reference_min_progress: tuple[float, ...],
    posture_success_windows: tuple[int, ...],
    reference_frontier_probability: float,
    adaptive_bin_duration_s: float,
    adaptive_ema_rate: float,
    adaptive_uniform_probability: float,
    adaptive_max_difficulty: float,
    adaptive_max_probability_ratio: float,
    curriculum_probe_probability: float,
    hard_bin_report_count: int,
    hard_bin_min_attempts: int,
    hard_bin_report_interval: int,
    reference_max_progress: float,
    fallen_max_progress: float,
    initial_posture_level: int,
    initial_assist_level: int,
  ) -> None:
    del (
      env,
      asset_cfg,
      dataset_dir,
      physical_init_file,
      require_physical_init,
      force_ranges,
      assist_success_windows,
      posture_mode_probabilities,
      posture_reference_min_progress,
      posture_success_windows,
      reference_frontier_probability,
      adaptive_bin_duration_s,
      adaptive_ema_rate,
      adaptive_uniform_probability,
      adaptive_max_difficulty,
      adaptive_max_probability_ratio,
      curriculum_probe_probability,
      hard_bin_report_count,
      hard_bin_min_attempts,
      hard_bin_report_interval,
      reference_max_progress,
      fallen_max_progress,
      initial_posture_level,
      initial_assist_level,
    )
    probabilities = self.mode_probabilities
    self.mode[env_ids] = torch.multinomial(
      probabilities, len(env_ids), replacement=True
    )
    recovery_mask = self.mode[env_ids] != STAND_MODE
    self.curriculum_probe[env_ids] = recovery_mask & (
      torch.rand(len(env_ids), device=self._env.device)
      < self._curriculum_probe_probability
    )
    clip_id = torch.zeros(len(env_ids), dtype=torch.long, device=self._env.device)
    frame = torch.zeros_like(clip_id)
    reference_rows = (
      (self.mode[env_ids] == REFERENCE_MODE).nonzero(as_tuple=False).squeeze(-1)
    )
    if len(reference_rows) > 0:
      reference_probe = self.curriculum_probe[env_ids[reference_rows]]
      for rows, adaptive in (
        (reference_rows[reference_probe], False),
        (reference_rows[~reference_probe], True),
      ):
        if len(rows) > 0:
          reference_clip, reference_frame = self._sample_reference(len(rows), adaptive)
          clip_id[rows] = reference_clip
          frame[rows] = reference_frame
    fallen_rows = (
      (self.mode[env_ids] == FALLEN_MODE).nonzero(as_tuple=False).squeeze(-1)
    )
    if len(fallen_rows) > 0:
      fallen_probe = self.curriculum_probe[env_ids[fallen_rows]]
      for rows, adaptive in (
        (fallen_rows[fallen_probe], False),
        (fallen_rows[~fallen_probe], True),
      ):
        if len(rows) > 0:
          fallen_clip, fallen_frame = self._sample_fallen(len(rows), adaptive)
          clip_id[rows] = fallen_clip
          frame[rows] = fallen_frame
    reference_mask = self.mode[env_ids] == REFERENCE_MODE
    fallen_mask = self.mode[env_ids] == FALLEN_MODE
    stand_mask = self.mode[env_ids] == STAND_MODE
    root_state = self._asset.data.default_root_state[env_ids].clone()
    joint_pos = self._asset.data.default_joint_pos[env_ids].clone()
    joint_vel = self._asset.data.default_joint_vel[env_ids].clone()
    sampled = reference_mask | fallen_mask
    root_state[sampled, :3] = (
      self.reference.reset_root_position[clip_id[sampled], frame[sampled]]
      + self._env.scene.env_origins[env_ids[sampled]]
    )
    quaternion = self.reference.reset_quaternion[clip_id, frame]
    root_state[sampled, 3:7] = quaternion[sampled][:, (3, 0, 1, 2)]
    root_state[sampled, 7:10] = self.reference.reset_linear_velocity[
      clip_id[sampled], frame[sampled]
    ]
    root_state[sampled, 10:13] = self.reference.reset_angular_velocity[
      clip_id[sampled], frame[sampled]
    ]
    joint_pos[sampled] = self.reference.reset_joint_position[
      clip_id[sampled], frame[sampled]
    ]
    joint_vel[sampled] = self.reference.reset_joint_velocity[
      clip_id[sampled], frame[sampled]
    ]
    if stand_mask.any():
      joint_pos[stand_mask] += torch.empty_like(joint_pos[stand_mask]).uniform_(
        -0.05, 0.05
      )
      root_state[stand_mask, 2] += torch.empty_like(root_state[stand_mask, 2]).uniform_(
        -0.02, 0.02
      )
      root_state[stand_mask, 7:13].uniform_(-0.05, 0.05)

    quaternion = self.reference.reset_quaternion[clip_id, frame]
    uprightness = (
      1.0 - 2.0 * (quaternion[:, 0].square() + quaternion[:, 1].square())
    ).clamp(0.0, 1.0)
    initial_progress = (
      self.reference.reset_root_position[clip_id, frame, 2] / 0.76
    ).clamp(0.0, 1.0) * uprightness
    initial_progress[stand_mask] = 1.0
    self.previous_progress[env_ids] = initial_progress
    self.current_progress[env_ids] = initial_progress
    self.reset_progress[env_ids] = initial_progress
    self.reset_clip[env_ids] = -1
    self.reset_frame[env_ids] = -1
    self.reset_temporal_bin[env_ids] = -1
    self.reset_clip[env_ids[sampled]] = clip_id[sampled]
    self.reset_frame[env_ids[sampled]] = frame[sampled]
    self.reset_temporal_bin[env_ids[sampled]] = self.reference.temporal_bin(
      clip_id[sampled], frame[sampled]
    )
    self.hold_count[env_ids] = 0
    self.succeeded[env_ids] = False
    self.just_succeeded[env_ids] = False
    self.episode_started[env_ids] = False
    self.applied_force[env_ids] = 0.0
    self.sampled_force[env_ids] = 0.0
    force_min, force_max = self.force_range
    recovery_rows = sampled.nonzero(as_tuple=False).squeeze(-1)
    if len(recovery_rows) > 0 and force_max > 0.0:
      recovery_env_ids = env_ids[recovery_rows]
      self.sampled_force[recovery_env_ids] = torch.empty(
        len(recovery_env_ids), device=self._env.device
      ).uniform_(force_min, force_max)
      self.applied_force[recovery_env_ids] = self.sampled_force[recovery_env_ids]
    self._asset.write_root_state_to_sim(root_state, env_ids=env_ids)
    self._asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    self._write_force(env_ids)

  def _sample_reference(
    self, count: int, adaptive: bool
  ) -> tuple[torch.Tensor, torch.Tensor]:
    difficulty = self.adaptive_bin_difficulty if adaptive else None
    probability_ratio = self._adaptive_max_probability_ratio if adaptive else None
    frontier_maximum = self.reference_frontier_max_progress
    if frontier_maximum is None:
      return self.reference.sample_progress(
        count,
        self.reference_min_progress,
        self._reference_max_progress,
        bin_difficulty=difficulty,
        uniform_probability=self._adaptive_uniform_probability,
        maximum_probability_ratio=probability_ratio,
      )
    return self.reference.sample_frontier_balanced(
      count,
      self.reference_min_progress,
      frontier_maximum,
      self._reference_max_progress,
      self._reference_frontier_probability,
      bin_difficulty=difficulty,
      uniform_probability=self._adaptive_uniform_probability,
      maximum_probability_ratio=probability_ratio,
    )

  def _sample_fallen(
    self, count: int, adaptive: bool
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return self.reference.sample_fallen(
      count,
      self._fallen_max_progress,
      bin_difficulty=self.adaptive_bin_difficulty if adaptive else None,
      uniform_probability=self._adaptive_uniform_probability,
      maximum_probability_ratio=(
        self._adaptive_max_probability_ratio if adaptive else None
      ),
    )

  @property
  def force_range(self) -> tuple[float, float]:
    force_range = self._force_ranges[self.assist_level]
    return float(force_range[0].item()), float(force_range[1].item())

  @property
  def mode_probabilities(self) -> torch.Tensor:
    """Return the explicit reset mixture for the posture curriculum."""
    return torch.tensor(
      self._posture_mode_probabilities[self.posture_level], device=self._env.device
    )

  @property
  def reference_min_progress(self) -> float:
    """Lower the autonomous reset band only after demonstrated success."""
    return self._posture_reference_min_progress[self.posture_level]

  @property
  def reference_frontier_max_progress(self) -> float | None:
    """Upper edge of the progress interval newly opened at this level."""
    if self.posture_level == 0:
      return None
    previous = self._posture_reference_min_progress[self.posture_level - 1]
    current = self.reference_min_progress
    return previous if previous > current else None

  @property
  def adaptive_bin_difficulty(self) -> torch.Tensor:
    """Bound failure EMA before turning it into reset sampling probabilities."""
    return self.bin_failure_ema.clamp(0.0, self._adaptive_max_difficulty)

  @property
  def posture_complete(self) -> bool:
    return self.posture_level == len(self._posture_mode_probabilities) - 1

  @property
  def assist_complete(self) -> bool:
    return self.assist_level == len(self._force_ranges) - 1

  @property
  def level(self) -> int:
    """Return a monotonic combined level for existing training dashboards."""
    if not self.posture_complete:
      return self.posture_level
    return len(self._posture_mode_probabilities) - 1 + self.assist_level

  @property
  def required_window(self) -> int:
    if not self.posture_complete:
      return self._posture_success_windows[self.posture_level]
    if self.assist_complete:
      return self._assist_success_windows[-1]
    return self._assist_success_windows[self.assist_level]

  def step(self, dt: float) -> None:
    """Advance reference time and write the next-step curriculum force."""
    del dt
    self.episode_started.fill_(True)
    self.previous_progress.copy_(self.current_progress)
    root_height = self._asset.data.root_link_pos_w[:, 2]
    ground_height = self._env.scene.env_origins[:, 2]
    nominal_height = self._asset.data.default_root_state[:, 2].clamp_min(1e-6)
    height = ((root_height - ground_height) / nominal_height).clamp(0.0, 1.0)
    upright = (-self._asset.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
    self.current_progress.copy_(height * upright)

    active = (self.mode != STAND_MODE) & ~self.succeeded
    self.applied_force.copy_(self.sampled_force * active.float())
    self._write_force()

  def _write_force(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(
        self._env.num_envs, device=self._env.device, dtype=torch.long
      )
    forces = torch.zeros(len(env_ids), 1, 3, device=self._env.device)
    forces[:, 0, 2] = self.applied_force[env_ids]
    self._asset.write_external_wrench_to_sim(
      forces,
      torch.zeros_like(forces),
      env_ids=env_ids,
      body_ids=self._body_ids,
    )

  def record_outcomes(self, env_ids: torch.Tensor) -> None:
    """Count completed non-stand recovery episodes toward curriculum progress."""
    valid = self.episode_started[env_ids] & (self.mode[env_ids] != STAND_MODE)
    recovery_ids = env_ids[valid]
    if len(recovery_ids) == 0:
      return
    self._record_adaptive_outcomes(recovery_ids)
    self.training_attempts += len(recovery_ids)
    self.training_successes += self.succeeded[recovery_ids].sum()
    probe_ids = recovery_ids[self.curriculum_probe[recovery_ids]]
    self.attempts += len(probe_ids)
    self.successes += self.succeeded[probe_ids].sum()

  def _record_adaptive_outcomes(self, recovery_ids: torch.Tensor) -> None:
    temporal_bins = self.reset_temporal_bin[recovery_ids]
    tracked = temporal_bins >= 0
    temporal_bins = temporal_bins[tracked]
    if len(temporal_bins) == 0:
      return
    successes = self.succeeded[recovery_ids[tracked]].float()
    unique_bins, inverse = torch.unique(temporal_bins, sorted=True, return_inverse=True)
    batch_attempts = torch.bincount(inverse, minlength=len(unique_bins))
    batch_successes = torch.zeros(
      len(unique_bins), device=self._env.device, dtype=torch.float32
    )
    batch_successes.scatter_add_(0, inverse, successes)
    observed_failure = 1.0 - batch_successes / batch_attempts.float()
    decay = torch.pow(1.0 - self._adaptive_ema_rate, batch_attempts.float())
    self.bin_failure_ema[unique_bins] = (
      decay * self.bin_failure_ema[unique_bins] + (1.0 - decay) * observed_failure
    )
    self.bin_attempts.scatter_add_(0, unique_bins, batch_attempts)
    self.bin_successes.scatter_add_(0, unique_bins, batch_successes.round().long())
    count = len(temporal_bins)
    self._adaptive_total_attempts += count
    self._adaptive_attempts_since_report += count
    if self._adaptive_attempts_since_report >= self._hard_bin_report_interval:
      self._adaptive_attempts_since_report %= self._hard_bin_report_interval
      self._refresh_hard_bin_report()

  def _active_temporal_bins(self) -> torch.Tensor:
    bins = self.reference.temporal_bins_in_progress(
      self.reference_min_progress, self._reference_max_progress
    )
    if float(self.mode_probabilities[FALLEN_MODE].item()) > 0.0:
      fallen = self.reference.temporal_bins_in_progress(0.0, self._fallen_max_progress)
      bins = torch.unique(torch.cat((bins, fallen)), sorted=True)
    return bins

  def _refresh_hard_bin_report(self) -> None:
    self._reported_hard_bins.fill_(-1)
    active_bins = self._active_temporal_bins()
    experienced = self.bin_attempts[active_bins] >= self._hard_bin_min_attempts
    candidates = active_bins[experienced]
    if len(candidates) > 0:
      count = min(self._hard_bin_report_count, len(candidates))
      ranking = torch.topk(self.bin_failure_ema[candidates], count).indices
      self._reported_hard_bins[:count] = candidates[ranking]
    self._hard_report_version += 1
    self._hard_report_posture_level = self.posture_level

  def update_curriculum(self, success_threshold: float) -> bool:
    """Advance at most one level after the active evidence window completes."""
    if not 0.0 <= success_threshold <= 1.0:
      raise ValueError("success_threshold must be between zero and one.")
    if int(self.attempts.item()) < self.required_window:
      return False
    self.last_window_attempts.copy_(self.attempts)
    self.last_success_rate.copy_(self.successes.float() / self.attempts.clamp_min(1))
    self.last_training_success_rate.copy_(
      self.training_successes.float() / self.training_attempts.clamp_min(1)
    )
    advanced = False
    if float(self.last_success_rate.item()) >= success_threshold:
      if not self.posture_complete:
        self.posture_level += 1
        advanced = True
      elif not self.assist_complete:
        self.assist_level += 1
        advanced = True
    self.attempts.zero_()
    self.successes.zero_()
    self.training_attempts.zero_()
    self.training_successes.zero_()
    return advanced

  def curriculum_state(self) -> dict[str, torch.Tensor]:
    force_min, force_max = self.force_range
    probabilities = self.mode_probabilities
    current_rate = self.successes.float() / self.attempts.clamp_min(1)
    current_training_rate = (
      self.training_successes.float() / self.training_attempts.clamp_min(1)
    )
    frontier_maximum = self.reference_frontier_max_progress
    state = {
      "level": torch.tensor(self.level, device=self._env.device),
      "phase": torch.tensor(int(self.posture_complete), device=self._env.device),
      "posture_level": torch.tensor(self.posture_level, device=self._env.device),
      "assist_level": torch.tensor(self.assist_level, device=self._env.device),
      "force_min_n": torch.tensor(force_min, device=self._env.device),
      "force_max_n": torch.tensor(force_max, device=self._env.device),
      "attempts": self.attempts,
      "required_window": torch.tensor(self.required_window, device=self._env.device),
      "current_success_rate": current_rate,
      "last_window_attempts": self.last_window_attempts,
      "last_success_rate": self.last_success_rate,
      "curriculum_probe_probability": torch.tensor(
        self._curriculum_probe_probability, device=self._env.device
      ),
      "training_attempts": self.training_attempts,
      "current_training_success_rate": current_training_rate,
      "last_training_success_rate": self.last_training_success_rate,
      "reference_probability": probabilities[REFERENCE_MODE],
      "fallen_probability": probabilities[FALLEN_MODE],
      "stand_probability": probabilities[STAND_MODE],
      "reference_min_progress": torch.tensor(
        self.reference_min_progress, device=self._env.device
      ),
      "reference_frontier_max_progress": torch.tensor(
        -1.0 if frontier_maximum is None else frontier_maximum,
        device=self._env.device,
      ),
      "reference_frontier_probability": torch.tensor(
        self._reference_frontier_probability, device=self._env.device
      ),
      "adaptive_total_attempts": torch.tensor(
        self._adaptive_total_attempts, device=self._env.device
      ),
      "adaptive_failure_ema_mean": self.bin_failure_ema.mean(),
      "adaptive_failure_ema_max": self.bin_failure_ema.max(),
      "hard_report_version": torch.tensor(
        self._hard_report_version, device=self._env.device
      ),
      "hard_report_posture_level": torch.tensor(
        self._hard_report_posture_level, device=self._env.device
      ),
    }
    for rank, temporal_bin in enumerate(self._reported_hard_bins):
      valid = temporal_bin >= 0
      safe_bin = temporal_bin.clamp_min(0)
      attempts = self.bin_attempts[safe_bin]
      successes = self.bin_successes[safe_bin]
      success_rate = torch.where(
        valid & (attempts > 0),
        successes.float() / attempts.clamp_min(1),
        torch.full((), -1.0, device=self._env.device),
      )
      prefix = f"hard_bin_{rank}"
      state[f"{prefix}_global_id"] = temporal_bin
      state[f"{prefix}_clip_index"] = torch.where(
        valid, self.reference.bin_clip[safe_bin], temporal_bin
      )
      state[f"{prefix}_temporal_index"] = torch.where(
        valid, self.reference.bin_temporal_index[safe_bin], temporal_bin
      )
      state[f"{prefix}_source_start_frame"] = torch.where(
        valid, self.reference.bin_source_start_frame[safe_bin], temporal_bin
      )
      state[f"{prefix}_source_end_frame"] = torch.where(
        valid, self.reference.bin_source_end_frame[safe_bin], temporal_bin
      )
      state[f"{prefix}_attempts"] = torch.where(valid, attempts, temporal_bin)
      state[f"{prefix}_success_rate"] = success_rate
      state[f"{prefix}_failure_ema"] = torch.where(
        valid,
        self.bin_failure_ema[safe_bin],
        temporal_bin.float(),
      )
    return state


def get_g1_recovery_state(env: ManagerBasedRlEnv, event_name: str) -> G1RecoveryReset:
  """Resolve the reset term instance shared across manager terms."""
  term = env.event_manager.get_term_cfg(event_name).func
  if not isinstance(term, G1RecoveryReset):
    raise TypeError(f"Event '{event_name}' is not a G1RecoveryReset.")
  return term


def step_g1_recovery_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  event_name: str,
) -> None:
  del env_ids
  get_g1_recovery_state(env, event_name).step(env.step_dt)


def g1_recovery_assist_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  event_name: str,
  success_threshold: float,
) -> dict[str, torch.Tensor]:
  """Advance difficulty after enough recovery attempts meet the target rate."""
  if isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  state = get_g1_recovery_state(env, event_name)
  state.record_outcomes(env_ids)
  state.update_curriculum(success_threshold)
  return state.curriculum_state()
