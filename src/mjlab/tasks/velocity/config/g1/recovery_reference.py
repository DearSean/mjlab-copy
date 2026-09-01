"""Boundary-safe 38-D G1 reference commands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mjlab.tasks.velocity.recovery_data.g1_schema import (
  G1_PHYSICAL_INIT_SCHEMA_VERSION,
)


class G1ReferenceLibrary:
  """Keep clip boundaries so reference windows never cross a clip."""

  def __init__(
    self,
    dataset_dir: Path,
    device: torch.device | str,
    temporal_bin_duration_s: float = 0.4,
    physical_init_file: Path | None = None,
    require_physical_init: bool = False,
  ) -> None:
    if temporal_bin_duration_s <= 0.0:
      raise ValueError("temporal_bin_duration_s must be positive.")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    clips = [clip for clip in manifest["clips"] if clip["split"] == "train"]
    self.clip_ids = tuple(str(clip["clip_id"]) for clip in clips)
    self.clip_metadata = tuple(clips)
    arrays = [np.load(dataset_dir / clip["file"]) for clip in clips]
    self.lengths = torch.tensor(
      [len(item["joint_position"]) for item in arrays], device=device
    )
    maximum = int(self.lengths.max())
    self.root_position = torch.zeros((len(arrays), maximum, 3), device=device)
    self.quaternion = torch.zeros((len(arrays), maximum, 4), device=device)
    self.linear_velocity = torch.zeros((len(arrays), maximum, 3), device=device)
    self.angular_velocity = torch.zeros((len(arrays), maximum, 3), device=device)
    self.joint_position = torch.zeros((len(arrays), maximum, 29), device=device)
    self.joint_velocity = torch.zeros((len(arrays), maximum, 29), device=device)
    for index, item in enumerate(arrays):
      length = int(self.lengths[index])
      self.root_position[index, :length] = torch.as_tensor(
        item["root_position"], device=device
      )
      self.quaternion[index, :length] = torch.as_tensor(
        item["root_quaternion_xyzw"], device=device
      )
      self.linear_velocity[index, :length] = torch.as_tensor(
        item["root_linear_velocity"], device=device
      )
      self.angular_velocity[index, :length] = torch.as_tensor(
        item["root_angular_velocity"], device=device
      )
      self.joint_position[index, :length] = torch.as_tensor(
        item["joint_position"], device=device
      )
      self.joint_velocity[index, :length] = torch.as_tensor(
        item["joint_velocity"], device=device
      )

    # Preserve the full, unmodified trajectories above for SMP/reference
    # consumers. Reset sampling uses a separate simulator-validated state bank.
    self.reset_root_position = self.root_position.clone()
    self.reset_root_position[..., :2] = 0.0
    self.reset_quaternion = self.quaternion.clone()
    self.reset_linear_velocity = self.linear_velocity.clone()
    self.reset_angular_velocity = self.angular_velocity.clone()
    self.reset_joint_position = self.joint_position.clone()
    self.reset_joint_velocity = self.joint_velocity.clone()
    self.reset_valid = torch.zeros(
      (len(arrays), maximum), dtype=torch.bool, device=device
    )
    physical_path = physical_init_file or dataset_dir / "physical_init.npz"
    if physical_path.is_file():
      self._load_physical_init(physical_path, device)
    elif require_physical_init:
      raise FileNotFoundError(
        "The G1 recovery task requires a simulator-validated reset bank: "
        f"{physical_path}. Run build_g1_physical_init first."
      )
    else:
      for index, length in enumerate(self.lengths.tolist()):
        self.reset_valid[index, : int(length)] = True

    # Fallen reset is deliberately broad: it accepts every low-progress frame
    # instead of trying to certify collision-perfect poses offline. PPO contact
    # costs are responsible for rejecting undesirable ways of using them.
    clip_grid = torch.arange(len(arrays), device=device)[:, None].expand(-1, maximum)
    frame_grid = torch.arange(maximum, device=device)[None, :].expand(len(arrays), -1)
    valid = (frame_grid < self.lengths[:, None]) & self.reset_valid
    x, y = self.reset_quaternion[..., 0], self.reset_quaternion[..., 1]
    uprightness = (1.0 - 2.0 * (x.square() + y.square())).clamp(0.0, 1.0)
    progress = (self.reset_root_position[..., 2] / 0.76).clamp(0.0, 1.0) * uprightness
    self._valid_clip = clip_grid[valid]
    self._valid_frame = frame_grid[valid]
    self._valid_progress = progress[valid]
    target_fps = torch.tensor(
      [float(clip["target_fps"]) for clip in clips],
      device=device,
      dtype=torch.float32,
    )
    bin_sizes = (target_fps * temporal_bin_duration_s).round().long().clamp_min(1)
    bin_counts = torch.div(
      self.lengths + bin_sizes - 1, bin_sizes, rounding_mode="floor"
    )
    bin_offsets = torch.cat(
      (
        torch.zeros(1, dtype=torch.long, device=device),
        torch.cumsum(bin_counts, dim=0)[:-1],
      )
    )
    self._valid_bin = (
      bin_offsets[self._valid_clip] + self._valid_frame // bin_sizes[self._valid_clip]
    )
    self._temporal_bin_lookup = torch.full(
      (len(arrays), maximum), -1, dtype=torch.long, device=device
    )
    self._temporal_bin_lookup[self._valid_clip, self._valid_frame] = self._valid_bin
    self.num_temporal_bins = int(bin_counts.sum().item())
    self.bin_clip = torch.repeat_interleave(
      torch.arange(len(clips), device=device), bin_counts
    )
    self.bin_temporal_index = torch.cat(
      [torch.arange(int(count), device=device) for count in bin_counts]
    )
    self.bin_start_frame = self.bin_temporal_index * bin_sizes[self.bin_clip]
    self.bin_end_frame = (self.bin_start_frame + bin_sizes[self.bin_clip]).minimum(
      self.lengths[self.bin_clip]
    )
    self.bin_start_time_s = self.bin_start_frame.float() / target_fps[self.bin_clip]
    self.bin_end_time_s = self.bin_end_frame.float() / target_fps[self.bin_clip]
    source_fps = torch.tensor(
      [float(clip["source_fps"]) for clip in clips],
      device=device,
      dtype=torch.float32,
    )
    source_start = torch.tensor(
      [int(clip["source_support_start_frame"]) for clip in clips],
      device=device,
      dtype=torch.long,
    )
    self.bin_source_start_frame = (
      source_start[self.bin_clip]
      + (self.bin_start_time_s * source_fps[self.bin_clip]).round().long()
    )
    self.bin_source_end_frame = (
      source_start[self.bin_clip]
      + (self.bin_end_time_s * source_fps[self.bin_clip]).round().long()
    )

  def _load_physical_init(
    self, physical_path: Path, device: torch.device | str
  ) -> None:
    with np.load(physical_path, allow_pickle=False) as states:
      schema = str(np.asarray(states["schema_version"]).item())
      if schema != G1_PHYSICAL_INIT_SCHEMA_VERSION:
        raise ValueError(
          f"Unsupported G1 physical initialization schema {schema!r}; "
          f"expected {G1_PHYSICAL_INIT_SCHEMA_VERSION!r}."
        )
      required = (
        "clip_id",
        "frame",
        "root_position",
        "root_quaternion_xyzw",
        "joint_position",
        "root_linear_velocity",
        "root_angular_velocity",
        "joint_velocity",
      )
      missing = [name for name in required if name not in states]
      if missing:
        raise ValueError(
          f"G1 physical initialization bank is missing: {', '.join(missing)}"
        )
      clip_ids = np.asarray(states["clip_id"])
      frames = np.asarray(states["frame"], dtype=np.int64)
      count = len(frames)
      expected_shapes = {
        "clip_id": (count,),
        "root_position": (count, 3),
        "root_quaternion_xyzw": (count, 4),
        "joint_position": (count, 29),
        "root_linear_velocity": (count, 3),
        "root_angular_velocity": (count, 3),
        "joint_velocity": (count, 29),
      }
      for name, shape in expected_shapes.items():
        if states[name].shape != shape:
          raise ValueError(
            f"Physical initialization {name} must have shape {shape}, "
            f"got {states[name].shape}."
          )
      numeric_names = tuple(name for name in required if name != "clip_id")
      if any(not np.all(np.isfinite(states[name])) for name in numeric_names):
        raise ValueError("Physical initialization contains non-finite values.")
      quaternion = np.asarray(states["root_quaternion_xyzw"])
      quaternion_norm = np.linalg.norm(quaternion, axis=-1)
      if np.any(np.abs(quaternion_norm - 1.0) > 1e-3):
        raise ValueError("Physical initialization contains invalid quaternions.")
      clip_lookup = {clip_id: index for index, clip_id in enumerate(self.clip_ids)}
      mapped_clip = np.asarray(
        [clip_lookup.get(str(clip_id), -1) for clip_id in clip_ids], dtype=np.int64
      )
      if np.any(mapped_clip < 0):
        unknown = sorted(
          {str(clip_ids[index]) for index in np.flatnonzero(mapped_clip < 0)}
        )
        raise ValueError(f"Physical initialization contains unknown clips: {unknown}")
      lengths = self.lengths.detach().cpu().numpy()
      if np.any(frames < 0) or np.any(frames >= lengths[mapped_clip]):
        raise ValueError("Physical initialization contains out-of-range frames.")
      pairs = np.stack((mapped_clip, frames), axis=-1)
      if len(np.unique(pairs, axis=0)) != count:
        raise ValueError("Physical initialization contains duplicate clip/frame rows.")
      clip_tensor = torch.as_tensor(mapped_clip, device=device)
      frame_tensor = torch.as_tensor(frames, device=device)
      self.reset_valid[clip_tensor, frame_tensor] = True
      mappings = (
        (self.reset_root_position, "root_position"),
        (self.reset_quaternion, "root_quaternion_xyzw"),
        (self.reset_linear_velocity, "root_linear_velocity"),
        (self.reset_angular_velocity, "root_angular_velocity"),
        (self.reset_joint_position, "joint_position"),
        (self.reset_joint_velocity, "joint_velocity"),
      )
      for target, name in mappings:
        target[clip_tensor, frame_tensor] = torch.as_tensor(
          np.asarray(states[name]), device=device, dtype=target.dtype
        )

  def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    clip_id = torch.randint(len(self.lengths), (count,), device=self.lengths.device)
    frame = (
      torch.rand(count, device=self.lengths.device) * self.lengths[clip_id]
    ).long()
    return clip_id, frame

  def sample_progress(
    self,
    count: int,
    minimum: float,
    maximum: float,
    bin_difficulty: torch.Tensor | None = None,
    uniform_probability: float = 0.2,
    maximum_probability_ratio: float | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample frames inside a normalized height-uprightness progress band."""
    if not 0.0 <= minimum < maximum <= 1.0:
      raise ValueError("progress bounds must satisfy 0 <= minimum < maximum <= 1")
    eligible = (self._valid_progress >= minimum) & (self._valid_progress < maximum)
    eligible_rows = eligible.nonzero(as_tuple=False).squeeze(-1)
    if len(eligible_rows) == 0:
      raise ValueError(
        f"The G1 recovery dataset has no frames in progress [{minimum}, {maximum})."
      )
    if bin_difficulty is None:
      selected = eligible_rows[
        torch.randint(len(eligible_rows), (count,), device=self.lengths.device)
      ]
    else:
      selected = self._sample_adaptive_rows(
        eligible_rows,
        count,
        bin_difficulty,
        uniform_probability,
        maximum_probability_ratio,
      )
    return self._valid_clip[selected], self._valid_frame[selected]

  def _sample_adaptive_rows(
    self,
    eligible_rows: torch.Tensor,
    count: int,
    bin_difficulty: torch.Tensor,
    uniform_probability: float,
    maximum_probability_ratio: float | None,
  ) -> torch.Tensor:
    """Sample temporal bins by difficulty, then frames uniformly inside them."""
    if bin_difficulty.shape != (self.num_temporal_bins,):
      raise ValueError("bin_difficulty must contain one value for every temporal bin.")
    if not 0.0 <= uniform_probability <= 1.0:
      raise ValueError("uniform_probability must be between zero and one.")
    if maximum_probability_ratio is not None and maximum_probability_ratio < 1.0:
      raise ValueError("maximum_probability_ratio must be at least one.")
    eligible_bins, inverse = torch.unique(
      self._valid_bin[eligible_rows], sorted=True, return_inverse=True
    )
    frames_per_bin = torch.bincount(inverse, minlength=len(eligible_bins)).float()
    difficulty = bin_difficulty[eligible_bins].float().clamp_min(0.0)
    if float(difficulty.sum().item()) <= 0.0:
      bin_probability = torch.full_like(difficulty, 1.0 / len(difficulty))
    else:
      adaptive = difficulty / difficulty.sum()
      uniform = torch.full_like(adaptive, 1.0 / len(adaptive))
      bin_probability = (
        1.0 - uniform_probability
      ) * adaptive + uniform_probability * uniform
    if maximum_probability_ratio is not None:
      probability_cap = maximum_probability_ratio / len(bin_probability)
      clipped = bin_probability.clamp_max(probability_cap)
      excess = 1.0 - clipped.sum()
      capacity = (probability_cap - clipped).clamp_min(0.0)
      bin_probability = clipped + excess * capacity / capacity.sum().clamp_min(1e-8)
    frame_probability = bin_probability[inverse] / frames_per_bin[inverse]
    sampled = torch.multinomial(frame_probability, count, replacement=True)
    return eligible_rows[sampled]

  def temporal_bins_in_progress(self, minimum: float, maximum: float) -> torch.Tensor:
    """Return bins containing at least one frame in a progress interval."""
    eligible = (self._valid_progress >= minimum) & (self._valid_progress < maximum)
    return torch.unique(self._valid_bin[eligible], sorted=True)

  def sample_frontier_balanced(
    self,
    count: int,
    minimum: float,
    frontier_maximum: float,
    maximum: float,
    frontier_probability: float,
    bin_difficulty: torch.Tensor | None = None,
    uniform_probability: float = 0.2,
    maximum_probability_ratio: float | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix newly opened progress with the previously mastered interval."""
    if not minimum < frontier_maximum < maximum:
      raise ValueError(
        "frontier bounds must satisfy minimum < frontier_maximum < maximum"
      )
    if not 0.0 <= frontier_probability <= 1.0:
      raise ValueError("frontier_probability must be between zero and one.")
    frontier_rows = torch.rand(count, device=self.lengths.device) < frontier_probability
    clip = torch.empty(count, dtype=torch.long, device=self.lengths.device)
    frame = torch.empty_like(clip)
    frontier_count = int(frontier_rows.sum().item())
    if frontier_count > 0:
      frontier_clip, frontier_frame = self.sample_progress(
        frontier_count,
        minimum,
        frontier_maximum,
        bin_difficulty,
        uniform_probability,
        maximum_probability_ratio,
      )
      clip[frontier_rows] = frontier_clip
      frame[frontier_rows] = frontier_frame
    mastered_count = count - frontier_count
    if mastered_count > 0:
      mastered_clip, mastered_frame = self.sample_progress(
        mastered_count,
        frontier_maximum,
        maximum,
        bin_difficulty,
        uniform_probability,
        maximum_probability_ratio,
      )
      clip[~frontier_rows] = mastered_clip
      frame[~frontier_rows] = mastered_frame
    return clip, frame

  def sample_fallen(
    self,
    count: int,
    maximum_progress: float = 0.25,
    bin_difficulty: torch.Tensor | None = None,
    uniform_probability: float = 0.2,
    maximum_probability_ratio: float | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample genuinely low-progress states for assisted full recovery."""
    return self.sample_progress(
      count,
      0.0,
      maximum_progress,
      bin_difficulty,
      uniform_probability,
      maximum_probability_ratio,
    )

  def temporal_bin(self, clip_id: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Map sampled clip-local frames to global temporal-bin identifiers."""
    return self._temporal_bin_lookup[clip_id, frame]

  def confidence(self, clip_id: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Smoothly fade a reference into the default pose over its last 0.4 s."""
    remaining = (self.lengths[clip_id] - 1 - frame).clamp_min(0)
    ratio = (remaining.float() / 20.0).clamp(0.0, 1.0)
    return ratio.square() * (3.0 - 2.0 * ratio)

  def window(
    self, clip_id: torch.Tensor, frame: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(-10, 11, device=frame.device)
    indices = frame[:, None] + offsets
    valid = (indices >= 0) & (indices < self.lengths[clip_id, None])
    indices = indices.clamp_min(0).minimum(self.lengths[clip_id, None] - 1)
    quaternion = self.quaternion[clip_id[:, None], indices]
    x, y, z, w = quaternion.unbind(-1)
    rotation = (
      torch.stack(
        (
          1 - 2 * (y * y + z * z),
          2 * (x * y - z * w),
          2 * (x * z + y * w),
          2 * (x * y + z * w),
          1 - 2 * (x * x + z * z),
          2 * (y * z - x * w),
          2 * (x * z - y * w),
          2 * (y * z + x * w),
          1 - 2 * (x * x + y * y),
        ),
        dim=-1,
      )
      .reshape(*quaternion.shape[:-1], 3, 3)
      .transpose(-1, -2)
    )
    linear = torch.einsum(
      "...ij,...j->...i", rotation, self.linear_velocity[clip_id[:, None], indices]
    )
    angular = torch.einsum(
      "...ij,...j->...i", rotation, self.angular_velocity[clip_id[:, None], indices]
    )
    gravity = torch.einsum(
      "...ij,j->...i", rotation, torch.tensor((0.0, 0.0, -1.0), device=frame.device)
    )
    command = torch.cat(
      (linear, angular, gravity, self.joint_position[clip_id[:, None], indices]), dim=-1
    )
    return command * valid[:, :, None], valid
