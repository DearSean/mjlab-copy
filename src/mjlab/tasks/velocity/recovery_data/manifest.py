"""Reproducible LaFAN inventory and recovery-segment extraction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mjlab.tasks.velocity.recovery_data.bvh import load_lafan_bvh
from mjlab.tasks.velocity.recovery_data.schema import (
  RECOVERY_SEMANTIC_SCHEMA_VERSION,
  RecoverySegment,
)
from mjlab.tasks.velocity.recovery_data.semantic import (
  RecoverySemanticEncoder,
  classify_initial_posture,
)

_RECOVERY_RECORDING_PREFIXES = (
  "fallAndGetUp",
  "ground",
  "pushAndFall",
  "pushAndStumble",
  "multipleActions",
)


@dataclass(frozen=True, kw_only=True)
class RecoverySegmenterCfg:
  """Thresholds for detecting low-to-stable-upright intervals."""

  low_progress: float = 0.35
  ready_progress: float = 0.85
  ready_hold_s: float = 0.5
  pre_roll_s: float = 0.25
  min_duration_s: float = 1.0
  max_duration_s: float = 8.0

  def __post_init__(self) -> None:
    if not 0.0 <= self.low_progress < self.ready_progress <= 1.0:
      raise ValueError("Progress thresholds must satisfy 0 <= low < ready <= 1.")
    if self.ready_hold_s <= 0.0 or self.pre_roll_s < 0.0:
      raise ValueError("Hold time must be positive and pre-roll non-negative.")
    if not 0.0 < self.min_duration_s <= self.max_duration_s:
      raise ValueError("Segment duration bounds are invalid.")


@dataclass(frozen=True)
class ManifestClip:
  """Serializable metadata for one source BVH."""

  relative_path: str
  sha256: str
  recording: str
  subject: str
  split: str
  frame_count: int
  fps: float
  duration_s: float
  recovery_source: bool
  segments: tuple[RecoverySegment, ...]


@dataclass(frozen=True)
class ManifestFailure:
  """A source file rejected while building a non-strict manifest."""

  relative_path: str
  error: str


@dataclass(frozen=True)
class RecoveryManifest:
  """Versioned, deterministic inventory of LaFAN and its recovery segments."""

  schema_version: str
  dataset_name: str
  clips: tuple[ManifestClip, ...]
  failures: tuple[ManifestFailure, ...] = ()

  @property
  def frame_count(self) -> int:
    return sum(clip.frame_count for clip in self.clips)

  @property
  def segment_count(self) -> int:
    return sum(len(clip.segments) for clip in self.clips)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


def build_recovery_manifest(
  dataset_root: str | Path,
  *,
  strict: bool = True,
  segmenter_cfg: RecoverySegmenterCfg | None = None,
) -> RecoveryManifest:
  """Build a deterministic manifest without loading the full corpus at once."""
  root = Path(dataset_root).resolve()
  if not root.is_dir():
    raise FileNotFoundError(f"LaFAN dataset directory does not exist: {root}")
  paths = sorted(root.glob("*.bvh"))
  if not paths:
    raise FileNotFoundError(f"No .bvh files found under {root}")

  encoder = RecoverySemanticEncoder()
  cfg = segmenter_cfg or RecoverySegmenterCfg()
  clips: list[ManifestClip] = []
  failures: list[ManifestFailure] = []
  for path in paths:
    relative_path = path.relative_to(root).as_posix()
    try:
      recording, subject = _parse_source_identity(path.stem)
      motion = load_lafan_bvh(path)
      recovery_source = recording.startswith(_RECOVERY_RECORDING_PREFIXES)
      if recovery_source:
        semantic = encoder.encode(motion)
        segments = extract_recovery_segments(motion, semantic.progress, cfg)
      else:
        segments = ()
      clips.append(
        ManifestClip(
          relative_path=relative_path,
          sha256=_sha256(path),
          recording=recording,
          subject=subject,
          split=_split_for_recording(recording),
          frame_count=motion.frame_count,
          fps=motion.fps,
          duration_s=motion.duration_s,
          recovery_source=recovery_source,
          segments=segments,
        )
      )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
      if strict:
        raise RuntimeError(f"Failed to process LaFAN file {path}: {exc}") from exc
      failures.append(ManifestFailure(relative_path=relative_path, error=str(exc)))

  return RecoveryManifest(
    schema_version=RECOVERY_SEMANTIC_SCHEMA_VERSION,
    dataset_name=root.name,
    clips=tuple(clips),
    failures=tuple(failures),
  )


def extract_recovery_segments(
  motion: Any,
  progress: np.ndarray,
  cfg: RecoverySegmenterCfg | None = None,
) -> tuple[RecoverySegment, ...]:
  """Find non-overlapping low-to-stable-upright intervals.

  ``motion`` intentionally uses a structural interface (``fps``, ``frame_count``)
  so synthetic clips can exercise this logic without constructing BVH data.
  End frames are exclusive.
  """
  config = cfg or RecoverySegmenterCfg()
  progress = np.asarray(progress, dtype=np.float64)
  if progress.shape != (motion.frame_count,):
    raise ValueError(
      f"progress must have shape ({motion.frame_count},), got {progress.shape}."
    )
  if not np.all(np.isfinite(progress)):
    raise ValueError("progress contains NaN or Inf.")

  hold_frames = max(1, int(np.ceil(config.ready_hold_s * motion.fps)))
  pre_roll = int(round(config.pre_roll_s * motion.fps))
  min_frames = max(2, int(np.ceil(config.min_duration_s * motion.fps)))
  max_frames = max(min_frames, int(np.ceil(config.max_duration_s * motion.fps)))
  low_frames = np.flatnonzero(progress <= config.low_progress)
  segments: list[RecoverySegment] = []
  cursor = 0

  while cursor < motion.frame_count:
    candidates = low_frames[low_frames >= cursor]
    if candidates.size == 0:
      break
    first_low = int(candidates[0])
    latest_terminal_start = min(
      motion.frame_count - hold_frames,
      first_low + max_frames - hold_frames,
    )
    terminal_start = _first_stable_ready_frame(
      progress,
      first_low + 1,
      latest_terminal_start,
      hold_frames,
      config.ready_progress,
    )
    if terminal_start is None:
      cursor = first_low + 1
      continue

    terminal_end = terminal_start + hold_frames
    low_region = progress[first_low:terminal_start]
    lowest_frame = first_low + int(np.argmin(low_region))
    start_frame = max(0, lowest_frame - pre_roll, terminal_end - max_frames)
    if terminal_end - start_frame < min_frames:
      cursor = terminal_end
      continue
    segments.append(
      RecoverySegment(
        start_frame=start_frame,
        end_frame=terminal_end,
        initial_posture=classify_initial_posture(motion, lowest_frame),
        min_progress=float(progress[lowest_frame]),
        terminal_progress=float(np.min(progress[terminal_start:terminal_end])),
      )
    )
    cursor = terminal_end

  return tuple(segments)


def write_recovery_manifest(
  manifest: RecoveryManifest, output_path: str | Path
) -> None:
  """Write a canonical JSON representation suitable for hashing and review."""
  path = Path(output_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  serialized = json.dumps(
    manifest.to_dict(),
    ensure_ascii=False,
    indent=2,
    sort_keys=True,
  )
  path.write_text(serialized + "\n", encoding="utf-8")


def _first_stable_ready_frame(
  progress: np.ndarray,
  start: int,
  latest_start: int,
  hold_frames: int,
  threshold: float,
) -> int | None:
  if latest_start < start:
    return None
  ready = progress >= threshold
  for frame in range(start, latest_start + 1):
    if np.all(ready[frame : frame + hold_frames]):
      return frame
  return None


def _parse_source_identity(stem: str) -> tuple[str, str]:
  try:
    recording, subject = stem.rsplit("_", maxsplit=1)
  except ValueError as exc:
    raise ValueError(
      f"Expected filename '<recording>_<subject>.bvh', got {stem!r}."
    ) from exc
  if not recording or not subject:
    raise ValueError(f"Invalid LaFAN filename stem {stem!r}.")
  return recording, subject


def _split_for_recording(recording: str) -> str:
  # All subjects sharing a synchronized recording stay in the same split.
  bucket = hashlib.sha256(recording.encode("utf-8")).digest()[0] % 10
  if bucket == 0:
    return "test"
  if bucket == 1:
    return "validation"
  return "train"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while block := stream.read(1024 * 1024):
      digest.update(block)
  return digest.hexdigest()
