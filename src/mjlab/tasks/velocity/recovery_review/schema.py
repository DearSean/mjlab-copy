"""Versioned schemas and deterministic persistence for recovery review."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from mjlab.tasks.velocity.recovery_data.semantic import CONTACT_NAMES

REVIEW_QUEUE_SCHEMA_VERSION = "recovery-review-queue-v1"
REVIEW_STATE_SCHEMA_VERSION = "recovery-review-state-v3"
REVIEWED_MANIFEST_SCHEMA_VERSION = "recovery-reviewed-manifest-v1"
_LEGACY_REVIEW_STATE_SCHEMA_VERSIONS = frozenset(
  ("recovery-review-state-v1", "recovery-review-state-v2")
)

Decision = Literal["pending", "accepted", "rejected"]
TerminalMode = Literal["stationary", "locomotion", "support_only", "failure"]
Outcome = Literal["success", "partial", "failure", "unknown"]
InitialPosture = Literal["supine", "prone", "left_side", "right_side", "other"]
PhaseLabel = Literal["unreviewed", "fallen", "transition", "standing", "uncertain"]
ContactLabel = Literal["unreviewed", "contact", "no_contact", "uncertain"]
ReviewedEvent = Literal[
  "start",
  "end",
  "support_start",
  "support_complete",
  "locomotion_takeover",
]
ReviewedChoice = Literal["terminal_mode", "outcome", "initial_posture"]
_REVIEWED_EVENTS = frozenset(
  ("start", "end", "support_start", "support_complete", "locomotion_takeover")
)
_REVIEWED_CHOICES = frozenset(("terminal_mode", "outcome", "initial_posture"))


@dataclass(frozen=True)
class RecoveryReviewCandidate:
  """One automatically proposed recovery interval awaiting human review."""

  candidate_id: str
  relative_path: str
  clip_sha256: str
  recording: str
  subject: str
  split: str
  clip_frame_count: int
  fps: float
  auto_start_frame: int
  auto_end_frame: int
  auto_support_complete_frame: int
  auto_terminal_mode: TerminalMode
  auto_initial_posture: InitialPosture
  auto_min_progress: float
  auto_terminal_progress: float

  def __post_init__(self) -> None:
    if not self.candidate_id or not self.relative_path or not self.clip_sha256:
      raise ValueError("Candidate identity fields must not be empty.")
    if self.clip_frame_count <= 1 or self.fps <= 0.0:
      raise ValueError("Candidate clip metadata is invalid.")
    if not 0 <= self.auto_start_frame < self.auto_end_frame <= self.clip_frame_count:
      raise ValueError("Candidate frame interval is invalid.")
    if not (
      self.auto_start_frame <= self.auto_support_complete_frame < self.auto_end_frame
    ):
      raise ValueError("Candidate support-complete frame is outside its interval.")
    if not 0.0 <= self.auto_min_progress <= 1.0:
      raise ValueError("Candidate minimum progress must lie in [0, 1].")
    if not 0.0 <= self.auto_terminal_progress <= 1.0:
      raise ValueError("Candidate terminal progress must lie in [0, 1].")


@dataclass(frozen=True)
class ReviewFrame:
  """One frame selected for an independent semantic-label audit."""

  frame_id: str
  relative_path: str
  clip_sha256: str
  split: str
  frame: int
  clip_frame_count: int
  fps: float
  auto_phase: Literal["fallen", "transition", "standing"]
  auto_contacts: tuple[bool, ...]

  def __post_init__(self) -> None:
    if not self.frame_id or not self.relative_path or not self.clip_sha256:
      raise ValueError("Review-frame identity fields must not be empty.")
    if not 0 <= self.frame < self.clip_frame_count:
      raise ValueError("Review frame is outside its source clip.")
    if len(self.auto_contacts) != len(CONTACT_NAMES):
      raise ValueError("An automatic label is required for every contact site.")


@dataclass(frozen=True)
class ReviewQueue:
  """Deterministic review work list derived from one immutable manifest."""

  schema_version: str
  source_manifest_sha256: str
  dataset_name: str
  candidates: tuple[RecoveryReviewCandidate, ...]
  validation_frames: tuple[ReviewFrame, ...]
  test_frames: tuple[ReviewFrame, ...]

  def __post_init__(self) -> None:
    if self.schema_version != REVIEW_QUEUE_SCHEMA_VERSION:
      raise ValueError(f"Unsupported review queue schema {self.schema_version!r}.")
    candidate_ids = [candidate.candidate_id for candidate in self.candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
      raise ValueError("Review candidate IDs must be unique.")
    frame_ids = [
      frame.frame_id for frame in (*self.validation_frames, *self.test_frames)
    ]
    if len(set(frame_ids)) != len(frame_ids):
      raise ValueError("Review frame IDs must be unique.")

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass
class SegmentAnnotation:
  """Editable human judgment for one recovery candidate."""

  decision: Decision = "pending"
  start_frame: int = 0
  end_frame: int = 1
  support_start_frame: int | None = None
  support_complete_frame: int | None = None
  locomotion_takeover_frame: int | None = None
  terminal_mode: TerminalMode = "stationary"
  outcome: Outcome = "unknown"
  initial_posture: InitialPosture = "other"
  terminal_safe: bool = False
  confirmed_events: tuple[ReviewedEvent, ...] = ()
  confirmed_choices: tuple[ReviewedChoice, ...] = ()
  issues: tuple[str, ...] = ()
  notes: str = ""

  def __post_init__(self) -> None:
    self.confirmed_events = tuple(self.confirmed_events)
    if len(set(self.confirmed_events)) != len(self.confirmed_events):
      raise ValueError("Confirmed review events must be unique.")
    invalid_events = set(self.confirmed_events) - _REVIEWED_EVENTS
    if invalid_events:
      raise ValueError(f"Unknown confirmed review events: {sorted(invalid_events)}")
    self.confirmed_choices = tuple(self.confirmed_choices)
    if len(set(self.confirmed_choices)) != len(self.confirmed_choices):
      raise ValueError("Confirmed review choices must be unique.")
    invalid_choices = set(self.confirmed_choices) - _REVIEWED_CHOICES
    if invalid_choices:
      raise ValueError(f"Unknown confirmed review choices: {sorted(invalid_choices)}")

  @classmethod
  def from_candidate(cls, candidate: RecoveryReviewCandidate) -> SegmentAnnotation:
    locomotion_frame = (
      candidate.auto_support_complete_frame
      if candidate.auto_terminal_mode == "locomotion"
      else None
    )
    return cls(
      start_frame=candidate.auto_start_frame,
      end_frame=candidate.auto_end_frame,
      support_complete_frame=candidate.auto_support_complete_frame,
      locomotion_takeover_frame=locomotion_frame,
      terminal_mode=candidate.auto_terminal_mode,
      outcome=(
        "partial" if candidate.auto_terminal_mode == "support_only" else "success"
      ),
      initial_posture=candidate.auto_initial_posture,
      terminal_safe=candidate.auto_terminal_mode != "support_only",
    )

  def validate(self, candidate: RecoveryReviewCandidate) -> None:
    if not 0 <= self.start_frame < self.end_frame <= candidate.clip_frame_count:
      raise ValueError(f"Invalid reviewed interval for {candidate.candidate_id}.")
    ordered = [
      frame
      for frame in (
        self.support_start_frame,
        self.support_complete_frame,
        self.locomotion_takeover_frame,
      )
      if frame is not None
    ]
    if any(not self.start_frame <= frame < self.end_frame for frame in ordered):
      raise ValueError(f"Reviewed event is outside {candidate.candidate_id}.")
    if ordered != sorted(ordered):
      raise ValueError(
        f"Reviewed events are out of order for {candidate.candidate_id}."
      )
    if self.decision == "accepted" and self.support_complete_frame is None:
      raise ValueError("Accepted segments require a support-complete frame.")
    if (
      self.decision == "accepted"
      and self.terminal_mode == "locomotion"
      and self.locomotion_takeover_frame is None
    ):
      raise ValueError("Locomotion segments require a takeover frame.")
    if (
      self.terminal_mode != "locomotion" and self.locomotion_takeover_frame is not None
    ):
      raise ValueError("Only locomotion segments may have a takeover frame.")
    if (
      self.locomotion_takeover_frame is None
      and "locomotion_takeover" in self.confirmed_events
    ):
      raise ValueError("An unset takeover frame cannot be marked as confirmed.")
    if self.terminal_mode == "failure" and self.outcome != "failure":
      raise ValueError("A failure terminal mode requires a failure outcome.")


@dataclass
class FrameAnnotation:
  """Human phase and contact labels for one audit frame."""

  phase: PhaseLabel = "unreviewed"
  contacts: dict[str, ContactLabel] = field(
    default_factory=lambda: {name: "unreviewed" for name in CONTACT_NAMES}
  )
  notes: str = ""

  def __post_init__(self) -> None:
    if set(self.contacts) != set(CONTACT_NAMES):
      raise ValueError("Frame annotation contacts do not match the semantic schema.")

  @property
  def complete(self) -> bool:
    return self.phase != "unreviewed" and all(
      label != "unreviewed" for label in self.contacts.values()
    )


@dataclass
class ReviewState:
  """Atomic, resumable annotations from one reviewer."""

  schema_version: str
  queue_sha256: str
  reviewer: str
  segment_annotations: dict[str, SegmentAnnotation] = field(default_factory=dict)
  frame_annotations: dict[str, FrameAnnotation] = field(default_factory=dict)

  @classmethod
  def empty(cls, queue_path: str | Path, reviewer: str) -> ReviewState:
    if not reviewer.strip():
      raise ValueError("reviewer must not be empty.")
    return cls(
      schema_version=REVIEW_STATE_SCHEMA_VERSION,
      queue_sha256=file_sha256(queue_path),
      reviewer=reviewer.strip(),
    )

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


def candidate_id(clip_sha256: str, start_frame: int, end_frame: int) -> str:
  """Build an order-independent stable identifier for an automatic segment."""
  payload = f"{clip_sha256}:{start_frame}:{end_frame}".encode()
  return hashlib.sha256(payload).hexdigest()[:20]


def frame_id(clip_sha256: str, frame: int) -> str:
  """Build an order-independent stable identifier for an audit frame."""
  return hashlib.sha256(f"{clip_sha256}:{frame}".encode()).hexdigest()[:20]


def file_sha256(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    while block := stream.read(1024 * 1024):
      digest.update(block)
  return digest.hexdigest()


def save_review_queue(queue: ReviewQueue, path: str | Path) -> None:
  write_json_atomic(queue.to_dict(), path)


def load_review_queue(path: str | Path) -> ReviewQueue:
  raw = _load_json(path)
  return ReviewQueue(
    schema_version=str(raw["schema_version"]),
    source_manifest_sha256=str(raw["source_manifest_sha256"]),
    dataset_name=str(raw["dataset_name"]),
    candidates=tuple(
      RecoveryReviewCandidate(**candidate) for candidate in raw["candidates"]
    ),
    validation_frames=tuple(
      _review_frame_from_dict(frame) for frame in raw["validation_frames"]
    ),
    test_frames=tuple(_review_frame_from_dict(frame) for frame in raw["test_frames"]),
  )


def save_review_state(state: ReviewState, path: str | Path) -> None:
  write_json_atomic(state.to_dict(), path)


def load_review_state(
  path: str | Path,
  *,
  queue_path: str | Path,
  reviewer: str | None = None,
) -> ReviewState:
  raw = _load_json(path)
  source_schema_version = str(raw["schema_version"])
  supported_schema_versions = {
    REVIEW_STATE_SCHEMA_VERSION,
    *_LEGACY_REVIEW_STATE_SCHEMA_VERSIONS,
  }
  if source_schema_version not in supported_schema_versions:
    raise ValueError(f"Unsupported review state schema {source_schema_version!r}.")
  state = ReviewState(
    schema_version=REVIEW_STATE_SCHEMA_VERSION,
    queue_sha256=str(raw["queue_sha256"]),
    reviewer=str(raw["reviewer"]),
    segment_annotations={
      key: SegmentAnnotation(**value)
      for key, value in cast(
        dict[str, dict[str, Any]], raw["segment_annotations"]
      ).items()
    },
    frame_annotations={
      key: FrameAnnotation(**value)
      for key, value in cast(
        dict[str, dict[str, Any]], raw["frame_annotations"]
      ).items()
    },
  )
  expected_queue_hash = file_sha256(queue_path)
  if state.queue_sha256 != expected_queue_hash:
    raise ValueError("Review state belongs to a different review queue.")
  if reviewer is not None and state.reviewer != reviewer.strip():
    raise ValueError(
      f"Review state belongs to {state.reviewer!r}, not {reviewer.strip()!r}."
    )
  return state


def _review_frame_from_dict(raw: dict[str, Any]) -> ReviewFrame:
  return ReviewFrame(
    frame_id=str(raw["frame_id"]),
    relative_path=str(raw["relative_path"]),
    clip_sha256=str(raw["clip_sha256"]),
    split=str(raw["split"]),
    frame=int(raw["frame"]),
    clip_frame_count=int(raw["clip_frame_count"]),
    fps=float(raw["fps"]),
    auto_phase=raw["auto_phase"],
    auto_contacts=tuple(bool(value) for value in raw["auto_contacts"]),
  )


def _load_json(path: str | Path) -> dict[str, Any]:
  with Path(path).open(encoding="utf-8") as stream:
    value = json.load(stream)
  if not isinstance(value, dict):
    raise ValueError(f"Expected a JSON object in {path}.")
  return value


def write_json_atomic(value: dict[str, Any], path: str | Path) -> None:
  """Write deterministic JSON through an atomic same-directory replacement."""
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
  temporary = destination.with_name(f".{destination.name}.tmp")
  temporary.write_text(serialized, encoding="utf-8")
  temporary.replace(destination)
