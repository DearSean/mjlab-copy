"""Interactive Viser application for recovery-segment and frame review.

Example:
  uv run python -m mjlab.tasks.velocity.recovery_review.viewer \
    --reviewer reviewer-01
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import tyro
import viser

from mjlab.tasks.velocity.recovery_data import (
  CanonicalMotionClip,
  RecoverySemanticEncoder,
  SemanticMotion,
  load_lafan_bvh,
)
from mjlab.tasks.velocity.recovery_data.semantic import (
  CONTACT_NAMES,
  recovery_contact_joint_indices,
)
from mjlab.tasks.velocity.recovery_review.prepare import DEFAULT_REVIEW_DIR
from mjlab.tasks.velocity.recovery_review.schema import (
  ContactLabel,
  Decision,
  FrameAnnotation,
  InitialPosture,
  Outcome,
  PhaseLabel,
  RecoveryReviewCandidate,
  ReviewedChoice,
  ReviewedEvent,
  ReviewFrame,
  ReviewState,
  SegmentAnnotation,
  TerminalMode,
  file_sha256,
  load_review_queue,
  load_review_state,
  save_review_state,
)

_POSTURES: tuple[InitialPosture, ...] = (
  "supine",
  "prone",
  "left_side",
  "right_side",
  "other",
)
_TERMINAL_MODES: tuple[TerminalMode, ...] = (
  "stationary",
  "locomotion",
  "support_only",
  "failure",
)
_OUTCOMES: tuple[Outcome, ...] = ("success", "partial", "failure", "unknown")
_PHASES: tuple[PhaseLabel, ...] = (
  "unreviewed",
  "fallen",
  "transition",
  "standing",
  "uncertain",
)
_CONTACT_LABELS: tuple[ContactLabel, ...] = (
  "unreviewed",
  "contact",
  "no_contact",
  "uncertain",
)
_REVIEWED_EVENTS: tuple[ReviewedEvent, ...] = (
  "start",
  "end",
  "support_start",
  "support_complete",
  "locomotion_takeover",
)
_REVIEWED_CHOICES: tuple[ReviewedChoice, ...] = (
  "terminal_mode",
  "outcome",
  "initial_posture",
)


@dataclass(kw_only=True)
class RecoveryReviewViewerCfg:
  """Configuration for the local recovery review application."""

  reviewer: str
  dataset_root: Path = Path("data/lafan1")
  review_dir: Path = DEFAULT_REVIEW_DIR
  mode: Literal["segments", "validation", "test"] = "segments"
  review_file: Path | None = None
  source_length_scale_m: float = 0.01
  """Metres represented by one source BVH length unit (CMU get-up: 0.1)."""
  allow_test: bool = False
  host: str = "127.0.0.1"
  port: int = 8080


class _MotionRepository:
  def __init__(self, dataset_root: Path, source_length_scale_m: float) -> None:
    self.dataset_root = dataset_root.resolve()
    self.source_length_scale_m = source_length_scale_m
    self.encoder = RecoverySemanticEncoder()
    self._cache: OrderedDict[str, tuple[CanonicalMotionClip, SemanticMotion]] = (
      OrderedDict()
    )

  def load(self, relative_path: str) -> tuple[CanonicalMotionClip, SemanticMotion]:
    cached = self._cache.get(relative_path)
    if cached is not None:
      self._cache.move_to_end(relative_path)
      return cached
    clip = load_lafan_bvh(
      self.dataset_root / relative_path,
      source_length_scale_m=self.source_length_scale_m,
    )
    result = (clip, self.encoder.encode(clip))
    self._cache[relative_path] = result
    if len(self._cache) > 2:
      self._cache.popitem(last=False)
    return result


class _SkeletonRenderer:
  def __init__(self, server: viser.ViserServer) -> None:
    self.server = server
    server.scene.add_grid(
      "/review/ground",
      width=4.0,
      height=4.0,
      cell_size=0.25,
      section_size=1.0,
      infinite_grid=True,
    )

  def draw(
    self,
    clip: CanonicalMotionClip,
    semantic: SemanticMotion,
    frame: int,
    *,
    trajectory_interval: tuple[int, int],
    show_auto_contacts: bool,
  ) -> None:
    frame = int(np.clip(frame, 0, clip.frame_count - 1))
    positions = clip.global_positions_m[frame].astype(np.float32, copy=True)
    origin = positions[0].copy()
    origin[2] = semantic.floor_height_m
    positions -= origin
    bone_points = np.asarray(
      [
        [positions[int(parent)], positions[joint]]
        for joint, parent in enumerate(clip.skeleton.parents)
        if int(parent) >= 0
      ],
      dtype=np.float32,
    )
    contact_indices = recovery_contact_joint_indices(clip)
    contact_points = positions[contact_indices]
    if show_auto_contacts:
      contact_colors = np.asarray(
        [
          (40, 210, 90) if contact else (230, 80, 70)
          for contact in semantic.contacts[frame]
        ],
        dtype=np.uint8,
      )
    else:
      contact_colors = np.full((len(CONTACT_NAMES), 3), (240, 190, 60), np.uint8)

    start, end = trajectory_interval
    root_path = clip.global_positions_m[start:end, 0].astype(np.float32, copy=True)
    root_path -= origin
    path_segments = (
      np.stack([root_path[:-1], root_path[1:]], axis=1)
      if root_path.shape[0] >= 2
      else np.zeros((0, 2, 3), dtype=np.float32)
    )
    with self.server.atomic():
      self.server.scene.add_line_segments(
        "/review/skeleton",
        points=bone_points,
        colors=(225, 225, 235),
        line_width=4.0,
      )
      self.server.scene.add_point_cloud(
        "/review/joints",
        points=positions,
        colors=(120, 175, 255),
        point_size=0.035,
        point_shape="circle",
      )
      self.server.scene.add_point_cloud(
        "/review/contacts",
        points=contact_points,
        colors=contact_colors,
        point_size=0.07,
        point_shape="circle",
      )
      self.server.scene.add_line_segments(
        "/review/root_path",
        points=path_segments,
        colors=(110, 120, 255),
        line_width=2.0,
      )


class _ReviewApplication:
  def __init__(self, cfg: RecoveryReviewViewerCfg) -> None:
    if cfg.mode == "test" and not cfg.allow_test:
      raise ValueError("Test review is locked; pass --allow-test only after tuning.")
    self.cfg = cfg
    self.queue_path = cfg.review_dir / "review_queue.json"
    self.queue = load_review_queue(self.queue_path)
    self.review_path = cfg.review_file or cfg.review_dir / f"{cfg.reviewer}.json"
    self.state = (
      load_review_state(
        self.review_path,
        queue_path=self.queue_path,
        reviewer=cfg.reviewer,
      )
      if self.review_path.exists()
      else ReviewState.empty(self.queue_path, cfg.reviewer)
    )
    self.repository = _MotionRepository(
      cfg.dataset_root, cfg.source_length_scale_m
    )
    self.server = viser.ViserServer(
      host=cfg.host,
      port=cfg.port,
      label=f"Recovery Review: {cfg.mode}",
    )
    self.renderer = _SkeletonRenderer(self.server)
    self._loading = False
    self._playing = False
    self._window_start = 0
    self._window_end = 1
    self._current_clip: CanonicalMotionClip | None = None
    self._current_semantic: SemanticMotion | None = None
    self._lock = threading.RLock()

  def run(self) -> None:
    if self.cfg.mode == "segments":
      reviewer: _SegmentReviewer | _FrameReviewer = _SegmentReviewer(self)
    else:
      reviewer = _FrameReviewer(self)
    reviewer.setup()
    thread = threading.Thread(target=reviewer.playback_loop, daemon=True)
    thread.start()
    print(f"Review state is saved after every decision: {self.review_path}")
    print("Press Ctrl+C to stop the local review server.")
    try:
      while True:
        time.sleep(0.2)
    except KeyboardInterrupt:
      self.server.stop()

  def load_motion(
    self,
    relative_path: str,
    expected_sha256: str,
  ) -> tuple[CanonicalMotionClip, SemanticMotion]:
    source_path = self.cfg.dataset_root / relative_path
    if file_sha256(source_path) != expected_sha256:
      raise ValueError(f"Source BVH changed since queue creation: {relative_path}")
    clip, semantic = self.repository.load(relative_path)
    self._current_clip = clip
    self._current_semantic = semantic
    return clip, semantic


class _ReviewerBase:
  def __init__(self, application: _ReviewApplication) -> None:
    self.app = application
    self._uses_global_frames = False

  def playback_loop(self) -> None:
    while True:
      time.sleep(0.02)
      with self.app._lock:
        if not self.app._playing or self.app._loading:
          continue
        clip = self.app._current_clip
        if clip is None:
          continue
        speed = float(self.speed.value)
        interval = max(1.0 / clip.fps / speed, 0.01)
      time.sleep(interval)
      with self.app._lock:
        if not self.app._playing:
          continue
        next_frame = int(self.frame.value) + 1
        first_frame = self.app._window_start if self._uses_global_frames else 0
        last_frame = (
          self.app._window_end - 1
          if self._uses_global_frames
          else self.app._window_end - self.app._window_start - 1
        )
        self.frame.value = first_frame if next_frame > last_frame else next_frame

  def _common_playback_gui(
    self, max_window_frames: int, *, use_global_frames: bool = False
  ) -> None:
    self._uses_global_frames = use_global_frames
    with self.app.server.gui.add_folder("Playback"):
      self.play = self.app.server.gui.add_checkbox("Play", initial_value=False)
      self.speed = self.app.server.gui.add_dropdown(
        "Speed",
        options=("0.25", "0.5", "1.0", "2.0"),
        initial_value="1.0",
      )
      self.frame = self.app.server.gui.add_slider(
        "Global frame" if use_global_frames else "Local frame",
        min=0,
        max=max(1, max_window_frames - 1),
        step=1,
        initial_value=0,
      )
      previous_frame = self.app.server.gui.add_button("Previous frame")
      next_frame = self.app.server.gui.add_button("Next frame")

      @self.play.on_update
      def _(_) -> None:
        self.app._playing = bool(self.play.value)

      @self.frame.on_update
      def _(_) -> None:
        if not self.app._loading:
          self.draw_current_frame()

      @previous_frame.on_click
      def _(_) -> None:
        self.app._playing = False
        self.play.value = False
        first_frame = self.app._window_start if self._uses_global_frames else 0
        self.frame.value = max(first_frame, int(self.frame.value) - 1)

      @next_frame.on_click
      def _(_) -> None:
        self.app._playing = False
        self.play.value = False
        window_last_frame = (
          self.app._window_end - 1
          if self._uses_global_frames
          else self.app._window_end - self.app._window_start - 1
        )
        self.frame.value = min(window_last_frame, int(self.frame.value) + 1)

  def draw_current_frame(self) -> None:
    raise NotImplementedError


class _SegmentReviewer(_ReviewerBase):
  def __init__(self, application: _ReviewApplication) -> None:
    super().__init__(application)
    self.items = application.queue.candidates
    if not self.items:
      raise ValueError("The review queue contains no recovery candidates.")

  def setup(self) -> None:
    server = self.app.server
    self.info = server.gui.add_markdown("Loading recovery candidate...")
    with server.gui.add_folder("Candidate"):
      self.item_index = server.gui.add_slider(
        "Candidate index",
        min=0,
        max=len(self.items) - 1,
        step=1,
        initial_value=0,
      )
      previous = server.gui.add_button("Previous")
      next_item = server.gui.add_button("Next")
      next_pending = server.gui.add_button("Next pending", color="blue")

    max_clip_frames = max(candidate.clip_frame_count for candidate in self.items)
    self._common_playback_gui(max_clip_frames, use_global_frames=True)
    with server.gui.add_folder("Reviewed events"):
      self.start_frame = server.gui.add_number(
        "Start frame (global)",
        0,
        min=0,
        max=max_clip_frames - 1,
        step=1,
        disabled=True,
      )
      set_start = server.gui.add_button("Set start = current global frame")
      start_frame_status = server.gui.add_html("")
      self.end_frame = server.gui.add_number(
        "End frame (global, exclusive)",
        1,
        min=1,
        max=max_clip_frames,
        step=1,
        disabled=True,
      )
      set_end = server.gui.add_button("Set end = current global frame + 1")
      end_frame_status = server.gui.add_html("")
      self.support_start = server.gui.add_number(
        "Support start (global, -1 unset)",
        -1,
        min=-1,
        max=max_clip_frames - 1,
        step=1,
        disabled=True,
      )
      set_support_start = server.gui.add_button(
        "Set support start = current global frame"
      )
      support_start_status = server.gui.add_html("")
      self.support_complete = server.gui.add_number(
        "Support complete (global)",
        0,
        min=-1,
        max=max_clip_frames - 1,
        step=1,
        disabled=True,
      )
      set_support_complete = server.gui.add_button(
        "Set support complete = current global frame"
      )
      support_complete_status = server.gui.add_html("")
      self.locomotion_takeover = server.gui.add_number(
        "Locomotion takeover (global, -1 unset)",
        -1,
        min=-1,
        max=max_clip_frames - 1,
        step=1,
        disabled=True,
      )
      set_locomotion_takeover = server.gui.add_button(
        "Set locomotion takeover = current global frame"
      )
      clear_locomotion_takeover = server.gui.add_button(
        "Clear locomotion takeover (N/A)"
      )
      locomotion_takeover_status = server.gui.add_html("")
      self.terminal_mode = server.gui.add_dropdown(
        "Terminal mode", options=_TERMINAL_MODES
      )
      terminal_mode_status = server.gui.add_html("")
      self.outcome = server.gui.add_dropdown("Outcome", options=_OUTCOMES)
      outcome_status = server.gui.add_html("")
      self.initial_posture = server.gui.add_dropdown(
        "Initial posture", options=_POSTURES
      )
      initial_posture_status = server.gui.add_html("")
      self.terminal_safe = server.gui.add_checkbox(
        "Terminal remains safe", initial_value=False
      )
      self.issues = server.gui.add_text("Issues (comma separated)", initial_value="")
      self.notes = server.gui.add_text("Notes", initial_value="", multiline=True)
    with server.gui.add_folder("Decision"):
      save_draft = server.gui.add_button("Save draft")
      accept = server.gui.add_button("Accept", color="green")
      reject = server.gui.add_button("Reject", color="red")
    with server.gui.add_folder("Modify finalized review", expand_by_default=True):
      modify_status = server.gui.add_html("")
      modify = server.gui.add_button("Modify", color="orange", disabled=True)

    self._event_values = {
      "start": self.start_frame,
      "end": self.end_frame,
      "support_start": self.support_start,
      "support_complete": self.support_complete,
      "locomotion_takeover": self.locomotion_takeover,
    }
    self._event_buttons = {
      "start": set_start,
      "end": set_end,
      "support_start": set_support_start,
      "support_complete": set_support_complete,
      "locomotion_takeover": set_locomotion_takeover,
    }
    self._event_status = {
      "start": start_frame_status,
      "end": end_frame_status,
      "support_start": support_start_status,
      "support_complete": support_complete_status,
      "locomotion_takeover": locomotion_takeover_status,
    }
    self._set_events: set[ReviewedEvent] = set()
    self._choice_values = {
      "terminal_mode": self.terminal_mode,
      "outcome": self.outcome,
      "initial_posture": self.initial_posture,
    }
    self._choice_status = {
      "terminal_mode": terminal_mode_status,
      "outcome": outcome_status,
      "initial_posture": initial_posture_status,
    }
    self._set_choices: set[ReviewedChoice] = set()
    self._clear_locomotion_button = clear_locomotion_takeover
    self._accept_button = accept
    self._review_fields_editable = True
    self._annotation_controls = (
      *self._event_buttons.values(),
      clear_locomotion_takeover,
      self.terminal_mode,
      self.outcome,
      self.initial_posture,
      self.terminal_safe,
      self.issues,
      self.notes,
      save_draft,
      accept,
      reject,
    )
    self._modify_status = modify_status
    self._modify_button = modify
    self._editing_finalized = False

    @self.item_index.on_update
    def _(_) -> None:
      if not self.app._loading:
        self.load_item(int(self.item_index.value))

    @previous.on_click
    def _(_) -> None:
      self.item_index.value = max(0, int(self.item_index.value) - 1)

    @next_item.on_click
    def _(_) -> None:
      self.item_index.value = min(len(self.items) - 1, int(self.item_index.value) + 1)

    @next_pending.on_click
    def _(_) -> None:
      self._go_to_next_pending()

    @set_start.on_click
    def _(_) -> None:
      self._set_event_value("start", self._current_global_frame())

    @set_end.on_click
    def _(_) -> None:
      candidate = self.items[int(self.item_index.value)]
      self._set_event_value(
        "end",
        min(candidate.clip_frame_count, self._current_global_frame() + 1),
      )

    @set_support_start.on_click
    def _(_) -> None:
      self._set_event_value("support_start", self._current_global_frame())

    @set_support_complete.on_click
    def _(_) -> None:
      self._set_event_value("support_complete", self._current_global_frame())

    @set_locomotion_takeover.on_click
    def _(_) -> None:
      self._set_event_value("locomotion_takeover", self._current_global_frame())

    @clear_locomotion_takeover.on_click
    def _(_) -> None:
      self._clear_locomotion_takeover()

    @self.terminal_mode.on_update
    def _(_) -> None:
      if self.app._loading:
        return
      self._confirm_choice("terminal_mode")
      self._handle_terminal_mode_change()

    @self.outcome.on_update
    def _(_) -> None:
      if not self.app._loading:
        self._confirm_choice("outcome")

    @self.initial_posture.on_update
    def _(_) -> None:
      if not self.app._loading:
        self._confirm_choice("initial_posture")

    @save_draft.on_click
    def _(_) -> None:
      self._save("pending", advance=False)

    @accept.on_click
    def _(_) -> None:
      self._save("accepted", advance=True)

    @reject.on_click
    def _(_) -> None:
      self._save("rejected", advance=True)

    @modify.on_click
    def _(_) -> None:
      self._begin_modify()

    self.load_item(0)

  def load_item(self, index: int) -> None:
    with self.app._lock:
      self.app._loading = True
      try:
        candidate = self.items[index]
        annotation = self.app.state.segment_annotations.get(
          candidate.candidate_id,
          SegmentAnnotation.from_candidate(candidate),
        )
        clip, _ = self.app.load_motion(
          candidate.relative_path,
          candidate.clip_sha256,
        )
        self.start_frame.value = annotation.start_frame
        self.end_frame.value = annotation.end_frame
        self.support_start.value = _optional_frame_value(annotation.support_start_frame)
        self.support_complete.value = _optional_frame_value(
          annotation.support_complete_frame
        )
        self.locomotion_takeover.value = _optional_frame_value(
          annotation.locomotion_takeover_frame
        )
        self.terminal_mode.value = annotation.terminal_mode
        self.outcome.value = annotation.outcome
        self.initial_posture.value = annotation.initial_posture
        self.terminal_safe.value = annotation.terminal_safe
        self.issues.value = ", ".join(annotation.issues)
        self.notes.value = annotation.notes
        self._set_events = set(annotation.confirmed_events)
        self._set_choices = set(annotation.confirmed_choices)
        if annotation.terminal_mode != "locomotion":
          self.locomotion_takeover.value = -1
          self._set_events.discard("locomotion_takeover")
        self._refresh_event_status()
        self._refresh_choice_status()
        self._editing_finalized = False
        self._refresh_edit_state(annotation)
        window_start, window_end = self._annotation_window(
          candidate,
          clip.frame_count,
          use_automatic_interval=False,
        )
        target = (
          annotation.support_complete_frame
          if annotation.support_complete_frame is not None
          else candidate.auto_support_complete_frame
        )
        self._set_playback_window(window_start, window_end, target=target)
        self._update_info(candidate, annotation)
      finally:
        self.app._loading = False
      self.draw_current_frame()

  def draw_current_frame(self) -> None:
    clip = self.app._current_clip
    semantic = self.app._current_semantic
    if clip is None or semantic is None:
      return
    global_frame = self._current_global_frame()
    candidate = self.items[int(self.item_index.value)]
    trajectory_interval = self._annotation_window(
      candidate,
      clip.frame_count,
      use_automatic_interval=True,
    )
    self.app.renderer.draw(
      clip,
      semantic,
      global_frame,
      trajectory_interval=trajectory_interval,
      show_auto_contacts=False,
    )
    annotation = self.app.state.segment_annotations.get(
      candidate.candidate_id,
      SegmentAnnotation.from_candidate(candidate),
    )
    self._update_info(candidate, annotation, current_frame=global_frame)

  def _current_global_frame(self) -> int:
    return int(
      np.clip(
        int(self.frame.value),
        self.app._window_start,
        self.app._window_end - 1,
      )
    )

  def _annotation_window(
    self,
    candidate: RecoveryReviewCandidate,
    clip_frame_count: int,
    *,
    use_automatic_interval: bool,
  ) -> tuple[int, int]:
    context = int(round(candidate.fps))
    if use_automatic_interval:
      start_frame = candidate.auto_start_frame
      end_frame = candidate.auto_end_frame
    else:
      start_frame = int(self.start_frame.value)
      end_frame = int(self.end_frame.value)
    return (
      max(0, start_frame - context),
      min(clip_frame_count, end_frame + context),
    )

  def _set_playback_window(
    self, window_start: int, window_end: int, *, target: int
  ) -> None:
    if window_end <= window_start:
      raise ValueError("Playback window must contain at least one frame.")
    self.app._window_start = window_start
    self.app._window_end = window_end
    target = int(np.clip(target, window_start, window_end - 1))
    with self.app.server.atomic():
      self.frame.min = window_start
      self.frame.max = window_end - 1
      self.frame.value = target

  def _begin_modify(self) -> None:
    candidate = self.items[int(self.item_index.value)]
    annotation = self.app.state.segment_annotations.get(candidate.candidate_id)
    clip = self.app._current_clip
    if annotation is None or annotation.decision == "pending" or clip is None:
      return
    current_frame = self._current_global_frame()
    self._editing_finalized = True
    self._refresh_edit_state(annotation)
    window_start, window_end = self._annotation_window(
      candidate,
      clip.frame_count,
      use_automatic_interval=True,
    )
    self._set_playback_window(window_start, window_end, target=current_frame)
    self.draw_current_frame()

  def _set_event_value(self, event: ReviewedEvent, value: int) -> None:
    self._event_values[event].value = value
    self._set_events.add(event)
    self._refresh_event_status()
    self._refresh_locomotion_controls()

  def _clear_locomotion_takeover(self) -> None:
    self.locomotion_takeover.value = -1
    self._set_events.discard("locomotion_takeover")
    self._refresh_event_status()
    self._refresh_locomotion_controls()

  def _handle_terminal_mode_change(self) -> None:
    if self.terminal_mode.value != "locomotion":
      self._clear_locomotion_takeover()
    else:
      self._refresh_event_status()
      self._refresh_locomotion_controls()

  def _refresh_event_status(self) -> None:
    for event, control in self._event_values.items():
      was_set = event in self._set_events
      self._event_status[event].content = _event_value_status_html(
        int(control.value),
        was_set=was_set,
        not_applicable=(
          event == "locomotion_takeover" and self.terminal_mode.value != "locomotion"
        ),
        required_missing=(
          event == "locomotion_takeover"
          and self.terminal_mode.value == "locomotion"
          and int(control.value) < 0
        ),
      )
      self._event_buttons[event].color = "green" if was_set else None

  def _confirm_choice(self, choice: ReviewedChoice) -> None:
    self._set_choices.add(choice)
    self._refresh_choice_status()

  def _refresh_choice_status(self) -> None:
    for choice, control in self._choice_values.items():
      self._choice_status[choice].content = _choice_value_status_html(
        str(control.value), was_set=choice in self._set_choices
      )

  def _refresh_edit_state(self, annotation: SegmentAnnotation) -> None:
    finalized = annotation.decision in ("accepted", "rejected")
    editable = not finalized or self._editing_finalized
    self._review_fields_editable = editable
    for control in self._annotation_controls:
      control.disabled = not editable
    self._modify_button.disabled = not finalized or self._editing_finalized
    if not finalized:
      mode = "draft"
    elif self._editing_finalized:
      mode = "editing"
    else:
      mode = "locked"
    self._modify_status.content = _modify_review_status_html(mode)
    self._refresh_locomotion_controls()

  def _refresh_locomotion_controls(self) -> None:
    is_locomotion = self.terminal_mode.value == "locomotion"
    is_missing = int(self.locomotion_takeover.value) < 0
    self._event_buttons["locomotion_takeover"].disabled = (
      not self._review_fields_editable or not is_locomotion
    )
    self._clear_locomotion_button.disabled = (
      not self._review_fields_editable or not is_locomotion or is_missing
    )
    self._accept_button.disabled = not self._review_fields_editable or (
      is_locomotion and is_missing
    )

  def _save(self, decision: Decision, *, advance: bool) -> None:
    candidate = self.items[int(self.item_index.value)]
    saved_annotation = self.app.state.segment_annotations.get(candidate.candidate_id)
    if (
      saved_annotation is not None
      and saved_annotation.decision in ("accepted", "rejected")
      and not self._editing_finalized
    ):
      self.info.content = (
        "## Review is locked\n\nClick **Modify** before changing or saving it."
      )
      return
    annotation = SegmentAnnotation(
      decision=decision,
      start_frame=int(self.start_frame.value),
      end_frame=int(self.end_frame.value),
      support_start_frame=_optional_frame(int(self.support_start.value)),
      support_complete_frame=_optional_frame(int(self.support_complete.value)),
      locomotion_takeover_frame=_optional_frame(int(self.locomotion_takeover.value)),
      terminal_mode=cast(TerminalMode, self.terminal_mode.value),
      outcome=cast(Outcome, self.outcome.value),
      initial_posture=cast(InitialPosture, self.initial_posture.value),
      terminal_safe=bool(self.terminal_safe.value),
      confirmed_events=tuple(
        event for event in _REVIEWED_EVENTS if event in self._set_events
      ),
      confirmed_choices=tuple(
        choice for choice in _REVIEWED_CHOICES if choice in self._set_choices
      ),
      issues=tuple(
        issue.strip() for issue in self.issues.value.split(",") if issue.strip()
      ),
      notes=self.notes.value.strip(),
    )
    try:
      annotation.validate(candidate)
    except ValueError as exc:
      self.info.content = f"## Cannot save\n\n{exc}"
      return
    self.app.state.segment_annotations[candidate.candidate_id] = annotation
    save_review_state(self.app.state, self.app.review_path)
    self._editing_finalized = False
    self._refresh_edit_state(annotation)
    self._update_info(candidate, annotation)
    if advance:
      self._go_to_next_pending()

  def _go_to_next_pending(self) -> None:
    current = int(self.item_index.value)
    for offset in range(1, len(self.items) + 1):
      index = (current + offset) % len(self.items)
      annotation = self.app.state.segment_annotations.get(
        self.items[index].candidate_id
      )
      if annotation is None or annotation.decision == "pending":
        self.item_index.value = index
        return

  def _update_info(
    self,
    candidate: RecoveryReviewCandidate,
    annotation: SegmentAnnotation,
    *,
    current_frame: int | None = None,
  ) -> None:
    decided = sum(
      annotation.decision != "pending"
      for annotation in self.app.state.segment_annotations.values()
    )
    current = "" if current_frame is None else f"- Current frame: `{current_frame}`\n"
    edit_state = (
      "editing"
      if self._editing_finalized
      else "locked"
      if annotation.decision in ("accepted", "rejected")
      else "draft"
    )
    self.info.content = (
      f"## Segment {int(self.item_index.value) + 1}/{len(self.items)}\n\n"
      f"- File: `{candidate.relative_path}`\n"
      f"- Recording/subject/split: `{candidate.recording}` / "
      f"`{candidate.subject}` / `{candidate.split}`\n"
      f"- Auto terminal: `{candidate.auto_terminal_mode}` at "
      f"`{candidate.auto_support_complete_frame}`\n"
      f"- Auto posture: `{candidate.auto_initial_posture}`\n"
      f"- Decision: `{annotation.decision}`\n"
      f"- Edit state: `{edit_state}`\n"
      f"- Completed decisions: `{decided}/{len(self.items)}`\n"
      f"- Selected interval (global): `[{int(self.start_frame.value)}, "
      f"{int(self.end_frame.value)})`\n"
      f"- Review window (global): `[{self.app._window_start}, "
      f"{self.app._window_end})`\n"
      f"{current}"
    )


class _FrameReviewer(_ReviewerBase):
  def __init__(self, application: _ReviewApplication) -> None:
    super().__init__(application)
    self.items: tuple[ReviewFrame, ...] = (
      application.queue.validation_frames
      if application.cfg.mode == "validation"
      else application.queue.test_frames
    )
    if not self.items:
      raise ValueError(f"The review queue contains no {application.cfg.mode} frames.")

  def setup(self) -> None:
    server = self.app.server
    self.info = server.gui.add_markdown("Loading audit frame...")
    with server.gui.add_folder("Audit item"):
      self.item_index = server.gui.add_slider(
        "Item index",
        min=0,
        max=len(self.items) - 1,
        step=1,
        initial_value=0,
      )
      previous = server.gui.add_button("Previous")
      next_item = server.gui.add_button("Next")
      next_pending = server.gui.add_button("Next pending", color="blue")
    self._common_playback_gui(31)
    with server.gui.add_folder("Blind human labels"):
      self.phase = server.gui.add_dropdown("Phase", options=_PHASES)
      self.contact_controls = {
        contact: server.gui.add_dropdown(contact, options=_CONTACT_LABELS)
        for contact in CONTACT_NAMES
      }
      self.notes = server.gui.add_text("Notes", initial_value="", multiline=True)
      save = server.gui.add_button("Save labels", color="green")

    @self.item_index.on_update
    def _(_) -> None:
      if not self.app._loading:
        self.load_item(int(self.item_index.value))

    @previous.on_click
    def _(_) -> None:
      self.item_index.value = max(0, int(self.item_index.value) - 1)

    @next_item.on_click
    def _(_) -> None:
      self.item_index.value = min(len(self.items) - 1, int(self.item_index.value) + 1)

    @next_pending.on_click
    def _(_) -> None:
      self._go_to_next_pending()

    @save.on_click
    def _(_) -> None:
      self._save()

    self.load_item(0)

  def load_item(self, index: int) -> None:
    with self.app._lock:
      self.app._loading = True
      try:
        item = self.items[index]
        annotation = self.app.state.frame_annotations.get(
          item.frame_id,
          FrameAnnotation(),
        )
        clip, _ = self.app.load_motion(item.relative_path, item.clip_sha256)
        self.app._window_start = max(0, item.frame - 15)
        self.app._window_end = min(clip.frame_count, item.frame + 16)
        self.frame.value = item.frame - self.app._window_start
        self.phase.value = annotation.phase
        for contact, control in self.contact_controls.items():
          control.value = annotation.contacts[contact]
        self.notes.value = annotation.notes
        self._update_info(item, annotation)
      finally:
        self.app._loading = False
      self.draw_current_frame()

  def draw_current_frame(self) -> None:
    clip = self.app._current_clip
    semantic = self.app._current_semantic
    if clip is None or semantic is None:
      return
    global_frame = min(
      self.app._window_start + int(self.frame.value),
      self.app._window_end - 1,
    )
    self.app.renderer.draw(
      clip,
      semantic,
      global_frame,
      trajectory_interval=(self.app._window_start, self.app._window_end),
      show_auto_contacts=False,
    )

  def _save(self) -> None:
    item = self.items[int(self.item_index.value)]
    annotation = FrameAnnotation(
      phase=cast(PhaseLabel, self.phase.value),
      contacts={
        contact: cast(ContactLabel, control.value)
        for contact, control in self.contact_controls.items()
      },
      notes=self.notes.value.strip(),
    )
    self.app.state.frame_annotations[item.frame_id] = annotation
    save_review_state(self.app.state, self.app.review_path)
    self._update_info(item, annotation)
    self._go_to_next_pending()

  def _go_to_next_pending(self) -> None:
    current = int(self.item_index.value)
    for offset in range(1, len(self.items) + 1):
      index = (current + offset) % len(self.items)
      annotation = self.app.state.frame_annotations.get(self.items[index].frame_id)
      if annotation is None or not annotation.complete:
        self.item_index.value = index
        return

  def _update_info(self, item: ReviewFrame, annotation: FrameAnnotation) -> None:
    completed = sum(
      annotation.complete for annotation in self.app.state.frame_annotations.values()
    )
    self.info.content = (
      f"## {self.app.cfg.mode.capitalize()} frame "
      f"{int(self.item_index.value) + 1}/{len(self.items)}\n\n"
      f"- File: `{item.relative_path}`\n"
      f"- Target frame: `{item.frame}` (center of the 31-frame window)\n"
      f"- Labels complete: `{annotation.complete}`\n"
      f"- Completed audit frames: `{completed}/{len(self.items)}`\n\n"
      "Automatic phase/contact labels are intentionally hidden during audit."
    )


def _event_value_status_html(
  value: int,
  *,
  was_set: bool,
  not_applicable: bool = False,
  required_missing: bool = False,
) -> str:
  if not_applicable:
    color, state, display_value = "#868e96", "N/A", "—"
  elif required_missing:
    color, state, display_value = "#e03131", "REQUIRED", "UNSET"
  else:
    color = "#2f9e44" if was_set else "#868e96"
    state = "SET" if was_set else "AUTO"
    display_value = str(value)
  return (
    '<div style="display:flex;justify-content:flex-end;align-items:center;'
    'gap:0.45rem;margin-top:-0.25rem;margin-bottom:0.25rem;">'
    f'<span style="color:{color};font-size:0.72rem;font-weight:700;">'
    f"{state}</span>"
    f'<span style="color:{color};font-size:0.9rem;font-weight:700;">'
    f"{display_value}</span></div>"
  )


def _choice_value_status_html(value: str, *, was_set: bool) -> str:
  color = "#2f9e44" if was_set else "#868e96"
  state = "SET" if was_set else "AUTO"
  return (
    '<div style="display:flex;justify-content:flex-end;align-items:center;'
    'gap:0.45rem;margin-top:-0.25rem;margin-bottom:0.25rem;">'
    f'<span style="color:{color};font-size:0.72rem;font-weight:700;">'
    f"{state}</span>"
    f'<span style="color:{color};font-size:0.9rem;font-weight:700;">'
    f"{value}</span></div>"
  )


def _modify_review_status_html(
  mode: Literal["draft", "locked", "editing"],
) -> str:
  if mode == "locked":
    background, border, color = "#fff3bf", "#f59f00", "#8f5b00"
    message = "LOCKED · Click Modify before changing finalized values."
  elif mode == "editing":
    background, border, color = "#d3f9d8", "#2f9e44", "#1b6b2a"
    message = "EDITING · Set buttons and review fields are unlocked."
  else:
    background, border, color = "#e7f5ff", "#339af0", "#1864ab"
    message = "DRAFT · Review fields are editable until Accept or Reject."
  return (
    f'<div style="padding:0.65rem 0.75rem;border-radius:0.4rem;'
    f"background:{background};border:1px solid {border};color:{color};"
    f'font-size:0.8rem;font-weight:700;">{message}</div>'
  )


def _optional_frame(value: int) -> int | None:
  return None if value < 0 else value


def _optional_frame_value(value: int | None) -> int:
  return -1 if value is None else value


def main() -> None:
  cfg = tyro.cli(RecoveryReviewViewerCfg)
  _ReviewApplication(cfg).run()


if __name__ == "__main__":
  main()
