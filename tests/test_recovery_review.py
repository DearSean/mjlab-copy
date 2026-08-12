from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mjlab.tasks.velocity.recovery_data import (
  RECOVERY_SEMANTIC_DIM,
  CanonicalMotionClip,
  SemanticMotion,
  Skeleton,
)
from mjlab.tasks.velocity.recovery_data.semantic import (
  CONTACT_NAMES,
  recovery_semantic_feature_names,
)
from mjlab.tasks.velocity.recovery_review.candidates import (
  detect_recovery_candidates,
)
from mjlab.tasks.velocity.recovery_review.compile import compile_reviewed_manifest
from mjlab.tasks.velocity.recovery_review.evaluate import evaluate_review_labels
from mjlab.tasks.velocity.recovery_review.schema import (
  REVIEW_QUEUE_SCHEMA_VERSION,
  FrameAnnotation,
  ReviewFrame,
  ReviewQueue,
  ReviewState,
  SegmentAnnotation,
  file_sha256,
  load_review_queue,
  load_review_state,
  save_review_queue,
  save_review_state,
  write_json_atomic,
)


def test_dynamic_recovery_candidate_does_not_require_static_standing():
  motion = _minimal_lafan_motion(frame_count=100, fps=20.0)
  semantic = _dynamic_recovery_semantic(frame_count=100)

  candidates = detect_recovery_candidates(
    motion,
    semantic,
    relative_path="fallAndGetUp1_subject1.bvh",
    clip_sha256="a" * 64,
    recording="fallAndGetUp1",
    subject="subject1",
    split="validation",
  )

  assert len(candidates) == 1
  assert candidates[0].auto_terminal_mode == "locomotion"
  assert candidates[0].auto_terminal_progress < 0.85
  assert candidates[0].auto_support_complete_frame >= 40


def test_locomotion_takeover_is_required_only_for_accepted_locomotion():
  candidate = detect_recovery_candidates(
    _minimal_lafan_motion(frame_count=100, fps=20.0),
    _dynamic_recovery_semantic(frame_count=100),
    relative_path="fallAndGetUp1_subject1.bvh",
    clip_sha256="a" * 64,
    recording="fallAndGetUp1",
    subject="subject1",
    split="validation",
  )[0]
  annotation = SegmentAnnotation.from_candidate(candidate)
  annotation.locomotion_takeover_frame = None
  annotation.validate(candidate)

  annotation.decision = "accepted"
  with pytest.raises(ValueError, match="require a takeover"):
    annotation.validate(candidate)

  annotation.decision = "pending"
  annotation.terminal_mode = "stationary"
  annotation.validate(candidate)
  annotation.locomotion_takeover_frame = candidate.auto_support_complete_frame
  with pytest.raises(ValueError, match="Only locomotion"):
    annotation.validate(candidate)


def test_review_state_metrics_and_compile_round_trip(tmp_path: Path):
  manifest_path = tmp_path / "lafan_manifest.json"
  write_json_atomic(
    {
      "schema_version": "recovery-semantic-v1",
      "dataset_name": "lafan",
      "clips": [
        {
          "relative_path": "fallAndGetUp1_subject1.bvh",
          "sha256": "a" * 64,
          "recording": "fallAndGetUp1",
          "subject": "subject1",
          "split": "validation",
          "frame_count": 100,
          "fps": 20.0,
          "duration_s": 4.95,
          "recovery_source": True,
          "segments": [],
        }
      ],
      "failures": [],
    },
    manifest_path,
  )
  candidate = detect_recovery_candidates(
    _minimal_lafan_motion(frame_count=100, fps=20.0),
    _dynamic_recovery_semantic(frame_count=100),
    relative_path="fallAndGetUp1_subject1.bvh",
    clip_sha256="a" * 64,
    recording="fallAndGetUp1",
    subject="subject1",
    split="validation",
  )[0]
  audit_frame = ReviewFrame(
    frame_id="frame-1",
    relative_path=candidate.relative_path,
    clip_sha256=candidate.clip_sha256,
    split="validation",
    frame=45,
    clip_frame_count=100,
    fps=20.0,
    auto_phase="transition",
    auto_contacts=(False, False, False, False, False, False, True, False),
  )
  queue = ReviewQueue(
    schema_version=REVIEW_QUEUE_SCHEMA_VERSION,
    source_manifest_sha256=file_sha256(manifest_path),
    dataset_name="lafan",
    candidates=(candidate,),
    validation_frames=(audit_frame,),
    test_frames=(),
  )
  queue_path = tmp_path / "review_queue.json"
  save_review_queue(queue, queue_path)
  assert load_review_queue(queue_path) == queue

  state = ReviewState.empty(queue_path, "reviewer-01")
  segment = SegmentAnnotation.from_candidate(candidate)
  segment.decision = "accepted"
  segment.terminal_safe = True
  segment.confirmed_events = ("start", "support_complete")
  segment.confirmed_choices = ("terminal_mode", "initial_posture")
  segment.validate(candidate)
  state.segment_annotations[candidate.candidate_id] = segment
  state.frame_annotations[audit_frame.frame_id] = FrameAnnotation(
    phase="transition",
    contacts={
      contact: "contact" if audit_frame.auto_contacts[index] else "no_contact"
      for index, contact in enumerate(CONTACT_NAMES)
    },
  )
  review_path = tmp_path / "reviewer-01.json"
  save_review_state(state, review_path)
  loaded_state = load_review_state(review_path, queue_path=queue_path)
  assert loaded_state.reviewer == "reviewer-01"
  assert loaded_state.segment_annotations[candidate.candidate_id].confirmed_events == (
    "start",
    "support_complete",
  )
  assert loaded_state.segment_annotations[candidate.candidate_id].confirmed_choices == (
    "terminal_mode",
    "initial_posture",
  )

  metrics = evaluate_review_labels(
    queue.validation_frames,
    loaded_state.frame_annotations,
  )
  assert metrics["phase"]["macro_f1"] == 1.0
  assert metrics["contacts"]["macro_f1"] == 1.0

  output_path = tmp_path / "lafan_manifest.reviewed.json"
  compiled = compile_reviewed_manifest(
    manifest_path=manifest_path,
    queue_path=queue_path,
    review_path=review_path,
    output_path=output_path,
  )
  assert compiled["review_summary"]["accepted_count"] == 1
  assert compiled["clips"][0]["segments"][0]["terminal_mode"] == "locomotion"
  assert compiled["clips"][0]["segments"][0]["confirmed_events"] == (
    "start",
    "support_complete",
  )
  assert compiled["clips"][0]["segments"][0]["confirmed_choices"] == (
    "terminal_mode",
    "initial_posture",
  )
  assert output_path.is_file()


def _minimal_lafan_motion(*, frame_count: int, fps: float) -> CanonicalMotionClip:
  names = (
    "Hips",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToe",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToe",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
  )
  parents = np.asarray([-1] + [0] * (len(names) - 1), dtype=np.int64)
  offsets = np.zeros((len(names), 3), dtype=np.float32)
  positions = np.zeros((frame_count, len(names), 3), dtype=np.float32)
  positions[:, names.index("LeftUpLeg")] = (-0.1, 0.0, 0.0)
  positions[:, names.index("RightUpLeg")] = (0.1, 0.0, 0.0)
  positions[:, names.index("Spine2")] = (0.0, 0.0, 0.5)
  positions[:, names.index("LeftShoulder")] = (-0.2, 0.0, 0.5)
  positions[:, names.index("RightShoulder")] = (0.2, 0.0, 0.5)
  quaternion = np.zeros((frame_count, len(names), 4), dtype=np.float32)
  quaternion[..., 0] = 1.0
  return CanonicalMotionClip(
    skeleton=Skeleton(joint_names=names, parents=parents, offsets_m=offsets),
    fps=fps,
    local_positions_m=positions.copy(),
    local_quat_wxyz=quaternion.copy(),
    global_positions_m=positions,
    global_quat_wxyz=quaternion,
  )


def _dynamic_recovery_semantic(*, frame_count: int) -> SemanticMotion:
  features = np.zeros((frame_count, RECOVERY_SEMANTIC_DIM), dtype=np.float32)
  features[:, 1] = 1.0
  features[:, 5] = 1.0
  features[:20, 0] = 0.2
  features[20:40, 0] = np.linspace(0.2, 0.45, 20)
  features[40:, 0] = 0.45
  features[40:, 7] = 0.3
  progress = np.zeros(frame_count, dtype=np.float32)
  progress[20:40] = np.linspace(0.0, 0.6, 20)
  progress[40:] = 0.6
  features[:, -1] = progress
  contacts = np.zeros((frame_count, len(CONTACT_NAMES)), dtype=np.bool_)
  contacts[40:, 6] = True
  return SemanticMotion(
    features=features,
    contacts=contacts,
    progress=progress,
    floor_height_m=0.0,
    nominal_height_m=1.0,
    feature_names=recovery_semantic_feature_names(),
  )
