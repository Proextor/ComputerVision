"""Deterministic M4 tests; no camera, model, sleeps, or GPU."""

from datetime import datetime, timedelta, timezone
import json

import pytest
from pydantic import ValidationError

from reactive_vision.behaviour import BehaviourAnalyzer
from reactive_vision.main import snapshot
from reactive_vision.room_state import RoomConfig, RoomStateAnalyzer
from reactive_vision.schemas import (
    ActivityLevel, ActivityObservation, BoundingBox, CameraStatus, EventType,
    GestureObservation, GestureStatus, MotionState, NormalizedPoint,
    OccupancyState, PersonObservation, PoseAnalysisObservation,
    PoseQuality, PoseQualityObservation,
)
from reactive_vision.zones import Zone, ZoneConfig


ORIGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)


def at(seconds: float) -> datetime:
    return ORIGIN + timedelta(seconds=seconds)


def person(track_id=1, x=0.5, y=0.8, proximity=0.4, score=None,
           quality=PoseQuality.GOOD, motion=MotionState.UNKNOWN,
           left=False, right=False, wave=False, confirmed_zones=(), dwell=0):
    gestures = GestureObservation(
        left_hand_raised=GestureStatus.ACTIVE if left else GestureStatus.INACTIVE,
        right_hand_raised=GestureStatus.ACTIVE if right else GestureStatus.INACTIVE,
        both_hands_raised=GestureStatus.ACTIVE if left and right else GestureStatus.INACTIVE,
        left_wave=GestureStatus.ACTIVE if wave else GestureStatus.INACTIVE,
        right_wave=GestureStatus.INACTIVE,
    )
    pose = PoseAnalysisObservation(
        quality=PoseQualityObservation(usable_keypoint_count=12, usable_keypoint_ratio=12 / 17,
                                       pose_confidence=0.9, pose_quality=quality),
        activity=ActivityObservation(activity_score=score,
                                     activity_level=ActivityLevel.UNKNOWN),
        gestures=gestures,
    )
    return PersonObservation(
        track_id=track_id,
        bbox=BoundingBox(x1=max(0, x - 0.1), y1=max(0, y - proximity),
                         x2=min(1, x + 0.1), y2=y),
        anchor_point=NormalizedPoint(x=x, y=y), proximity_score=proximity,
        motion_state=motion, pose_analysis=pose,
        confirmed_zones=list(confirmed_zones),
        dwell_seconds={z: dwell for z in confirmed_zones},
    )


def analyze(room, people, seconds, emitter=None, status=CameraStatus.OK,
            occupancy=None, age=0):
    events = []
    emit = emitter or (lambda kind, at, track_id, zone_id, metadata: events.append((kind, metadata)))
    state = room.update(people, status, at(seconds), occupancy or {}, emit, age)
    return state, events


def test_occupancy_bands_dropout_and_single_transitions():
    room = RoomStateAnalyzer(config=RoomConfig(occupied_debounce_seconds=0.3,
                                                empty_debounce_seconds=0.7))
    state, _ = analyze(room, [], 0)
    assert state.occupancy.state == OccupancyState.UNKNOWN
    state, _ = analyze(room, [], 0.7)
    assert (state.occupancy.state, state.occupancy.stabilized_count) == (OccupancyState.EMPTY, 0)
    state, _ = analyze(room, [person()], 1)
    assert (state.occupancy.visible_count, state.occupancy.state) == (1, OccupancyState.EMPTY)
    state, events = analyze(room, [person()], 1.3)
    assert state.occupancy.state == OccupancyState.SINGLE
    assert [e[0] for e in events] == [EventType.OCCUPANCY_STATE_CHANGED, EventType.ROOM_BECAME_OCCUPIED]
    state, _ = analyze(room, [], 1.4)
    assert state.occupancy.visible_count == 0
    assert state.occupancy.stabilized_count == 1
    assert state.occupancy.state == OccupancyState.SINGLE
    state, _ = analyze(room, [person()], 1.5)
    assert state.occupancy.state == OccupancyState.SINGLE
    state, _ = analyze(room, [], 2)
    state, events = analyze(room, [], 2.7)
    assert state.occupancy.state == OccupancyState.EMPTY
    assert [e[0] for e in events] == [EventType.OCCUPANCY_STATE_CHANGED, EventType.ROOM_BECAME_EMPTY]
    _, events = analyze(room, [], 3)
    assert events == []


def test_small_group_crowd_and_configurable_thresholds():
    room = RoomStateAnalyzer(config=RoomConfig(small_group_max=3, crowd_min=4,
                                                occupied_debounce_seconds=0))
    state, _ = analyze(room, [person(i) for i in range(2)], 0)
    assert state.occupancy.state == OccupancyState.SMALL_GROUP
    state, _ = analyze(room, [person(i) for i in range(3)], 1)
    assert (state.occupancy.state, state.occupancy.stabilized_count) == (OccupancyState.SMALL_GROUP, 3)
    state, events = analyze(room, [person(i) for i in range(4)], 2)
    assert state.occupancy.state == OccupancyState.CROWD
    assert [e[0] for e in events] == [EventType.OCCUPANCY_STATE_CHANGED]


def test_unavailable_eof_and_stale_are_unknown_not_empty():
    room = RoomStateAnalyzer(config=RoomConfig(empty_debounce_seconds=0))
    analyze(room, [], 0)
    for index, (status, age) in enumerate([(CameraStatus.UNAVAILABLE, 0),
                                            (CameraStatus.EOF, 0),
                                            (CameraStatus.OK, 6)], start=1):
        state, events = analyze(room, [], index, status=status, age=age)
        assert state.occupancy.state == OccupancyState.UNKNOWN
        assert state.interaction.state.value == "unknown"
        assert not any(e[0] == EventType.ROOM_BECAME_EMPTY for e in events)


def test_spatial_centroid_bins_dispersion_overlap_and_tie():
    room = RoomStateAnalyzer(config=RoomConfig(horizontal_boundaries=(0.3, 0.7)))
    people = [person(1, x=0.1, y=0.5, proximity=0.2),
              person(2, x=0.5, y=0.5, proximity=0.4),
              person(3, x=0.9, y=0.5, proximity=0.6)]
    state, _ = analyze(room, people, 0, occupancy={"z": 2, "a": 2})
    spatial = state.spatial
    assert spatial.centroid == NormalizedPoint(x=0.5, y=0.5)
    assert spatial.horizontal_counts.model_dump() == {"left": 1, "centre": 1, "right": 1}
    assert spatial.dispersion == pytest.approx((0.32 / 3) ** 0.5)
    assert spatial.mean_proximity_score == pytest.approx(0.4)
    assert spatial.zone_occupancy == {"z": 2, "a": 2}
    assert spatial.most_occupied_zone_id == "a"
    empty, _ = analyze(room, [], 1, occupancy={"z": 0, "a": 0})
    assert empty.spatial.centroid is None
    assert empty.spatial.mean_proximity_score is None
    assert empty.spatial.dispersion is None
    assert empty.spatial.most_occupied_zone_id is None


def test_activity_weighting_coverage_smoothing_and_once_only_event():
    config = RoomConfig(activity_debounce_seconds=0.2, activity_smoothing_alpha=0.35)
    room = RoomStateAnalyzer(config=config)
    low = [person(1, score=0), person(2, score=0)]
    analyze(room, low, 0)
    state, _ = analyze(room, low, 0.2)
    assert state.activity.aggregate_activity_level == ActivityLevel.STILL
    high = [person(1, score=1), person(2, score=1)]
    state, _ = analyze(room, high, 0.3)
    assert state.activity.aggregate_activity_score == pytest.approx(1 - 0.65 ** 0.5)
    assert state.activity.aggregate_activity_level == ActivityLevel.STILL
    state, _ = analyze(room, low, 0.4)
    assert state.activity.aggregate_activity_level == ActivityLevel.STILL
    analyze(room, high, 1.0)
    state, events = analyze(room, high, 1.2)
    assert state.activity.aggregate_activity_level != ActivityLevel.STILL
    assert [e[0] for e in events] == [EventType.ROOM_ACTIVITY_CHANGED]
    _, events = analyze(room, high, 1.4)
    assert not any(e[0] == EventType.ROOM_ACTIVITY_CHANGED for e in events)

    room.reset()
    state, _ = analyze(room, [person(1, score=1), person(2, score=1, quality=PoseQuality.POOR)], 0)
    assert state.activity.contributing_count == 1
    assert state.activity.coverage_ratio == 0.5
    state, _ = analyze(room, [person(1, score=0), person(2, score=1, quality=PoseQuality.POOR)], 0.2)
    assert state.activity.aggregate_activity_score == pytest.approx(0.65)
    room.reset()
    state, _ = analyze(room, [person(1, score=1)] + [person(i) for i in range(2, 6)], 0)
    assert state.activity.aggregate_activity_level == ActivityLevel.UNKNOWN
    assert state.activity.aggregate_activity_score is None
    assert state.activity.coverage_ratio == 0.2


def test_activity_smoothing_uses_elapsed_time_not_frame_count():
    one_step = RoomStateAnalyzer()
    two_steps = RoomStateAnalyzer()
    analyze(one_step, [person(score=0)], 0)
    analyze(two_steps, [person(score=0)], 0)
    state_one, _ = analyze(one_step, [person(score=1)], 0.2)
    analyze(two_steps, [person(score=1)], 0.1)
    state_two, _ = analyze(two_steps, [person(score=1)], 0.2)
    assert state_one.activity.aggregate_activity_score == pytest.approx(0.35)
    assert state_two.activity.aggregate_activity_score == pytest.approx(0.35)


def test_motion_gesture_counts_and_interaction_precedence():
    zone = Zone(id="front", label="front", role="interaction",
                polygon=[(0, 0), (1, 0), (1, 1), (0, 1)])
    room = RoomStateAnalyzer(ZoneConfig(zones=[zone]),
                             RoomConfig(interaction_debounce_seconds=0))
    people = [person(1, left=True, right=True, wave=True, motion=MotionState.APPROACHING,
                     confirmed_zones=("front",)),
              person(2, left=True, motion=MotionState.DEPARTING),
              person(3, motion=MotionState.STATIONARY),
              person(4)]
    state, _ = analyze(room, people, 0)
    assert state.motion_summary.model_dump() == {"approaching": 1, "departing": 1,
                                                  "stationary": 1, "unknown": 1}
    assert state.gesture_summary.model_dump() == {"left_hand_raised": 2, "right_hand_raised": 1,
                                                   "both_hands_raised": 1, "recently_waving": 1,
                                                   "unknown": 0}
    assert state.interaction.state.value == "signal_detected"
    room.reset()
    state, _ = analyze(room, [person(1, confirmed_zones=("front",))], 0)
    assert state.interaction.state.value == "approaching"
    room.reset()
    state, _ = analyze(room, [person(1)], 0)
    assert state.interaction.state.value == "presence"


def test_interaction_debounce_and_unknown_gesture():
    room = RoomStateAnalyzer(config=RoomConfig(interaction_debounce_seconds=0.3))
    baseline = [person(1)]
    analyze(room, baseline, 0)
    state, _ = analyze(room, baseline, 0.3)
    assert state.interaction.state.value == "presence"
    state, _ = analyze(room, [person(1, left=True)], 0.4)
    assert state.interaction.state.value == "presence"
    state, _ = analyze(room, baseline, 0.5)
    assert state.interaction.state.value == "presence"
    analyze(room, [person(1, left=True)], 0.6)
    state, events = analyze(room, [person(1, left=True)], 0.9)
    assert state.interaction.state.value == "signal_detected"
    assert [e[0] for e in events if e[0] == EventType.INTERACTION_STATE_CHANGED] == [EventType.INTERACTION_STATE_CHANGED]
    unknown_person = person(2).model_copy(update={"pose_analysis": None})
    state, _ = analyze(room, [unknown_person], 1.0)
    assert state.gesture_summary.unknown == 1


def test_focus_margin_loss_switch_clear_tie_and_evidence():
    room = RoomStateAnalyzer(config=RoomConfig(focus_switch_margin=0.15,
                                                focus_loss_tolerance_seconds=0.5))
    state, _ = analyze(room, [person(2), person(1)], 0)
    assert state.focus_candidate is None  # presence and proximity alone do not nominate
    state, events = analyze(room, [person(2, left=True), person(1, left=True)], 0.1)
    assert state.focus_candidate.track_id == 1  # exact tie uses lowest temporary ID
    assert state.focus_candidate.evidence == ["one_hand_raised", "proximity"]
    assert [e[0] for e in events] == [EventType.FOCUS_CANDIDATE_CHANGED]
    state, events = analyze(room, [person(1, left=True), person(2, left=True, proximity=0.9)], 0.2)
    assert state.focus_candidate.track_id == 1
    assert not events
    state, events = analyze(room, [person(1, left=True), person(2, left=True, wave=True)], 0.3)
    assert state.focus_candidate.track_id == 2
    assert [e[0] for e in events] == [EventType.FOCUS_CANDIDATE_CHANGED]
    state, _ = analyze(room, [], 0.4)
    assert state.focus_candidate.track_id == 2 and not state.focus_candidate.is_visible
    state, events = analyze(room, [], 0.9)
    assert state.focus_candidate is None
    assert [e[0] for e in events] == [EventType.FOCUS_CANDIDATE_CHANGED]


def test_focus_survives_brief_cue_gap_while_track_visible():
    room = RoomStateAnalyzer(config=RoomConfig(focus_loss_tolerance_seconds=0.5))
    analyze(room, [person(3, left=True)], 0)
    state, events = analyze(room, [person(3)], 0.2)
    assert state.focus_candidate.track_id == 3
    assert state.focus_candidate.is_visible
    assert events == []
    state, events = analyze(room, [person(3)], 0.6)
    assert state.focus_candidate is None
    assert [e[0] for e in events] == [EventType.FOCUS_CANDIDATE_CHANGED]


def test_shared_event_buffer_reset_and_schema4_json():
    behaviour = BehaviourAnalyzer("session")
    room = RoomStateAnalyzer(config=RoomConfig(occupied_debounce_seconds=0,
                                                empty_debounce_seconds=0))
    raw = person(7, left=True)
    people, occupancy = behaviour.update([raw], at(0))
    room_state = room.update(people, CameraStatus.OK, at(0), occupancy,
                             behaviour.emit_room_event)
    assert behaviour.pending_event_count >= 1
    scene = snapshot("session", 0, "test", CameraStatus.OK, 640, 480, people,
                     timestamp_utc=at(0), zone_occupancy=occupancy,
                     room_state=room_state, events=behaviour.drain_events())
    value = json.loads(scene.model_dump_json(),
                       parse_constant=lambda x: pytest.fail(f"nonfinite: {x}"))
    assert value["schema_version"] == "4.0"
    assert value["room_state"]["occupancy"]["visible_count"] == 1
    assert [e["event_id"] for e in value["events"]] == list(range(1, len(value["events"]) + 1))
    room.update([], CameraStatus.OK, at(0.1), {}, behaviour.emit_room_event)
    room.update([], CameraStatus.OK, at(0.2), {}, behaviour.emit_room_event)
    assert behaviour.pending_event_count > 0  # events survive frames omitted by JSON rate limit
    behaviour.reset("new-session")
    room.reset()
    assert behaviour.drain_events() == [] and room.history_size == 0
    state, _ = analyze(room, [], 1)
    assert state.occupancy.state == OccupancyState.EMPTY


def test_video_terminal_time_after_source_timeline(tmp_path, monkeypatch, capsys):
    """A source timestamp can be ahead of wall time when playback is accelerated."""
    import numpy as np
    from reactive_vision.capture import CaptureResult
    from reactive_vision.main import arguments, run

    path = tmp_path / "clip.mp4"
    path.touch()
    args = arguments(["--video", str(path), "--headless"])

    class Source:
        label = "clip.mp4"
        reads = 0

        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            self.reads += 1
            if self.reads <= 2:
                return CaptureResult(CameraStatus.OK, np.zeros((32, 32, 3), dtype=np.uint8),
                                     source_frame_index=self.reads - 1,
                                     source_time_seconds=float(self.reads - 1))
            return CaptureResult(CameraStatus.EOF)

    class Tracker:
        def __init__(self, *_):
            pass

        def process(self, _frame):
            return []

    monkeypatch.setattr("reactive_vision.main.FrameSource", Source)
    monkeypatch.setattr("reactive_vision.main.PoseTracker", Tracker)
    monkeypatch.setattr("reactive_vision.main.time.sleep", lambda _: None)
    assert run(args) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[-1]["camera_status"] == "eof"
    assert records[-1]["room_state"]["occupancy"]["state"] == "unknown"


@pytest.mark.parametrize("changes", [
    {"small_group_max": 4, "crowd_min": 6},
    {"occupied_debounce_seconds": -1},
    {"horizontal_boundaries": (0.8, 0.2)},
    {"activity_thresholds": (0.3, 0.2, 0.6)},
    {"minimum_activity_coverage": 1.2},
    {"focus_weights": {"recent_wave": -0.1}},
])
def test_invalid_room_configuration(changes):
    with pytest.raises(ValidationError):
        RoomConfig(**changes)
