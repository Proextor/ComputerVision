from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest
from pydantic import ValidationError

from reactive_vision.behaviour import BehaviourAnalyzer, BehaviourConfig
from reactive_vision.main import arguments, run, snapshot
from reactive_vision.render import draw_preview
from reactive_vision.schemas import (
    BoundingBox, CameraStatus, EventType, MotionState,
    NormalizedPoint, PersonObservation, SceneSnapshot,
)
from reactive_vision.zones import Zone, ZoneConfig, contains, load_zones


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def at(seconds: float) -> datetime:
    return BASE + timedelta(seconds=seconds)


def person(track_id: int | None = 1, x: float = 0.25, height: float = 0.3, bottom: float = 0.9) -> PersonObservation:
    return PersonObservation(
        track_id=track_id,
        bbox=BoundingBox(x1=x - 0.08, x2=x + 0.08, y1=bottom - height, y2=bottom),
        confidence=0.9,
    )


def zone_config() -> ZoneConfig:
    return ZoneConfig(zones=[Zone(
        id="left", label="Left side", role="interaction",
        polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
    )])


def event_types(analyzer: BehaviourAnalyzer) -> list[EventType]:
    return [event.type for event in analyzer.drain_events()]


def test_polygon_membership_and_overlapping_occupancy():
    left = zone_config().zones[0]
    assert contains(left, NormalizedPoint(x=0.25, y=0.9))
    assert contains(left, NormalizedPoint(x=0.5, y=0.9))  # edge included
    assert not contains(left, NormalizedPoint(x=0.75, y=0.9))
    right = Zone(id="right", label="Right", polygon=[(0.2, 0), (1, 0), (1, 1), (0.2, 1)])
    analyzer = BehaviourAnalyzer("s", ZoneConfig(zones=[left, right]))
    observed, occupancy = analyzer.update([person(x=0.25), person(track_id=2, x=0.75)], at(0))
    assert occupancy == {"left": 1, "right": 2}
    assert observed[0].zones == ["left", "right"]
    assert observed[0].anchor_point == NormalizedPoint(x=0.25, y=0.9)
    assert observed[0].bbox_area_ratio == pytest.approx(0.16 * 0.3)
    assert observed[0].proximity_score == pytest.approx(0.3)


@pytest.mark.parametrize("data", [
    {"zones": [{"id": "bad id", "label": "A", "polygon": [[0, 0], [1, 0], [0, 1]]}]},
    {"zones": [{"id": "a", "label": "  ", "polygon": [[0, 0], [1, 0], [0, 1]]}]},
    {"zones": [{"id": "a", "label": "A", "polygon": [[0, 0], [1, 0]]}]},
    {"zones": [{"id": "a", "label": "A", "polygon": [[0, 0], [1.1, 0], [0, 1]]}]},
    {"zones": [{"id": "a", "label": "A", "polygon": [[0, 0], [0.5, 0.5], [1, 1]]}]},
    {"zones": [{"id": "a", "label": "A", "polygon": [[0, 0], [1, 1], [0, 1], [1, 0.2]]}]},
    {},
    {"zones": [
        {"id": "a", "label": "A", "polygon": [[0, 0], [1, 0], [0, 1]]},
        {"id": "a", "label": "B", "polygon": [[0, 0], [1, 0], [0, 1]]},
    ]},
])
def test_invalid_zone_configuration(data):
    with pytest.raises(ValidationError):
        ZoneConfig.model_validate(data)


def test_zone_file_errors_are_useful(tmp_path):
    path = tmp_path / "zones.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid zone configuration"):
        load_zones(path)
    with pytest.raises(SystemExit) as failure:
        arguments(["--zones", str(path)])
    assert failure.value.code == 2


def test_zone_enter_exit_debounce_and_no_repeated_events():
    analyzer = BehaviourAnalyzer("s", zone_config())
    inside = person(x=0.25)
    outside = person(x=0.75)
    first, _ = analyzer.update([inside], at(0))
    assert first[0].zones == ["left"]
    assert first[0].confirmed_zones == []
    assert event_types(analyzer) == [EventType.PERSON_APPEARED]
    analyzer.update([inside], at(0.2))
    assert event_types(analyzer) == []
    entered, _ = analyzer.update([inside], at(0.4))
    assert entered[0].confirmed_zones == ["left"]
    assert entered[0].dwell_seconds == {"left": 0.0}
    assert event_types(analyzer) == [EventType.ZONE_ENTERED]
    analyzer.update([inside], at(0.5))
    assert event_types(analyzer) == []
    analyzer.update([outside], at(0.6))
    analyzer.update([inside], at(0.7))  # boundary jitter cancels exit
    analyzer.update([outside], at(0.8))
    assert event_types(analyzer) == []
    exited, _ = analyzer.update([outside], at(1.2))
    assert exited[0].confirmed_zones == []
    assert exited[0].dwell_seconds == {}
    assert event_types(analyzer) == [EventType.ZONE_EXITED]
    analyzer.update([outside], at(1.4))
    assert EventType.ZONE_EXITED not in event_types(analyzer)


def test_missing_observation_interrupts_pending_zone_confirmation():
    analyzer = BehaviourAnalyzer("s", zone_config())
    analyzer.update([person()], at(0))
    event_types(analyzer)
    analyzer.update([], at(0.2))
    returned, _ = analyzer.update([person()], at(0.6))
    assert returned[0].confirmed_zones == []
    assert event_types(analyzer) == []
    confirmed, _ = analyzer.update([person()], at(1.0))
    assert confirmed[0].confirmed_zones == ["left"]
    assert event_types(analyzer) == [EventType.ZONE_ENTERED]


def test_observed_exit_only_and_grace_period():
    analyzer = BehaviourAnalyzer("s", zone_config(), BehaviourConfig(zone_debounce_seconds=0))
    analyzer.update([person()], at(0))
    assert event_types(analyzer) == [EventType.PERSON_APPEARED, EventType.ZONE_ENTERED]
    analyzer.update([], at(0.2))
    assert event_types(analyzer) == []
    returned, _ = analyzer.update([person()], at(0.4))
    assert returned[0].confirmed_zones == ["left"]
    assert returned[0].dwell_seconds["left"] == pytest.approx(0.4)
    assert event_types(analyzer) == []
    analyzer.update([], at(1.2))  # >0.75 seconds since last observation
    assert event_types(analyzer) == [EventType.PERSON_DISAPPEARED]
    analyzer.update([], at(1.4))
    assert event_types(analyzer) == []


def test_return_after_grace_is_new_appearance():
    analyzer = BehaviourAnalyzer("s")
    analyzer.update([person()], at(0))
    event_types(analyzer)
    analyzer.update([person()], at(1.0))
    assert event_types(analyzer) == [EventType.PERSON_DISAPPEARED, EventType.PERSON_APPEARED]


def test_short_history_is_unknown_and_velocity_uses_seconds():
    analyzer = BehaviourAnalyzer("s")
    first, _ = analyzer.update([person(x=0.2)], at(0))
    assert first[0].motion_state == MotionState.UNKNOWN
    assert first[0].velocity_x is None
    second, _ = analyzer.update([person(x=0.3)], at(0.5))
    assert second[0].motion_state == MotionState.UNKNOWN
    assert second[0].velocity_x == pytest.approx(0.07)  # EMA alpha .35 / 0.5 s
    assert second[0].speed == pytest.approx(0.07)


def test_untracked_detection_gets_current_spatial_fields_without_history_events():
    analyzer = BehaviourAnalyzer("s", zone_config())
    observed, occupancy = analyzer.update([person(track_id=None)], at(0))
    assert occupancy == {"left": 1}
    assert observed[0].zones == ["left"]
    assert observed[0].confirmed_zones == []
    assert observed[0].motion_state == MotionState.UNKNOWN
    assert observed[0].velocity_x is None
    assert analyzer.drain_events() == []


def test_stationary_with_minor_jitter_has_one_transition():
    analyzer = BehaviourAnalyzer("s")
    states = []
    for i in range(10):
        observed, _ = analyzer.update([person(x=0.25 + (0.002 if i % 2 else -0.002),
                                              height=0.3 + (0.001 if i % 3 else -0.001))], at(i * 0.2))
        states.append(observed[0].motion_state)
    assert states[0] == MotionState.UNKNOWN
    assert states[-1] == MotionState.STATIONARY
    assert event_types(analyzer).count(EventType.BECAME_STATIONARY) == 1


def test_single_scale_spike_does_not_change_stationary_state():
    analyzer = BehaviourAnalyzer("s")
    states = []
    for i in range(17):
        height = 0.5 if i == 10 else 0.3
        observed, _ = analyzer.update([person(height=height)], at(i * 0.2))
        states.append(observed[0].motion_state)
    assert states[5:] == [MotionState.STATIONARY] * 12
    assert EventType.APPROACH_STARTED not in event_types(analyzer)


@pytest.mark.parametrize(("heights", "expected_state", "expected_event"), [
    ([0.18, 0.20, 0.23, 0.27, 0.31, 0.35, 0.40, 0.45, 0.50, 0.55], MotionState.APPROACHING, EventType.APPROACH_STARTED),
    ([0.55, 0.50, 0.45, 0.40, 0.35, 0.31, 0.27, 0.23, 0.20, 0.18], MotionState.DEPARTING, EventType.DEPARTURE_STARTED),
])
def test_camera_relative_scale_motion(heights, expected_state, expected_event):
    analyzer = BehaviourAnalyzer("s")
    states = []
    for i, height in enumerate(heights):
        observed, _ = analyzer.update([person(height=height)], at(i * 0.2))
        states.append(observed[0].motion_state)
    assert states[-1] == expected_state
    assert event_types(analyzer).count(expected_event) == 1


def test_events_buffer_until_drained_and_reset_between_sessions():
    analyzer = BehaviourAnalyzer("first", zone_config(), BehaviourConfig(zone_debounce_seconds=0.3))
    analyzer.update([person()], at(0))
    analyzer.update([person()], at(0.4))
    events = analyzer.drain_events()
    assert [event.type for event in events] == [EventType.PERSON_APPEARED, EventType.ZONE_ENTERED]
    assert [event.event_id for event in events] == [1, 2]
    assert analyzer.drain_events() == []
    analyzer.update([person()], at(0.5))
    analyzer.reset("second")
    assert analyzer.drain_events() == []
    observed, _ = analyzer.update([person()], at(1.0))
    assert observed[0].motion_state == MotionState.UNKNOWN
    assert observed[0].confirmed_zones == []
    assert [(event.event_id, event.session_id) for event in analyzer.drain_events()] == [(1, "second")]


def test_extended_snapshot_json_and_empty_scene():
    analyzer = BehaviourAnalyzer("s", zone_config(), BehaviourConfig(zone_debounce_seconds=0))
    observed, occupancy = analyzer.update([person()], at(0))
    scene = snapshot("s", 0, "webcam:0", CameraStatus.OK, 640, 480, observed,
                     timestamp_utc=at(0), zone_occupancy=occupancy,
                     events=analyzer.drain_events())
    decoded = json.loads(scene.model_dump_json(exclude_none=False),
                         parse_constant=lambda value: pytest.fail(f"non-finite JSON: {value}"))
    assert decoded["schema_version"] == "4.0"
    assert decoded["zone_occupancy"] == {"left": 1}
    assert decoded["people"][0]["anchor_point"] == {"x": 0.25, "y": 0.9}
    assert [event["type"] for event in decoded["events"]] == ["person_appeared", "zone_entered"]
    empty_people, empty_occupancy = analyzer.update([], at(0.2))
    empty = snapshot("s", 1, "webcam:0", CameraStatus.OK, 640, 480, empty_people,
                     timestamp_utc=at(0.2), zone_occupancy=empty_occupancy)
    assert empty.visible_count == 0
    assert empty.zone_occupancy == {"left": 0}
    assert empty.camera_status == CameraStatus.OK
    preview = draw_preview(np.zeros((480, 640, 3), dtype=np.uint8), scene,
                           zones=zone_config().zones)
    assert preview.shape == (480, 640, 3)
    assert np.any(preview != 0)


def test_events_survive_rate_limited_frames_until_terminal_emission(tmp_path, monkeypatch, capsys):
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(zone_config().model_dump()), encoding="utf-8")
    args = arguments(["--zones", str(path), "--zone-debounce", "0", "--rate", "0.00001", "--headless"])

    class FakeSource:
        label = "webcam:0"
        count = 0

        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            from reactive_vision.capture import CaptureResult
            self.count += 1
            if self.count <= 2:
                return CaptureResult(CameraStatus.OK, np.zeros((100, 100, 3), dtype=np.uint8))
            return CaptureResult(CameraStatus.EOF)

    class FakeTracker:
        count = 0

        def __init__(self, *_):
            pass

        def process(self, _frame):
            self.count += 1
            return [] if self.count == 1 else [person()]

    monkeypatch.setattr("reactive_vision.main.FrameSource", FakeSource)
    monkeypatch.setattr("reactive_vision.main.PoseTracker", FakeTracker)
    assert run(args) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["camera_status"] for record in records] == ["ok", "eof"]
    assert records[1]["room_state"]["occupancy"]["state"] == "unknown"
    assert records[0]["events"] == []
    assert [event["type"] for event in records[1]["events"]] == [
        "person_appeared", "zone_entered", "focus_candidate_changed",
    ]
