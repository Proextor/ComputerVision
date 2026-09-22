from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest
from pydantic import ValidationError

from reactive_vision.behaviour import BehaviourAnalyzer, BehaviourConfig
from reactive_vision.main import arguments, run, snapshot
from reactive_vision.pose_behaviour import PoseConfig, load_pose_config
from reactive_vision.render import draw_preview
from reactive_vision.schemas import (
    ActivityLevel, BoundingBox, CameraStatus, EventType, GestureStatus,
    Keypoint, MotionState, PersonObservation, PoseQuality,
)
from reactive_vision.tracking import COCO_KEYPOINTS


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
BASE_POINTS = (
    (0.50, 0.27), (0.47, 0.25), (0.53, 0.25), (0.44, 0.26), (0.56, 0.26),
    (0.42, 0.40), (0.58, 0.40), (0.36, 0.52), (0.64, 0.52),
    (0.32, 0.70), (0.68, 0.70), (0.45, 0.65), (0.55, 0.65),
    (0.44, 0.78), (0.56, 0.78), (0.43, 0.89), (0.57, 0.89),
)


def at(seconds: float) -> datetime:
    return BASE + timedelta(seconds=seconds)


def pose_person(
    *, track_id: int | None = 1, left_up: bool = False, right_up: bool = False,
    left_x: float | None = None, right_x: float | None = None,
    left_y: float | None = None, right_y: float | None = None,
    shift_x: float = 0, missing: set[int] | None = None,
    low_confidence: set[int] | None = None, point_count: int = 17,
) -> PersonObservation:
    points = [list(point) for point in BASE_POINTS]
    if left_up:
        points[7][1], points[9][1] = 0.32, 0.25
    if right_up:
        points[8][1], points[10][1] = 0.32, 0.25
    if left_x is not None:
        points[9][0] = left_x
    if right_x is not None:
        points[10][0] = right_x
    if left_y is not None:
        points[9][1] = left_y
    if right_y is not None:
        points[10][1] = right_y
    missing = missing or set()
    low_confidence = low_confidence or set()
    keypoints = [Keypoint(
        index=index, name=COCO_KEYPOINTS[index],
        x=None if index in low_confidence else x + shift_x,
        y=None if index in low_confidence else y,
        confidence=0.1 if index in low_confidence else 0.9,
    ) for index, (x, y) in enumerate(points[:point_count]) if index not in missing]
    return PersonObservation(
        track_id=track_id,
        bbox=BoundingBox(x1=0.15 + shift_x, y1=0.10, x2=0.85 + shift_x, y2=0.95),
        confidence=0.95, keypoints=keypoints,
    )


def update(analyzer: BehaviourAnalyzer, seconds: float, **kwargs):
    return analyzer.update([pose_person(**kwargs)], at(seconds))[0][0]


def events(analyzer: BehaviourAnalyzer, kind: EventType | None = None):
    drained = analyzer.drain_events()
    return [event for event in drained if event.type == kind] if kind is not None else drained


def test_pose_quality_and_missing_required_wrist():
    analyzer = BehaviourAnalyzer("s")
    good = update(analyzer, 0)
    quality = good.pose_analysis.quality
    assert quality.usable_keypoint_count == 17
    assert quality.usable_keypoint_ratio == 1
    assert quality.pose_confidence == pytest.approx(0.9)
    assert quality.pose_quality == PoseQuality.GOOD
    partial = update(analyzer, 0.2, point_count=8)
    assert partial.pose_analysis.quality.pose_quality == PoseQuality.PARTIAL
    poor = update(analyzer, 0.4, point_count=3)
    assert poor.pose_analysis.quality.pose_quality == PoseQuality.POOR
    assert poor.pose_analysis.gestures.left_hand_raised == GestureStatus.UNKNOWN
    assert poor.pose_analysis.activity.activity_score is None
    absent = PersonObservation(track_id=None, bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.9, y2=0.9))
    untracked = analyzer.update([absent], at(0.6))[0][0]
    assert untracked.pose_analysis.quality.pose_quality == PoseQuality.UNAVAILABLE
    assert untracked.pose_analysis.quality.pose_confidence is None
    assert untracked.pose_analysis.activity.activity_level == ActivityLevel.UNKNOWN
    low_wrist = update(analyzer, 0.8, left_up=True, low_confidence={9})
    assert low_wrist.pose_analysis.quality.usable_keypoint_count == 16
    assert low_wrist.pose_analysis.gestures.left_hand_raised == GestureStatus.UNKNOWN


def test_configurable_confidence_and_invalid_pose_config(tmp_path):
    strict = BehaviourAnalyzer("s", pose_config=PoseConfig(min_keypoint_confidence=0.95))
    observed = update(strict, 0, left_up=True)
    assert observed.pose_analysis.quality.usable_keypoint_count == 0
    assert observed.pose_analysis.quality.pose_quality == PoseQuality.POOR
    assert observed.pose_analysis.gestures.left_hand_raised == GestureStatus.UNKNOWN
    with pytest.raises(ValidationError):
        PoseConfig(activity_thresholds=(0.3, 0.2, 0.7))
    path = tmp_path / "bad-pose.json"
    path.write_text('{"wave_min_reversals": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid pose configuration"):
        load_pose_config(path)
    with pytest.raises(SystemExit) as failure:
        arguments(["--pose-config", str(path)])
    assert failure.value.code == 2


@pytest.mark.parametrize("side", ["left", "right"])
def test_independent_hand_raise_hold_and_release(side):
    analyzer = BehaviourAnalyzer("s")
    down = update(analyzer, 0)
    assert down.pose_analysis.gestures.left_hand_raised == GestureStatus.INACTIVE
    assert down.pose_analysis.gestures.right_hand_raised == GestureStatus.INACTIVE
    events(analyzer)
    kwargs = {f"{side}_up": True}
    pending = update(analyzer, 0.1, **kwargs)
    assert getattr(pending.pose_analysis.gestures, f"{side}_hand_raised") != GestureStatus.ACTIVE
    raised = update(analyzer, 0.45, **kwargs)
    assert getattr(raised.pose_analysis.gestures, f"{side}_hand_raised") == GestureStatus.ACTIVE
    assert getattr(raised.pose_analysis.gestures, f"{'right' if side == 'left' else 'left'}_hand_raised") == GestureStatus.INACTIVE
    started = events(analyzer, EventType.HAND_RAISE_STARTED)
    assert len(started) == 1 and started[0].metadata["side"] == side
    for time in (0.6, 0.8):
        held = update(analyzer, time, **kwargs)
        assert getattr(held.pose_analysis.gestures, f"{side}_hand_raised") == GestureStatus.ACTIVE
    assert events(analyzer, EventType.HAND_RAISE_STARTED) == []
    update(analyzer, 0.9)
    lowered = update(analyzer, 1.3)
    assert getattr(lowered.pose_analysis.gestures, f"{side}_hand_raised") == GestureStatus.INACTIVE
    ended = events(analyzer, EventType.HAND_RAISE_ENDED)
    assert len(ended) == 1 and ended[0].metadata["side"] == side


def test_both_hands_and_shoulder_margin_do_not_flicker():
    analyzer = BehaviourAnalyzer("s")
    update(analyzer, 0)
    events(analyzer)
    for time in (0.1, 0.45):
        both = update(analyzer, time, left_up=True, right_up=True)
    assert both.pose_analysis.gestures.both_hands_raised == GestureStatus.ACTIVE
    emitted = events(analyzer)
    assert [event.type for event in emitted].count(EventType.BOTH_HANDS_RAISED_STARTED) == 1
    assert [event.type for event in emitted].count(EventType.HAND_RAISE_STARTED) == 2
    for time in (0.6, 0.8, 1.0):
        update(analyzer, time, left_up=True, right_up=True)
    assert not any(event.type == EventType.BOTH_HANDS_RAISED_STARTED for event in events(analyzer))
    update(analyzer, 1.2, right_up=True)
    lowered = update(analyzer, 1.6, right_up=True)
    assert lowered.pose_analysis.gestures.both_hands_raised == GestureStatus.INACTIVE
    assert [event.type for event in events(analyzer)].count(EventType.BOTH_HANDS_RAISED_ENDED) == 1

    margin = BehaviourAnalyzer("m")
    update(margin, 0)
    events(margin)
    for i in range(1, 6):
        near = update(margin, i * 0.2, left_y=0.37 if i % 2 else 0.42)
        assert near.pose_analysis.gestures.left_hand_raised != GestureStatus.ACTIVE
    assert not any(event.type == EventType.HAND_RAISE_STARTED for event in events(margin))


def test_keypoint_loss_tolerance_and_unknown_without_false_end():
    analyzer = BehaviourAnalyzer("s")
    update(analyzer, 0, left_up=True)
    raised = update(analyzer, 0.4, left_up=True)
    assert raised.pose_analysis.gestures.left_hand_raised == GestureStatus.ACTIVE
    events(analyzer)
    brief = update(analyzer, 0.5, left_up=True, low_confidence={9})
    assert brief.pose_analysis.gestures.left_hand_raised == GestureStatus.ACTIVE
    assert brief.pose_analysis.gestures.left_wave == GestureStatus.UNKNOWN
    expired = update(analyzer, 1.1, left_up=True, low_confidence={9})
    assert expired.pose_analysis.gestures.left_hand_raised == GestureStatus.UNKNOWN
    assert not any(event.type == EventType.HAND_RAISE_ENDED for event in events(analyzer))


def test_short_tracking_dropout_keeps_raised_state_without_new_start():
    analyzer = BehaviourAnalyzer("s")
    update(analyzer, 0, left_up=True)
    update(analyzer, 0.4, left_up=True)
    assert len(events(analyzer, EventType.HAND_RAISE_STARTED)) == 1
    analyzer.update([], at(0.5))
    returned = update(analyzer, 0.7, left_up=True)
    assert returned.pose_analysis.gestures.left_hand_raised == GestureStatus.ACTIVE
    assert not any(event.type in (EventType.HAND_RAISE_STARTED, EventType.HAND_RAISE_ENDED)
                   for event in events(analyzer))


def test_stationary_hand_and_one_direction_are_not_waves():
    for xs in ([0.27] * 14, [0.20 + i * 0.025 for i in range(14)]):
        analyzer = BehaviourAnalyzer("s")
        states = []
        for i, x in enumerate(xs):
            observed = update(analyzer, i * 0.15, left_up=True, left_x=x)
            states.append(observed.pose_analysis.gestures.left_wave)
        assert GestureStatus.ACTIVE not in states
        assert not any(event.type == EventType.WAVE_DETECTED for event in events(analyzer))


@pytest.mark.parametrize("side", ["left", "right"])
def test_reversals_produce_one_side_specific_wave_with_ttl_and_cooldown(side):
    analyzer = BehaviourAnalyzer("s")
    xs = [0.24, 0.24, 0.54, 0.54] * 8  # continuous wave extends beyond cooldown
    seen_active = False
    for i, x in enumerate(xs):
        kwargs = {f"{side}_up": True, f"{side}_x": x}
        observed = update(analyzer, i * 0.15, **kwargs)
        seen_active |= getattr(observed.pose_analysis.gestures, f"{side}_wave") == GestureStatus.ACTIVE
    waved = [event for event in events(analyzer) if event.type == EventType.WAVE_DETECTED]
    assert seen_active
    assert len(waved) == 1
    assert waved[0].metadata["side"] == side
    assert waved[0].metadata["reversals"] >= 2
    assert waved[0].metadata["amplitude_ratio"] >= 0.15


def test_whole_body_translation_does_not_become_wave():
    analyzer = BehaviourAnalyzer("s")
    scores = []
    for i in range(10):
        observed = update(analyzer, i * 0.15, left_up=True,
                          left_x=0.28 + i * 0.01, shift_x=i * 0.01)
        assert observed.pose_analysis.gestures.left_wave != GestureStatus.ACTIVE
        if observed.pose_analysis.activity.activity_score is not None:
            scores.append(observed.pose_analysis.activity.activity_score)
    assert not any(event.type == EventType.WAVE_DETECTED for event in events(analyzer))
    assert scores and max(scores) < 0.30


def test_wave_display_ttl_then_inactive_without_duplicate_event():
    analyzer = BehaviourAnalyzer("s")
    xs = [0.24, 0.24, 0.54, 0.54, 0.24, 0.24, 0.54, 0.54,
          0.24, 0.24, 0.54, 0.54, 0.24, 0.24]
    detected_at = None
    for i, x in enumerate(xs):
        update(analyzer, i * 0.15, left_up=True, left_x=x)
        if any(event.type == EventType.WAVE_DETECTED for event in events(analyzer)):
            detected_at = i * 0.15
            break
    assert detected_at is not None
    during = update(analyzer, detected_at + 0.3, left_up=True, left_x=0.24)
    assert during.pose_analysis.gestures.left_wave == GestureStatus.ACTIVE
    after = update(analyzer, detected_at + 0.9, left_up=True, left_x=0.24)
    assert after.pose_analysis.gestures.left_wave == GestureStatus.INACTIVE
    assert not any(event.type == EventType.WAVE_DETECTED for event in events(analyzer))


def test_activity_jitter_joint_movement_elapsed_time_and_bounds():
    quiet = BehaviourAnalyzer("quiet")
    quiet_scores = []
    for i in range(9):
        observed = update(quiet, i * 0.2, left_x=0.32 + (0.001 if i % 2 else -0.001))
        if observed.pose_analysis.activity.activity_score is not None:
            quiet_scores.append(observed.pose_analysis.activity.activity_score)
    assert quiet_scores and max(quiet_scores) < 0.12
    assert observed.pose_analysis.activity.activity_level == ActivityLevel.STILL

    moving = BehaviourAnalyzer("moving")
    scores = []
    for i in range(10):
        observed = update(moving, i * 0.2, left_up=True,
                          left_x=0.22 if i % 2 else 0.50,
                          right_x=0.48 if i % 2 else 0.76)
        activity = observed.pose_analysis.activity
        if activity.activity_score is not None:
            scores.append(activity.activity_score)
            assert 0 <= activity.activity_score <= 1
            assert activity.contributing_joint_count >= 6
    assert max(scores) > max(quiet_scores)
    assert any(event.type == EventType.ACTIVITY_LEVEL_CHANGED for event in events(moving))

    fast, slow = BehaviourAnalyzer("fast"), BehaviourAnalyzer("slow")
    update(fast, 0, left_x=0.25)
    update(slow, 0, left_x=0.25)
    fast_score = update(fast, 0.1, left_x=0.55).pose_analysis.activity.activity_score
    slow_score = update(slow, 0.5, left_x=0.55).pose_analysis.activity.activity_score
    assert fast_score is not None and slow_score is not None and fast_score > slow_score


def test_insufficient_pose_history_and_track_expiry_reset():
    analyzer = BehaviourAnalyzer("s")
    first = update(analyzer, 0, left_up=True)
    assert first.pose_analysis.activity.activity_level == ActivityLevel.UNKNOWN
    assert first.pose_analysis.activity.activity_score is None
    raised = update(analyzer, 0.4, left_up=True)
    assert raised.pose_analysis.gestures.left_hand_raised == GestureStatus.ACTIVE
    events(analyzer)
    fresh = update(analyzer, 1.5)
    assert fresh.pose_analysis.gestures.left_hand_raised == GestureStatus.INACTIVE
    assert fresh.pose_analysis.activity.activity_score is None
    kinds = [event.type for event in events(analyzer)]
    assert kinds.count(EventType.PERSON_DISAPPEARED) == 1
    assert kinds.count(EventType.PERSON_APPEARED) == 1
    assert EventType.HAND_RAISE_ENDED not in kinds
    analyzer.reset("new")
    after_reset = update(analyzer, 2.0, left_up=True)
    assert after_reset.pose_analysis.gestures.left_hand_raised == GestureStatus.UNKNOWN
    assert after_reset.pose_analysis.activity.activity_score is None
    assert [(event.event_id, event.session_id) for event in events(analyzer)] == [(1, "new")]


def test_schema_3_json_preview_and_explicit_unavailable():
    analyzer = BehaviourAnalyzer("s")
    observed = update(analyzer, 0, left_up=True)
    scene = snapshot("s", 0, "webcam:0", CameraStatus.OK, 640, 480, [observed],
                     timestamp_utc=at(0), events=events(analyzer))
    encoded = scene.model_dump_json(exclude_none=False)
    decoded = json.loads(encoded, parse_constant=lambda value: pytest.fail(f"non-finite JSON: {value}"))
    assert decoded["schema_version"] == "4.0"
    assert decoded["people"][0]["pose_analysis"]["quality"]["pose_quality"] == "good"
    assert decoded["people"][0]["pose_analysis"]["activity"]["activity_score"] is None
    image = draw_preview(np.zeros((480, 640, 3), dtype=np.uint8), scene)
    assert image.shape == (480, 640, 3) and np.any(image)


def test_hand_event_buffered_across_json_intervals(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "pose.json"
    config_path.write_text(json.dumps({"gesture_activation_seconds": 0}), encoding="utf-8")
    args = arguments(["--pose-config", str(config_path), "--rate", "0.00001", "--headless"])

    class FakeSource:
        label = "webcam:0"

        def __init__(self, *_):
            self.count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            from reactive_vision.capture import CaptureResult
            self.count += 1
            if self.count <= 3:
                return CaptureResult(CameraStatus.OK, np.zeros((100, 100, 3), dtype=np.uint8))
            return CaptureResult(CameraStatus.EOF)

    class FakeTracker:
        def __init__(self, *_):
            self.count = 0

        def process(self, _frame):
            self.count += 1
            return [pose_person(left_up=self.count > 1)]

    monkeypatch.setattr("reactive_vision.main.FrameSource", FakeSource)
    monkeypatch.setattr("reactive_vision.main.PoseTracker", FakeTracker)
    assert run(args) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["camera_status"] for record in records] == ["ok", "eof"]
    assert not any(event["type"] == "hand_raise_started" for event in records[0]["events"])
    starts = [event for event in records[1]["events"] if event["type"] == "hand_raise_started"]
    assert len(starts) == 1 and starts[0]["metadata"]["side"] == "left"
