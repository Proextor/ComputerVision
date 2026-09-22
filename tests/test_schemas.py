import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from reactive_vision.capture import FrameSource
from reactive_vision.main import snapshot
from reactive_vision.schemas import CameraStatus, SceneSnapshot, normalized_box
from reactive_vision.tracking import PoseTracker, people_from_result


def fake_result(xyxy, confidences=None, ids=None, keypoints=None, keypoint_confidences=None):
    boxes = SimpleNamespace(
        xyxy=np.asarray(xyxy, dtype=float).reshape(-1, 4),
        conf=None if confidences is None else np.asarray(confidences, dtype=float),
        id=None if ids is None else np.asarray(ids, dtype=float),
        cls=np.zeros(len(xyxy), dtype=float),
    )
    pose = None if keypoints is None else SimpleNamespace(
        xy=np.asarray(keypoints, dtype=float),
        conf=None if keypoint_confidences is None else np.asarray(keypoint_confidences, dtype=float),
    )
    return SimpleNamespace(boxes=boxes, keypoints=pose)


def test_normalized_box_clips_to_frame_and_rejects_invalid():
    box = normalized_box([-10, 20, 110, 60], 100, 80)
    assert box is not None
    assert box.model_dump() == {"x1": 0.0, "y1": 0.25, "x2": 1.0, "y2": 0.75}
    assert normalized_box([float("nan"), 0, 10, 10], 100, 80) is None
    assert normalized_box([110, 0, 120, 10], 100, 80) is None


def test_empty_scene_is_valid_and_camera_failure_is_distinct():
    empty = people_from_result(fake_result([]), 640, 480)
    scene = snapshot("session", 0, "webcam:0", CameraStatus.OK, 640, 480, empty)
    assert scene.visible_count == 0
    assert scene.people == []
    assert scene.camera_status == CameraStatus.OK
    failure = snapshot("session", 1, "webcam:0", CameraStatus.UNAVAILABLE)
    assert failure.visible_count == 0
    assert failure.camera_status == CameraStatus.UNAVAILABLE
    assert failure.frame_width is None


def test_missing_id_and_keypoints_are_explicit_nulls():
    people = people_from_result(fake_result([[10, 20, 110, 220]], [0.8]), 200, 400)
    assert len(people) == 1
    assert people[0].track_id is None
    assert people[0].keypoints is None
    assert people[0].confidence == 0.8


def test_low_confidence_keypoint_has_no_fabricated_coordinates():
    result = fake_result(
        [[0, 0, 100, 100]], [0.95], [7],
        [[[50, 25], [0, 0], [float("nan"), 20]]],
        [[0.9, 0.1, float("nan")]],
    )
    person = people_from_result(result, 100, 100)[0]
    assert person.track_id == 7
    assert person.keypoints is not None
    assert person.keypoints[0].x == 0.5
    assert person.keypoints[0].confidence == 0.9
    assert person.keypoints[1].x is None
    assert person.keypoints[1].confidence == 0.1
    assert person.keypoints[2].x is None
    assert person.keypoints[2].confidence is None


def test_json_has_only_native_finite_values_and_nulls():
    person = people_from_result(
        fake_result([[10, 20, 110, 220]], [np.float32(0.7)], [float("nan")],
                    [[[20, 30]]], [[float("nan")]]), 200, 400
    )[0]
    scene = snapshot("session", 3, "video.mp4", CameraStatus.OK, 200, 400,
                     [person], np.float32(12.5), 12, np.float32(0.4))
    encoded = scene.model_dump_json(exclude_none=False)
    decoded = json.loads(encoded, parse_constant=lambda value: pytest.fail(f"invalid JSON: {value}"))
    assert decoded["people"][0]["track_id"] is None
    assert decoded["people"][0]["keypoints"][0]["confidence"] is None
    assert decoded["processing_fps"] == 12.5
    assert decoded["timestamp_utc"].endswith("Z")


def test_schema_rejects_count_mismatch_and_nonfinite_values():
    with pytest.raises(ValidationError):
        snapshot("s", 0, "webcam:0", CameraStatus.OK, 100, 100, processing_fps=float("nan"))
    with pytest.raises(ValidationError):
        scene = snapshot("s", 0, "webcam:0", CameraStatus.OK, 100, 100)
        SceneSnapshot.model_validate({**scene.model_dump(), "visible_count": 1})


def test_tracker_uses_bytetrack_and_persists_without_loading_weights():
    calls = []

    class FakeModel:
        def track(self, frame, **kwargs):
            calls.append(kwargs)
            return [fake_result([])]

    tracker = PoseTracker.__new__(PoseTracker)
    tracker.model = FakeModel()
    tracker.imgsz = 320
    tracker.confidence = 0.25
    tracker.device = "cpu"
    tracker.keypoint_threshold = 0.25
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert tracker.process(frame) == []
    assert tracker.process(frame) == []
    assert len(calls) == 2
    assert all(call["persist"] is True and call["tracker"] == "bytetrack.yaml" for call in calls)


def test_capture_releases_on_exit_and_distinguishes_failed_read(monkeypatch):
    class FakeCapture:
        released = False

        def isOpened(self):
            return True

        def set(self, *_):
            return True

        def read(self):
            return False, None

        def release(self):
            self.released = True

    fake = FakeCapture()
    monkeypatch.setattr("reactive_vision.capture.cv2.VideoCapture", lambda *_: fake)
    with FrameSource(0, None, 640, 480) as source:
        assert source.read().status == CameraStatus.READ_ERROR
    assert fake.released


def test_video_early_read_failure_is_not_eof(monkeypatch):
    class FakeVideo:
        def isOpened(self):
            return True

        def get(self, _property):
            return 10

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr("reactive_vision.capture.cv2.VideoCapture", lambda *_: FakeVideo())
    with FrameSource(0, Path("example.mp4"), 640, 480) as source:
        assert source.read().status == CameraStatus.READ_ERROR
