"""Ultralytics pose + ByteTrack adapter and result conversion."""

from contextlib import redirect_stdout
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .schemas import Keypoint, PersonObservation, clamp01, finite_or_none, normalized_box


COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _keypoints_for_person(
    xy: np.ndarray | None,
    confidences: np.ndarray | None,
    person_index: int,
    width: int,
    height: int,
    min_confidence: float,
) -> list[Keypoint] | None:
    if xy is None or xy.ndim != 3 or person_index >= len(xy):
        return None
    points: list[Keypoint] = []
    for index, point in enumerate(xy[person_index]):
        confidence = None
        if confidences is not None and confidences.ndim == 2 and person_index < len(confidences) and index < confidences.shape[1]:
            confidence = finite_or_none(confidences[person_index, index])
            if confidence is not None and not 0 <= confidence <= 1:
                confidence = None
        x = finite_or_none(point[0]) if len(point) >= 2 else None
        y = finite_or_none(point[1]) if len(point) >= 2 else None
        # Ultralytics masks low-confidence coordinates to zero. Expose them as unavailable.
        if x is None or y is None or (confidence is not None and confidence < min_confidence):
            x = y = None
        else:
            x, y = clamp01(x / width), clamp01(y / height)
        points.append(Keypoint(
            index=index,
            name=COCO_KEYPOINTS[index] if index < len(COCO_KEYPOINTS) else f"keypoint_{index}",
            x=x, y=y, confidence=confidence,
        ))
    return points


def people_from_result(result: Any, width: int, height: int, keypoint_threshold: float = 0.25) -> list[PersonObservation]:
    """Return current-frame detections; missing IDs/keypoints remain null."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _array(getattr(boxes, "xyxy", None))
    if xyxy is None or xyxy.ndim != 2:
        return []
    confidence_values = _array(getattr(boxes, "conf", None))
    ids = _array(getattr(boxes, "id", None))
    classes = _array(getattr(boxes, "cls", None))
    pose = getattr(result, "keypoints", None)
    point_xy = _array(getattr(pose, "xy", None)) if pose is not None else None
    point_conf = _array(getattr(pose, "conf", None)) if pose is not None else None

    people = []
    for index, raw_box in enumerate(xyxy):
        if classes is not None and index < len(classes):
            class_id = finite_or_none(classes[index])
            if class_id is None or int(class_id) != 0:
                continue
        box = normalized_box(raw_box, width, height)
        if box is None:
            continue
        confidence = finite_or_none(confidence_values[index]) if confidence_values is not None and index < len(confidence_values) else None
        if confidence is not None and not 0 <= confidence <= 1:
            confidence = None
        raw_id = finite_or_none(ids[index]) if ids is not None and index < len(ids) else None
        track_id = int(raw_id) if raw_id is not None and raw_id >= 0 and raw_id.is_integer() else None
        people.append(PersonObservation(
            track_id=track_id,
            bbox=box,
            confidence=confidence,
            keypoints=_keypoints_for_person(point_xy, point_conf, index, width, height, keypoint_threshold),
        ))
    return people


class PoseTracker:
    def __init__(self, model_path: str | Path, imgsz: int, confidence: float, device: str, keypoint_threshold: float):
        self.model_path = str(model_path)
        self.imgsz = imgsz
        self.confidence = confidence
        self.device = device
        self.keypoint_threshold = keypoint_threshold
        self.reset()

    def reset(self) -> None:
        """Start a fresh Ultralytics predictor/tracker for a new source session."""
        with redirect_stdout(sys.stderr):
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        if self.model.task != "pose":
            raise ValueError(f"model must be a pose model, got {self.model.task!r}")

    def process(self, frame: np.ndarray) -> list[PersonObservation]:
        height, width = frame.shape[:2]
        with redirect_stdout(sys.stderr):
            result = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                imgsz=self.imgsz,
                conf=self.confidence,
                device=self.device,
                verbose=False,
            )[0]
        return people_from_result(result, width, height, self.keypoint_threshold)
