"""Inspect and extract no-pixel tracked-observation traces."""

from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from statistics import mean
import tempfile
from time import perf_counter
from typing import Iterator

from pydantic import ValidationError

from reactive_vision.capture import FrameSource
from reactive_vision.schemas import CameraStatus
from reactive_vision.tracking import PoseTracker

from .schemas import TraceFrame, TraceHeader, TracePerson


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    low = int(position)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)


def write_trace(path: Path, header: TraceHeader, frames: list[TraceFrame]) -> None:
    validate_frames(header, frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header.model_dump_json() + "\n")
        for frame in frames:
            handle.write(frame.model_dump_json() + "\n")


def validate_frames(header: TraceHeader, frames: list[TraceFrame]) -> None:
    if not frames:
        raise ValueError("trace contains no frames")
    last_id, last_time = -1, -1.0
    terminal = False
    for frame in frames:
        if frame.source_id != header.source_id:
            raise ValueError("trace frame source differs from metadata")
        if frame.frame_id <= last_id or frame.time_seconds < last_time:
            raise ValueError("trace frame IDs must increase and timestamps must not decrease")
        if terminal:
            raise ValueError("trace has frames after a terminal source status")
        last_id, last_time = frame.frame_id, frame.time_seconds
        terminal = frame.camera_status != CameraStatus.OK


def load_trace(path: Path) -> tuple[TraceHeader, list[TraceFrame]]:
    try:
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
            if not first:
                raise ValueError("trace is empty")
            header = TraceHeader.model_validate_json(first)
            frames = [TraceFrame.model_validate_json(line) for line in handle if line.strip()]
        validate_frames(header, frames)
        return header, frames
    except (OSError, ValidationError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid or incompatible trace {path.name}: {error}") from error


def extract(video: Path, trace: Path, *, model: str = "yolo26n-pose.pt",
            imgsz: int = 640, confidence: float = 0.25,
            joint_confidence: float = 0.25, device: str = "cpu",
            source_id: str | None = None,
            source_factory=FrameSource, tracker_factory=PoseTracker) -> TraceHeader:
    if not video.is_file():
        raise ValueError(f"video not found: {video}")
    if not 0 <= confidence <= 1 or not 0 <= joint_confidence <= 1 or imgsz <= 0:
        raise ValueError("invalid detector settings")
    source_id = source_id or video.stem
    weights = Path(model)
    header = TraceHeader(
        source_id=source_id, model_name=weights.name,
        model_sha256=hash_file(weights) if weights.is_file() else None,
        video_sha256=hash_file(video),
        dependencies={name: version(name) for name in ("opencv-python", "ultralytics", "pydantic", "numpy")},
        detector_settings={"imgsz": imgsz, "confidence": confidence,
                           "joint_confidence": joint_confidence, "device": device,
                           "tracker": "bytetrack.yaml", "persist": True},
    )
    trace.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    durations: list[float] = []
    count = 0
    last_time = 0.0
    last_index: int | None = None
    last_dimensions: tuple[int | None, int | None] = (None, None)
    started = perf_counter()
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=trace.parent,
                                         suffix=".frames.tmp", delete=False) as spool:
            temporary = Path(spool.name)
            with source_factory(0, video, 1280, 720) as source:
                tracker = None
                while True:
                    item = source.read()
                    if item.status == CameraStatus.OK:
                        if item.source_time_seconds is None:
                            raise ValueError("video has no source timestamp or FPS; cannot create deterministic trace")
                        if item.source_time_seconds < last_time:
                            raise ValueError("video timestamps are out of order")
                        assert item.frame is not None
                        height, width = item.frame.shape[:2]
                        last_dimensions = width, height
                        if tracker is None:
                            tracker = tracker_factory(model, imgsz, confidence, device, joint_confidence)
                        beginning = perf_counter()
                        observed = tracker.process(item.frame)
                        durations.append(perf_counter() - beginning)
                        people = [TracePerson.model_validate(person.model_dump(include={
                            "track_id", "bbox", "confidence", "keypoints",
                        })) for person in observed]
                        last_time = item.source_time_seconds
                        last_index = item.source_frame_index
                    else:
                        if not durations:
                            raise ValueError("video source produced no frames; extraction aborted")
                        width, height = last_dimensions
                        people = []
                    frame = TraceFrame(source_id=source_id, frame_id=count,
                                       time_seconds=last_time, frame_width=width,
                                       frame_height=height, camera_status=item.status,
                                       people=people, source_frame_index=item.source_frame_index if item.status == CameraStatus.OK else last_index,
                                       source_time_seconds=item.source_time_seconds if item.status == CameraStatus.OK else last_time)
                    spool.write(frame.model_dump_json() + "\n")
                    count += 1
                    if item.status != CameraStatus.OK:
                        break
        elapsed = perf_counter() - started
        if header.model_sha256 is None and weights.is_file():
            header.model_sha256 = hash_file(weights)
        header.extraction_runtime = {
            "frames_processed": len(durations), "wall_seconds": elapsed,
            "mean_model_seconds": mean(durations) if durations else None,
            "p95_model_seconds": percentile(durations, 95),
            "effective_fps": len(durations) / elapsed if elapsed > 0 else None,
            "real_time_factor": last_time / elapsed if elapsed > 0 else None,
        }
        # Write the header only after extraction succeeds; rename atomically.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=trace.parent,
                                         suffix=".trace.tmp", delete=False) as output:
            finished = Path(output.name)
            output.write(header.model_dump_json() + "\n")
            with temporary.open(encoding="utf-8") as source:
                for line in source:
                    output.write(line)
        load_trace(finished)
        os.replace(finished, trace)
        return header
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if "finished" in locals():
            finished.unlink(missing_ok=True)
