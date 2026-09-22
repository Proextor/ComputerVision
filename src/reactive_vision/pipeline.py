"""One synchronous vision session shared by the CLI and service worker."""

from collections import deque
from datetime import datetime, timedelta, timezone
import threading
import time
from typing import Callable
from uuid import uuid4

from .behaviour import BehaviourAnalyzer
from .capture import FrameSource
from .room_state import RoomStateAnalyzer
from .schemas import CameraStatus, PersonObservation, RoomState, SceneEvent, SceneSnapshot
from .tracking import PoseTracker


class ModelInitializationError(RuntimeError):
    """Sanitized model-loading failure for service health reporting."""


class TrackedFrameProcessor:
    """Shared M2–M4 boundary for live tracking and deterministic trace replay."""

    def __init__(self, session_id: str, zone_config, behaviour_config,
                 pose_config, room_config):
        self.session_id = session_id
        self.analyzer = BehaviourAnalyzer(session_id, zone_config,
                                          behaviour_config, pose_config)
        self.room_analyzer = RoomStateAnalyzer(zone_config, room_config)
        self.empty_occupancy = {zone.id: 0 for zone in zone_config.zones}

    def process(self, *, frame_id: int, source: str, status: CameraStatus,
                observed_at: datetime, people: list[PersonObservation],
                width: int | None, height: int | None,
                source_frame_index: int | None = None,
                source_time_seconds: float | None = None,
                processing_fps: float | None = None,
                drain_events: bool = False) -> SceneSnapshot:
        if status == CameraStatus.OK:
            analyzed, occupancy = self.analyzer.update(people, observed_at)
        else:
            if people:
                raise ValueError("failed source cannot contain tracked people")
            analyzed, occupancy = [], self.empty_occupancy.copy()
        room = self.room_analyzer.update(analyzed, status, observed_at,
                                         occupancy, self.analyzer.emit_room_event)
        return snapshot(
            self.session_id, frame_id, source, status, width, height, analyzed,
            processing_fps, source_frame_index, source_time_seconds,
            timestamp_utc=observed_at, zone_occupancy=occupancy,
            room_state=room,
            events=self.analyzer.drain_events() if drain_events else None,
        )


def snapshot(
    session_id: str, frame_id: int, source: str, status: CameraStatus,
    width: int | None = None, height: int | None = None,
    people: list[PersonObservation] | None = None, processing_fps: float | None = None,
    source_frame_index: int | None = None, source_time_seconds: float | None = None,
    timestamp_utc: datetime | None = None,
    zone_occupancy: dict[str, int] | None = None,
    room_state: RoomState | None = None,
    events: list[SceneEvent] | None = None,
) -> SceneSnapshot:
    current_people = people if people is not None else []
    scene_time = timestamp_utc or datetime.now(timezone.utc)
    current_occupancy = zone_occupancy or {}
    if room_state is None:
        # Direct callers receive an explicit unknown baseline; a running session
        # always supplies the state from its persistent room analyzer.
        room_state = RoomStateAnalyzer().update(
            current_people, status, scene_time, current_occupancy, lambda *_: None,
        )
    return SceneSnapshot(
        session_id=session_id, frame_id=frame_id, timestamp_utc=scene_time,
        source=source, source_frame_index=source_frame_index,
        source_time_seconds=source_time_seconds, frame_width=width,
        frame_height=height, camera_status=status, processing_fps=processing_fps,
        visible_count=len(current_people), people=current_people,
        zone_occupancy=current_occupancy, room_state=room_state, events=events or [],
    )


class VisionSession:
    """Own one source, model and complete M1–M4 analysis state."""

    def __init__(self, args, source_factory=FrameSource, tracker_factory=PoseTracker,
                 wait: Callable[[float], object] = time.sleep, session_id: str | None = None):
        self.args = args
        self.session_id = session_id or str(uuid4())
        self.processor = TrackedFrameProcessor(
            self.session_id, args.zone_config, args.behaviour_config,
            args.pose_config, args.room_config,
        )
        self.analyzer = self.processor.analyzer
        self.room_analyzer = self.processor.room_analyzer
        self.source = source_factory(args.webcam, args.video, args.width, args.height)
        if hasattr(self.source, "__enter__"):
            self.source.__enter__()
        self._tracker_factory = tracker_factory
        self._tracker = None
        self._wait = wait
        self._close_lock = threading.Lock()
        self._closed = False
        self._terminal = False
        self.frame_id = 0
        self.latest_frame = None
        self.last_scene: SceneSnapshot | None = None
        self._durations: deque[float] = deque(maxlen=30)
        self._last_width = self._last_height = None
        self._last_source_index = self._last_source_time = None
        self._video_start = None
        self._video_epoch: datetime | None = None
        self._last_observed_at: datetime | None = None
        self._empty_occupancy = self.processor.empty_occupancy

    @property
    def pending_event_count(self) -> int:
        return self.analyzer.pending_event_count

    def flush_events(self, scene: SceneSnapshot) -> SceneSnapshot:
        return scene.model_copy(update={"events": self.analyzer.drain_events()})

    def step(self) -> SceneSnapshot:
        """Read and analyze exactly one frame; never queue frames."""
        if self._terminal:
            raise StopIteration("session reached terminal source state")
        current = self.source.read()
        if current.status != CameraStatus.OK:
            self._terminal = True
            now = datetime.now(timezone.utc)
            if self._last_observed_at is not None:
                now = max(now, self._last_observed_at)
            scene = self.processor.process(
                frame_id=self.frame_id, source=self.source.label, status=current.status,
                observed_at=now, people=[], width=self._last_width,
                height=self._last_height,
                source_frame_index=self._last_source_index,
                source_time_seconds=self._last_source_time,
            )
            self.last_scene = scene
            self.latest_frame = None
            return scene
        frame = current.frame
        assert frame is not None
        if self.args.video is not None and current.source_time_seconds is not None:
            if self._video_start is None:
                self._video_start = time.monotonic() - current.source_time_seconds
            delay = self._video_start + current.source_time_seconds - time.monotonic()
            if delay > 0:
                self._wait(delay)
        height, width = frame.shape[:2]
        started = time.perf_counter()
        if self._tracker is None:
            try:
                self._tracker = self._tracker_factory(
                    self.args.model, self.args.imgsz, self.args.confidence,
                    self.args.device, self.args.joint_confidence,
                )
            except Exception as error:
                raise ModelInitializationError("pose model initialization failed") from error
        raw_people = self._tracker.process(frame)
        observed_at = datetime.now(timezone.utc)
        if self.args.video is not None and current.source_time_seconds is not None:
            if self._video_epoch is None:
                self._video_epoch = observed_at - timedelta(seconds=current.source_time_seconds)
            source_timestamp = self._video_epoch + timedelta(seconds=current.source_time_seconds)
            observed_at = max(source_timestamp, self._last_observed_at) if self._last_observed_at else source_timestamp
        self._last_observed_at = observed_at
        scene = self.processor.process(
            frame_id=self.frame_id, source=self.source.label, status=CameraStatus.OK,
            observed_at=observed_at, people=raw_people, width=width, height=height,
            source_frame_index=current.source_frame_index,
            source_time_seconds=current.source_time_seconds,
        )
        elapsed = time.perf_counter() - started
        if elapsed > 0:
            self._durations.append(elapsed)
        fps = len(self._durations) / sum(self._durations) if self._durations else None
        scene.processing_fps = fps
        self.last_scene = scene
        self.latest_frame = frame
        self._last_width, self._last_height = width, height
        self._last_source_index = current.source_frame_index
        self._last_source_time = current.source_time_seconds
        self.frame_id += 1
        return scene

    def close(self) -> None:
        """Best-effort release, callable by shutdown even if read is blocked."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        if hasattr(self.source, "__exit__"):
            self.source.__exit__(None, None, None)
        else:
            self.source.close()
