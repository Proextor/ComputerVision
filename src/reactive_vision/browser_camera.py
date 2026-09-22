"""Ephemeral browser frames, bounded capture and the unchanged M1–M4 session."""

from collections import deque
import threading
import time
from uuid import uuid4

import cv2
import numpy as np

from .broker import SnapshotBroker
from .capture import CaptureResult
from .pipeline import VisionSession, snapshot
from .schemas import CameraStatus
from .service_schemas import HealthResponse, ServiceStatus
from .tracking import PoseTracker

MAX_FRAME_BYTES = 512 * 1024
MAX_DIMENSION = 1280
MAX_PIXELS = 921600
MAX_FPS = 12


def decode_frame(payload: bytes) -> np.ndarray:
    """Accept JPEG only; check SOF dimensions before allocating decoded pixels."""
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise ValueError("frame_size")
    if payload[:2] != b"\xff\xd8":
        raise ValueError("invalid_jpeg")
    pos = 2
    dimensions = None
    while pos + 3 < len(payload):
        if payload[pos] != 0xff:
            raise ValueError("invalid_jpeg")
        while pos < len(payload) and payload[pos] == 0xff:
            pos += 1
        if pos >= len(payload):
            break
        marker = payload[pos]
        pos += 1
        if marker in (0xda, 0xd9):
            break
        if marker in (0x01, *range(0xd0, 0xd8)):
            continue
        length = int.from_bytes(payload[pos:pos + 2], "big")
        if length < 2 or pos + length > len(payload):
            raise ValueError("invalid_jpeg")
        if marker in (0xc0, 0xc1, 0xc2):
            if length < 8:
                raise ValueError("invalid_jpeg")
            h = int.from_bytes(payload[pos + 3:pos + 5], "big")
            w = int.from_bytes(payload[pos + 5:pos + 7], "big")
            dimensions = (h, w)
            if not (0 < h <= MAX_DIMENSION and 0 < w <= MAX_DIMENSION and h * w <= MAX_PIXELS):
                raise ValueError("frame_dimensions")
            break
        pos += length
    if dimensions is None:
        raise ValueError("invalid_jpeg")
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape[:2] != dimensions:
        raise ValueError("invalid_jpeg")
    return frame


class FrameRateLimit:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.arrivals = deque(maxlen=MAX_FPS)

    def accept(self) -> bool:
        now = self.clock()
        while self.arrivals and now - self.arrivals[0] >= 1:
            self.arrivals.popleft()
        if len(self.arrivals) >= MAX_FPS:
            return False
        self.arrivals.append(now)
        return True


class BrowserFrameSource:
    """Exactly one newest pending decoded frame; disconnect wakes blocking read."""
    label = "browser-camera"

    def __init__(self):
        self.session_id = str(uuid4())
        self.condition = threading.Condition()
        self.pending = None
        self.closed = False
        self.last_received = None

    def offer(self, frame) -> None:
        with self.condition:
            if not self.closed:
                self.pending = frame
                self.last_received = time.monotonic()
                self.condition.notify()

    def read(self):
        with self.condition:
            self.condition.wait_for(lambda: self.closed or self.pending is not None)
            if self.closed:
                return CaptureResult(CameraStatus.UNAVAILABLE)
            frame, self.pending = self.pending, None
            return CaptureResult(CameraStatus.OK, frame)

    def close(self):
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify_all()


class BrowserCameraRuntime:
    def __init__(self, args, tracker_factory=PoseTracker):
        self.args = args
        self.broker = SnapshotBroker()
        self.tracker_factory = tracker_factory
        self.condition = threading.Condition()
        self.source = None
        self.thread = None
        self.stopping = False
        self.error = None
        self.connected_at = None

    def start(self):
        with self.condition:
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True, name="browser-vision-worker")
                self.thread.start()

    def connect(self):
        with self.condition:
            if self.source is not None or self.stopping:
                return None
            self.source = BrowserFrameSource()
            self.error = None
            self.connected_at = time.monotonic()
            self.broker.publish(snapshot(self.source.session_id, 0, "browser-camera", CameraStatus.UNAVAILABLE))
            self.condition.notify_all()
            return self.source

    def disconnect(self, source):
        with self.condition:
            source.close()
            if self.source is source:
                self.source = None
                self.broker.publish(snapshot(source.session_id, 0, "browser-camera", CameraStatus.UNAVAILABLE))
                self.condition.notify_all()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopping or self.source is not None)
                if self.stopping:
                    return
                source = self.source
            session = None
            try:
                session = VisionSession(self.args, source_factory=lambda *_: source,
                                        tracker_factory=self.tracker_factory, session_id=source.session_id)
                last_emit = float("-inf")
                while not source.closed:
                    scene = session.step()
                    with self.condition:
                        # Never publish an old producer's in-flight inference after replacement.
                        if self.source is not source or source.closed or self.stopping:
                            break
                        now = time.monotonic()
                        if now - last_emit >= 1 / self.args.rate:
                            self.broker.publish(session.flush_events(scene))
                            last_emit = now
            except Exception:
                # Do not log payloads, tokens or exception strings from image/model libraries.
                with self.condition:
                    if self.source is source:
                        self.error = "browser vision processing failed"
                source.close()
            finally:
                if session is not None:
                    session.close()
            # A failed source keeps its producer slot until its socket is disconnected.
            with self.condition:
                self.condition.wait_for(lambda: self.stopping or self.source is not source)

    def stop(self):
        with self.condition:
            self.stopping = True
            if self.source is not None:
                self.source.close()
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=self.args.shutdown_timeout_seconds)
        return self.thread is None or not self.thread.is_alive()

    def health(self):
        publication = self.broker.latest()
        scene = publication.snapshot if publication else None
        age = self.broker.age_seconds()
        with self.condition:
            source, error = self.source, self.error
            if self.stopping:
                status = ServiceStatus.STOPPING
            elif error:
                status = ServiceStatus.ERROR
            elif source is None or source.closed:
                status = ServiceStatus.UNAVAILABLE
            elif time.monotonic() - (source.last_received or self.connected_at) > self.args.freshness_timeout_seconds:
                status = ServiceStatus.STALE
            elif scene is None or scene.camera_status != CameraStatus.OK:
                status = ServiceStatus.STARTING
            elif age is not None and age > self.args.freshness_timeout_seconds:
                status = ServiceStatus.STALE
            else:
                status = ServiceStatus.READY
            return HealthResponse(service_status=status, worker_alive=bool(self.thread and self.thread.is_alive()),
                                  session_id=source.session_id if source else (scene.session_id if scene else None),
                                  camera_status=scene.camera_status if scene else None,
                                  latest_frame_id=scene.frame_id if scene else None,
                                  publication_sequence=self.broker.publication_sequence,
                                  last_snapshot_age_seconds=age, connected_websocket_clients=self.broker.client_count,
                                  source="browser-camera", last_worker_error=error)
