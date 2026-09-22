"""Dedicated blocking vision worker and responsive service health state."""

import logging
from pathlib import Path
import threading
import time
from typing import Callable

from .broker import SnapshotBroker
from .capture import FrameSource
from .pipeline import VisionSession
from .schemas import CameraStatus, SceneSnapshot
from .service_schemas import HealthResponse, ServiceStatus
from .tracking import PoseTracker


LOG = logging.getLogger("reactive_vision.runtime")


class VisionRuntime:
    def __init__(self, args, broker: SnapshotBroker | None = None,
                 session_factory=VisionSession, source_factory=FrameSource,
                 tracker_factory=PoseTracker,
                 clock: Callable[[], float] = time.monotonic):
        self.args = args
        self.broker = broker or SnapshotBroker(clock=clock)
        self._session_factory = session_factory
        self._source_factory = source_factory
        self._tracker_factory = tracker_factory
        self._clock = clock
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._session: VisionSession | None = None
        self._session_id: str | None = None
        self._worker_error: str | None = None
        self._stopping = False
        self._source_description = (Path(args.video).name if args.video is not None
                                    else f"webcam:{args.webcam}")

    def start(self) -> None:
        with self._state_lock:
            if self._thread is not None:
                return
            self._started_at = self._clock()
            self._thread = threading.Thread(target=self._run, name="vision-worker", daemon=True)
            self._thread.start()

    def _publish(self, session: VisionSession, scene: SceneSnapshot) -> None:
        completed = session.flush_events(scene)
        self.broker.publish(completed)
        if getattr(self.args, "debug_jsonl", False):
            print(completed.model_dump_json(exclude_none=False), flush=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            session = None
            terminal_status = None
            try:
                session = self._session_factory(
                    self.args, source_factory=self._source_factory,
                    tracker_factory=self._tracker_factory, wait=self._stop.wait,
                )
                with self._state_lock:
                    self._session = session
                    self._session_id = session.session_id
                last_emit = float("-inf")
                while not self._stop.is_set():
                    scene = session.step()
                    if scene.camera_status != CameraStatus.OK:
                        terminal_status = scene.camera_status
                        self._publish(session, scene)
                        break
                    now = self._clock()
                    if now - last_emit >= 1 / self.args.rate:
                        self._publish(session, scene)
                        last_emit = now
                if self._stop.is_set() and session.pending_event_count and session.last_scene is not None:
                    self._publish(session, session.last_scene)
            except Exception as error:
                with self._state_lock:
                    # No exception text or paths are returned over HTTP.
                    self._worker_error = f"{type(error).__name__} in vision worker"
                LOG.exception("vision worker failed")
                break
            finally:
                if session is not None:
                    try:
                        session.close()
                    except Exception:
                        with self._state_lock:
                            self._worker_error = "capture release failed"
                        LOG.exception("capture release failed")
                with self._state_lock:
                    if self._session is session:
                        self._session = None
            if self._stop.is_set():
                break
            if self.args.video is None and terminal_status in (CameraStatus.UNAVAILABLE, CameraStatus.READ_ERROR):
                LOG.warning("webcam unavailable; retrying in %.2f seconds", self.args.webcam_retry_seconds)
                if self._stop.wait(self.args.webcam_retry_seconds):
                    break
                continue
            if self.args.video is not None and terminal_status == CameraStatus.EOF and self.args.loop_video:
                if self._stop.wait(0.01):
                    break
                continue
            break

    def stop(self) -> bool:
        """Signal shutdown and bound all waiting, including a blocked native release."""
        with self._state_lock:
            self._stopping = True
            session = self._session
            worker = self._thread
        self._stop.set()
        timeout = self.args.shutdown_timeout_seconds
        deadline = self._clock() + timeout
        release_thread = None
        if session is not None:
            def release() -> None:
                try:
                    session.close()
                except Exception:
                    LOG.exception("capture release failed during shutdown")
            release_thread = threading.Thread(target=release, name="capture-release", daemon=True)
            release_thread.start()
            release_thread.join(timeout=min(timeout / 2, max(0.0, deadline - self._clock())))
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=max(0.0, deadline - self._clock()))
        exited = worker is None or not worker.is_alive()
        if not exited:
            LOG.warning("vision worker did not exit within %.2f seconds", timeout)
        if release_thread is not None and release_thread.is_alive():
            LOG.warning("capture release did not finish within shutdown timeout")
        return exited

    def health(self) -> HealthResponse:
        publication = self.broker.latest()
        age = self.broker.age_seconds()
        with self._state_lock:
            stopping = self._stopping
            error = self._worker_error
            session_id = self._session_id
            worker = self._thread
            started_at = self._started_at
        alive = worker is not None and worker.is_alive()
        scene = publication.snapshot if publication is not None else None
        if stopping:
            status = ServiceStatus.STOPPING
        elif error is not None:
            status = ServiceStatus.ERROR
        elif scene is None:
            status = (ServiceStatus.STALE if started_at is not None and
                      self._clock() - started_at > self.args.freshness_timeout_seconds
                      else ServiceStatus.STARTING)
        elif scene.camera_status == CameraStatus.EOF:
            status = ServiceStatus.EOF
        elif age is not None and age > self.args.freshness_timeout_seconds:
            status = ServiceStatus.STALE
        elif scene.camera_status != CameraStatus.OK:
            status = ServiceStatus.UNAVAILABLE
        else:
            status = ServiceStatus.READY
        return HealthResponse(
            service_status=status, worker_alive=alive,
            session_id=session_id or (scene.session_id if scene else None),
            camera_status=scene.camera_status if scene else None,
            latest_frame_id=scene.frame_id if scene else None,
            publication_sequence=self.broker.publication_sequence,
            last_snapshot_age_seconds=age,
            connected_websocket_clients=self.broker.client_count,
            source=self._source_description, last_worker_error=error,
        )
