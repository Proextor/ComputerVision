"""M5 transport and worker tests without a camera, model, GPU or video file."""

import asyncio
import threading
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

from reactive_vision.broker import SnapshotBroker
from reactive_vision.capture import CaptureResult
from reactive_vision.pipeline import snapshot
from reactive_vision.runtime import VisionRuntime
from reactive_vision.schemas import BoundingBox, CameraStatus, PersonObservation
from reactive_vision.server import arguments, create_app


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def scene(status=CameraStatus.OK, session="one", frame=0, people=None):
    return snapshot(session, frame, "webcam:0", status, 320, 240,
                    people=people or [])


def setup_service(*, freshness=3.0, heartbeat=0.1, origins=None, strict=False):
    clock = Clock()
    args = arguments(["--freshness-timeout", str(freshness),
                      "--heartbeat-interval", str(heartbeat)])
    broker = SnapshotBroker(clock=clock)
    runtime = VisionRuntime(args, broker=broker, clock=clock)
    app = create_app(runtime, origins=origins, strict_origin=strict,
                     heartbeat_seconds=heartbeat, manage_runtime=False)
    return clock, runtime, app


def receive_until(ws, predicate):
    for _ in range(20):
        message = ws.receive_json()
        if predicate(message):
            return message
    pytest.fail("expected WebSocket message was not received")


def test_http_starting_ready_stale_and_openapi():
    clock, runtime, app = setup_service()
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["service_status"] == "starting"
        assert health.json()["source"] == "webcam:0"
        assert client.get("/ready").status_code == 503
        unavailable = client.get("/api/v1/scene/latest")
        assert unavailable.status_code == 503
        assert unavailable.json()["code"] == "snapshot_not_available"
        with runtime._state_lock:
            runtime._started_at = clock()
        clock.now += 3.1
        assert client.get("/health").json()["service_status"] == "stale"
        runtime.broker.publish(scene())
        assert client.get("/health").json()["service_status"] == "ready"
        assert client.get("/ready").json()["ready"] is True
        latest = client.get("/api/v1/scene/latest")
        assert latest.status_code == 200
        assert latest.json()["schema_version"] == "4.0"
        assert latest.json()["room_state"]["occupancy"]["state"] == "unknown"
        clock.now += 3.1
        assert client.get("/health").json()["service_status"] == "stale"
        assert client.get("/ready").status_code == 503
        api = client.get("/openapi.json").json()
        assert "HealthResponse" in api["components"]["schemas"]
        assert "SceneSnapshot" in api["components"]["schemas"]
        assert api["paths"]["/api/v1/scene/latest"]["get"]["responses"]["503"]
        assert "access-control-allow-origin" not in client.get(
            "/health", headers={"Origin": "http://unlisted.example"}).headers


def test_unavailable_eof_and_error_health_do_not_rewrite_scene():
    _, runtime, app = setup_service()
    with TestClient(app) as client:
        runtime.broker.publish(scene(CameraStatus.UNAVAILABLE))
        assert client.get("/health").json()["service_status"] == "unavailable"
        assert client.get("/ready").status_code == 503
        value = client.get("/api/v1/scene/latest").json()
        assert value["camera_status"] == "unavailable"
        assert value["room_state"]["occupancy"]["state"] == "unknown"
        runtime.broker._clock.now += 4
        assert client.get("/health").json()["service_status"] == "stale"
        runtime.broker.publish(scene(CameraStatus.EOF, frame=1))
        assert client.get("/health").json()["service_status"] == "eof"
        assert client.get("/ready").status_code == 503
        with runtime._state_lock:
            runtime._worker_error = "RuntimeError in vision worker"
        result = client.get("/health").json()
        assert result["service_status"] == "error"
        assert result["last_worker_error"] == "RuntimeError in vision worker"


def test_websocket_latest_subsequent_heartbeat_reconnect_and_disconnect():
    clock, runtime, app = setup_service()
    runtime.broker.publish(scene(frame=0))
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/scene/stream") as ws:
            first = ws.receive_json()
            assert first["type"] == "snapshot"
            assert first["publication_sequence"] == 1
            assert first["snapshot"]["frame_id"] == 0
            assert runtime.broker.client_count == 1
            clock.now += 4
            heartbeat = receive_until(ws, lambda value: value["type"] == "heartbeat" and
                                       value["service_status"] == "stale")
            assert heartbeat["type"] == "heartbeat"
            assert heartbeat["service_status"] == "stale"
            runtime.broker.publish(scene(frame=1))
            second = receive_until(ws, lambda value: value["type"] == "snapshot" and
                                    value["publication_sequence"] == 2)
            assert second["type"] == "snapshot"
            assert second["publication_sequence"] == 2
            assert second["snapshot"]["frame_id"] == 1
            ws.send_text("{malformed client message")
        assert runtime.broker.client_count == 0
        with client.websocket_connect("/api/v1/scene/stream") as ws:
            current = ws.receive_json()
            assert current["publication_sequence"] == 2
            assert current["snapshot"]["frame_id"] == 1
        assert runtime.broker.client_count == 0


def test_websocket_multiple_clients_and_origins():
    _, runtime, app = setup_service(origins=["http://localhost:3000"])
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/scene/stream",
                                      headers={"Origin": "http://localhost:3000"}) as first:
            with client.websocket_connect("/api/v1/scene/stream") as second:
                assert runtime.broker.client_count == 2
                runtime.broker.publish(scene(frame=3))
                assert receive_until(first, lambda v: v["type"] == "snapshot")["snapshot"]["frame_id"] == 3
                assert receive_until(second, lambda v: v["type"] == "snapshot")["snapshot"]["frame_id"] == 3
        assert runtime.broker.client_count == 0
        with pytest.raises(Exception):
            with client.websocket_connect("/api/v1/scene/stream",
                                          headers={"Origin": "http://evil.example"}):
                pass
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    _, _, strict_app = setup_service(strict=True)
    with TestClient(strict_app) as client:
        with pytest.raises(Exception):
            with client.websocket_connect("/api/v1/scene/stream"):
                pass


def test_broker_sequence_latest_bounded_slow_clients_and_immutable_publication():
    async def scenario():
        broker = SnapshotBroker()
        first, _ = broker.subscribe(asyncio.get_running_loop())
        second, _ = broker.subscribe(asyncio.get_running_loop())
        for frame in range(100):
            publication = broker.publish(scene(frame=frame))
            assert publication.sequence == frame + 1
        received_one = await asyncio.wait_for(first.queue.get(), timeout=1)
        received_two = await asyncio.wait_for(second.queue.get(), timeout=1)
        assert received_one.sequence == received_two.sequence == 100
        assert received_one.snapshot.frame_id == 99
        assert first.queue.maxsize == second.queue.maxsize == 1
        assert first.queue.qsize() == second.queue.qsize() == 0
        copy = broker.latest().snapshot
        copy.frame_id = 1234
        assert broker.latest().snapshot.frame_id == 99
        broker.unsubscribe(first)
        broker.unsubscribe(second)
        assert broker.client_count == 0
    asyncio.run(scenario())


def test_worker_exception_and_bounded_shutdown_with_blocked_read():
    args = arguments(["--shutdown-timeout", "0.05", "--webcam-retry", "0.01"])
    began = threading.Event()
    unblock = threading.Event()
    released = threading.Event()

    class BlockedSource:
        label = "webcam:0"

        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            released.set()

        def read(self):
            began.set()
            unblock.wait()
            return CaptureResult(CameraStatus.UNAVAILABLE)

    runtime = VisionRuntime(args, source_factory=BlockedSource)
    runtime.start()
    assert began.wait(1)
    app = create_app(runtime, manage_runtime=False)
    with TestClient(app) as client:
        checked = time.monotonic()
        assert client.get("/health").json()["service_status"] == "starting"
        assert client.get("/ready").status_code == 503
        assert time.monotonic() - checked < 0.5
    started = time.monotonic()
    assert runtime.stop() is False
    assert time.monotonic() - started < 0.5
    assert released.is_set()
    assert runtime.health().service_status.value == "stopping"
    unblock.set()
    runtime._thread.join(timeout=1)
    assert not runtime._thread.is_alive()

    class FrameSource:
        label = "webcam:0"

        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return CaptureResult(CameraStatus.OK, np.zeros((32, 32, 3), dtype=np.uint8))

    class BrokenTracker:
        def __init__(self, *_):
            raise RuntimeError("private C:\\sensitive\\weights.pt")

    broken = VisionRuntime(args, source_factory=FrameSource, tracker_factory=BrokenTracker)
    broken._run()
    health = broken.health()
    assert health.service_status.value == "error"
    assert "ModelInitializationError" in health.last_worker_error
    assert "sensitive" not in health.last_worker_error
    assert health.worker_alive is False


def test_webcam_reconnect_resets_session_and_video_eof_policy(tmp_path):
    class TrackedBroker(SnapshotBroker):
        def __init__(self):
            super().__init__()
            self.all = []

        def publish(self, value):
            self.all.append(value)
            return super().publish(value)

    class Source:
        created = 0

        def __init__(self, *_):
            type(self).created += 1
            self.number = type(self).created
            self.label = "webcam:0"
            self.reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            self.reads += 1
            if self.number == 1:
                return CaptureResult(CameraStatus.UNAVAILABLE)
            if self.reads == 1:
                return CaptureResult(CameraStatus.OK, np.zeros((32, 32, 3), dtype=np.uint8))
            return CaptureResult(CameraStatus.EOF)

    class Tracker:
        def __init__(self, *_):
            pass

        def process(self, _frame):
            return [PersonObservation(track_id=1, bbox=BoundingBox(x1=0.1, y1=0.1,
                                                                      x2=0.5, y2=0.9))]

    class NoWait:
        def is_set(self):
            return False

        def wait(self, _seconds):
            return False

    args = arguments(["--webcam-retry", "0.01"])
    broker = TrackedBroker()
    runtime = VisionRuntime(args, broker=broker, source_factory=Source, tracker_factory=Tracker)
    runtime._stop = NoWait()
    runtime._run()
    assert Source.created == 2
    assert broker.all[0].camera_status == CameraStatus.UNAVAILABLE
    assert broker.all[1].session_id != broker.all[0].session_id
    assert broker.all[1].events[0].event_id == 1
    assert broker.all[-1].camera_status == CameraStatus.EOF

    video = tmp_path / "clip.mp4"
    video.touch()
    args = arguments(["--video", str(video)])

    class EofSource:
        created = 0

        def __init__(self, *_):
            type(self).created += 1
            self.label = str(video)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return CaptureResult(CameraStatus.EOF)

    no_loop = VisionRuntime(args, source_factory=EofSource)
    no_loop._run()
    assert EofSource.created == 1
    assert no_loop.health().service_status.value == "eof"
    args.loop_video = True
    loop_broker = TrackedBroker()
    looped = VisionRuntime(args, broker=loop_broker, source_factory=EofSource)

    class StopAfterTwo:
        calls = 0

        def is_set(self):
            return self.calls >= 2

        def wait(self, _seconds):
            self.calls += 1
            return self.is_set()

    looped._stop = StopAfterTwo()
    looped._run()
    assert EofSource.created == 3
    assert looped.broker.publication_sequence == 2
    assert loop_broker.all[0].session_id != loop_broker.all[1].session_id


def test_slow_model_loading_does_not_block_health():
    args = arguments([])
    loading = threading.Event()
    continue_loading = threading.Event()

    class Source:
        label = "webcam:0"
        reads = 0

        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            self.reads += 1
            if self.reads == 1:
                return CaptureResult(CameraStatus.OK, np.zeros((32, 32, 3), dtype=np.uint8))
            return CaptureResult(CameraStatus.EOF)

    class SlowTracker:
        def __init__(self, *_):
            loading.set()
            continue_loading.wait()

        def process(self, _frame):
            return []

    runtime = VisionRuntime(args, source_factory=Source, tracker_factory=SlowTracker)
    runtime.start()
    assert loading.wait(1)
    with TestClient(create_app(runtime, manage_runtime=False)) as client:
        checked = time.monotonic()
        assert client.get("/health").json()["service_status"] == "starting"
        assert time.monotonic() - checked < 0.5
    continue_loading.set()
    runtime._thread.join(timeout=1)
    assert runtime.health().service_status.value == "eof"


def test_health_video_source_is_basename_only(tmp_path):
    video = tmp_path / "private" / "clip.mp4"
    video.parent.mkdir()
    video.touch()
    runtime = VisionRuntime(arguments(["--video", str(video)]))
    assert runtime.health().source == "clip.mp4"
    assert str(video.parent) not in runtime.health().model_dump_json()


@pytest.mark.parametrize("argv", [
    ["--freshness-timeout", "0"], ["--heartbeat-interval", "nan"],
    ["--shutdown-timeout", "-1"], ["--webcam-retry", "inf"],
    ["--port", "70000"], ["--cors-origin", "*"],
    ["--cors-origin", "http://localhost:3000/path"],
    ["--cors-origin", "http://localhost:3000/"],
])
def test_invalid_service_options(argv):
    with pytest.raises(ValueError):
        arguments(argv)
