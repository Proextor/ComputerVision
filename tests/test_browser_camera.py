"""Browser ingestion tests use synthetic JPEGs and the real shared analysis boundary."""
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from reactive_vision.browser_camera import (BrowserCameraRuntime, BrowserFrameSource, FrameRateLimit,
                                           MAX_FRAME_BYTES, decode_frame)
from reactive_vision.server import arguments, create_app

TOKEN = "synthetic-test-secret-123"

class Tracker:
    instances = 0
    def __init__(self, *_):
        type(self).instances += 1
    def process(self, frame):
        assert frame.shape == (24, 32, 3)
        return []

def jpeg():
    return cv2.imencode('.jpg', np.zeros((24, 32, 3), np.uint8))[1].tobytes()

def setup():
    runtime = BrowserCameraRuntime(arguments(['--browser-camera', '--ingest-token', TOKEN]), tracker_factory=Tracker)
    return runtime, create_app(runtime, ingest_token=TOKEN)

def auth(ws):
    ws.send_json({'type': 'auth', 'token': TOKEN})
    return ws.receive_json()

@pytest.mark.parametrize('payload', [b'', b'bad', b'x' * (MAX_FRAME_BYTES + 1)], ids=['empty', 'invalid', 'oversize'])
def test_frame_validation(payload):
    with pytest.raises(ValueError): decode_frame(payload)

def test_jpeg_dimensions_before_decode(monkeypatch):
    raw = bytearray(jpeg())
    pos = raw.index(b'\xff\xc0')
    raw[pos + 5:pos + 7] = (65000).to_bytes(2, 'big')
    monkeypatch.setattr(cv2, 'imdecode', lambda *_: pytest.fail('must reject before decoding'))
    with pytest.raises(ValueError, match='frame_dimensions'): decode_frame(bytes(raw))

def test_latest_only_and_rate_limit():
    source = BrowserFrameSource()
    for n in range(100): source.offer(n)
    assert source.read().frame == 99
    assert source.pending is None
    source.close()
    assert source.read().status.value == 'unavailable'
    clock = [0.0]
    limiter = FrameRateLimit(lambda: clock[0])
    assert all(limiter.accept() for _ in range(12))
    assert not limiter.accept()
    clock[0] = 1.01
    assert limiter.accept()

def test_unauthorized_second_producer_and_bad_frame():
    runtime, app = setup()
    with TestClient(app) as client:
        with client.websocket_connect('/api/v1/camera/ingest') as ws:
            ws.send_json({'type': 'auth', 'token': 'wrong'})
            with pytest.raises(WebSocketDisconnect) as exc: ws.receive_json()
            assert exc.value.code == 1008
        with client.websocket_connect('/api/v1/camera/ingest') as first:
            assert auth(first)['type'] == 'ready'
            with client.websocket_connect('/api/v1/camera/ingest') as second:
                second.send_json({'type': 'auth', 'token': TOKEN})
                with pytest.raises(WebSocketDisconnect) as exc: second.receive_json()
                assert exc.value.code == 1013
            first.send_bytes(b'not an image')
            with pytest.raises(WebSocketDisconnect) as exc: first.receive_json()
            assert exc.value.code == 1003
        assert runtime.health().service_status.value == 'unavailable'

@pytest.mark.parametrize('payload,code', [(b'x' * (MAX_FRAME_BYTES + 1), 1009), (b'broken', 1003)], ids=['oversize', 'malformed'])
def test_ingest_frame_limits(payload, code):
    _, app = setup()
    with TestClient(app) as client, client.websocket_connect('/api/v1/camera/ingest') as ws:
        auth(ws); ws.send_bytes(payload)
        with pytest.raises(WebSocketDisconnect) as exc: ws.receive_json()
        assert exc.value.code == code

def test_pipeline_reconnect_staleness_and_no_persistence(monkeypatch):
    runtime, app = setup()
    monkeypatch.setattr(cv2, 'imwrite', lambda *_: pytest.fail('no frame writes'))
    with TestClient(app) as client:
        sessions = []
        for _ in range(2):
            with client.websocket_connect('/api/v1/camera/ingest') as ws:
                sessions.append(auth(ws)['session_id'])
                ws.send_bytes(jpeg())
                deadline = time.monotonic() + 2
                while runtime.health().service_status.value != 'ready' and time.monotonic() < deadline:
                    time.sleep(.01)
                data = client.get('/api/v1/scene/latest').json()
                assert data['schema_version'] == '4.0'
                assert data['camera_status'] == 'ok'
                assert data['visible_count'] == 0
                assert data['frame_width'] == 32
                assert data['session_id'] == sessions[-1]
                runtime.source.last_received -= 10
                assert runtime.health().service_status.value == 'stale'
            deadline = time.monotonic() + 1
            while runtime.source is not None and time.monotonic() < deadline: time.sleep(.01)
            assert runtime.health().service_status.value == 'unavailable'
        assert sessions[0] != sessions[1]

def test_rate_enforced_on_socket():
    _, app = setup()
    with TestClient(app) as client, client.websocket_connect('/api/v1/camera/ingest') as ws:
        auth(ws)
        for _ in range(13): ws.send_bytes(jpeg())
        with pytest.raises(WebSocketDisconnect) as exc: ws.receive_json()
        assert exc.value.code == 1008

def test_static_routing_and_same_origin(tmp_path):
    (tmp_path / 'index.html').write_text('<html>test display</html>')
    (tmp_path / 'app.js').write_text('ok')
    runtime, _ = setup()
    app = create_app(runtime, ingest_token=TOKEN, serve_display=tmp_path)
    with TestClient(app) as client:
        assert 'test display' in client.get('/').text
        assert 'test display' in client.get('/review').text
        assert client.get('/app.js').text == 'ok'
        assert client.get('/api/missing').status_code == 404
        assert client.get('/health').json()['source'] == 'browser-camera'
        assert client.get('/ready').status_code == 503
        assert client.get('/assets/missing.js').status_code == 404
        with client.websocket_connect('/api/v1/camera/ingest', headers={'Origin':'http://testserver'}) as ws:
            assert auth(ws)['type'] == 'ready'

def test_source_options():
    with pytest.raises(SystemExit): arguments(['--browser-camera', '--webcam', '0'])
    with pytest.raises(ValueError): arguments(['--browser-camera'])
    with pytest.raises(ValueError): arguments(['--ssl-certfile', 'missing'])
