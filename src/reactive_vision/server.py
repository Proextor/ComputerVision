"""FastAPI HTTP and WebSocket transport for the shared vision pipeline."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import logging
import os
import secrets
from pathlib import Path
from math import isfinite
from urllib.parse import urlsplit

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import cv2

from .main import arguments as vision_arguments
from .runtime import VisionRuntime
from .browser_camera import BrowserCameraRuntime, FrameRateLimit, decode_frame, MAX_FRAME_BYTES, MAX_FPS
from .schemas import SceneSnapshot
from .service_schemas import (
    ApiErrorResponse, HealthResponse, HeartbeatEnvelope, ReadinessResponse,
    ServiceStatus, SnapshotEnvelope,
)


LOG = logging.getLogger("reactive_vision.server")


def _service_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ingest-token", default=os.environ.get("VISION_INGEST_TOKEN"), help="browser ingest secret (or VISION_INGEST_TOKEN)")
    parser.add_argument("--ssl-certfile", type=Path, help="user-supplied trusted TLS certificate")
    parser.add_argument("--ssl-keyfile", type=Path, help="user-supplied TLS private key")
    parser.add_argument("--serve-display", type=Path, help="explicit built display directory")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--freshness-timeout", dest="freshness_timeout_seconds", type=float,
                        default=3.0, help="seconds without a fresh snapshot before stale")
    parser.add_argument("--heartbeat-interval", dest="heartbeat_seconds", type=float,
                        default=1.0, help="seconds between idle WebSocket heartbeats")
    parser.add_argument("--shutdown-timeout", dest="shutdown_timeout_seconds", type=float,
                        default=2.0, help="maximum seconds to wait for worker shutdown")
    parser.add_argument("--webcam-retry", dest="webcam_retry_seconds", type=float,
                        default=2.0, help="seconds between webcam reopen attempts")
    parser.add_argument("--loop-video", action="store_true", help="restart video with a new session at EOF")
    parser.add_argument("--cors-origin", action="append", default=[],
                        help="allowed browser origin; repeat for each origin")
    parser.add_argument("--strict-origin", action="store_true",
                        help="reject WebSocket clients without an Origin header")
    parser.add_argument("--log-level", choices=["critical", "error", "warning", "info", "debug"],
                        default="info")
    parser.add_argument("--debug-jsonl", action="store_true",
                        help="also write published scenes as JSONL on stdout")


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    args = vision_arguments(argv, extend=_service_options)
    if args.browser_camera and (not args.ingest_token or not 16 <= len(args.ingest_token) <= 256):
        raise ValueError("browser camera requires an ingest token of 16–256 characters")
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        raise ValueError("TLS requires both --ssl-certfile and --ssl-keyfile")
    for path in (args.ssl_certfile, args.ssl_keyfile):
        if path is not None and not path.is_file():
            raise ValueError("TLS file does not exist")
    if args.serve_display is not None and not (args.serve_display / "index.html").is_file():
        raise ValueError("display directory must contain a built index.html")
    for name in ("freshness_timeout_seconds", "heartbeat_seconds",
                 "shutdown_timeout_seconds", "webcam_retry_seconds"):
        value = getattr(args, name)
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    for origin in args.cors_origin:
        parsed = urlsplit(origin)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or
                parsed.username or parsed.password or parsed.path != "" or
                parsed.query or parsed.fragment or
                origin != f"{parsed.scheme}://{parsed.netloc}"):
            raise ValueError(f"invalid CORS origin: {origin}")
    args.headless = True
    return args


async def _watch_disconnect(websocket: WebSocket) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        # M5 has no control commands. Ignore text, binary and malformed messages.


def create_app(runtime: VisionRuntime, *, origins: list[str] | None = None,
               strict_origin: bool = False, heartbeat_seconds: float = 1.0,
               manage_runtime: bool = True, serve_display: Path | None = None,
               ingest_token: str | None = None) -> FastAPI:
    allowed_origins = set(origins or [])

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if manage_runtime:
            runtime.start()
        try:
            yield
        finally:
            if manage_runtime:
                runtime.stop()

    app = FastAPI(title="Reactive Vision Live Data", version="0.5.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(allowed_origins),
                       allow_credentials=False, allow_methods=["GET"], allow_headers=[])

    def origin_allowed(websocket: WebSocket) -> bool:
        origin = websocket.headers.get("origin")
        same_origin = f"{'https' if websocket.url.scheme == 'wss' else 'http'}://{websocket.url.netloc}"
        return ((origin is None and not strict_origin) or origin in allowed_origins or
                (serve_display is not None and origin == same_origin))

    @app.websocket("/api/v1/camera/ingest")
    async def ingest(websocket: WebSocket) -> None:
        if not isinstance(runtime, BrowserCameraRuntime) or not ingest_token or not origin_allowed(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        source = None
        try:
            # First message only: no secret in URL, logs, scene snapshots or errors.
            message = await asyncio.wait_for(websocket.receive(), timeout=5)
            import json
            raw = message.get("text")
            if raw is None or len(raw) > 1024:
                await websocket.close(code=1008)
                return
            try:
                auth = json.loads(raw)
            except ValueError:
                auth = None
            if (not isinstance(auth, dict) or auth.get("type") != "auth" or
                    not isinstance(auth.get("token"), str) or
                    not secrets.compare_digest(auth["token"].encode(), ingest_token.encode())):
                await websocket.close(code=1008)
                return
            source = runtime.connect()
            if source is None:
                await websocket.close(code=1013, reason="producer_busy")
                return
            await websocket.send_json({"type": "ready", "session_id": source.session_id,
                                       "max_frame_bytes": MAX_FRAME_BYTES, "max_fps": MAX_FPS})
            limiter = FrameRateLimit()
            while not source.closed:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                payload = message.get("bytes")
                if payload is None or len(payload) > MAX_FRAME_BYTES:
                    await websocket.close(code=1009 if payload else 1008)
                    break
                if not limiter.accept():
                    await websocket.close(code=1008, reason="frame_rate")
                    break
                try:
                    frame = await asyncio.to_thread(decode_frame, payload)
                except (ValueError, cv2.error):
                    await websocket.close(code=1003, reason="invalid_frame")
                    break
                source.offer(frame)
        except asyncio.TimeoutError:
            await websocket.close(code=1008)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if source is not None:
                runtime.disconnect(source)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return runtime.health()

    @app.get("/ready", response_model=ReadinessResponse,
             responses={503: {"model": ReadinessResponse}})
    def ready():
        state = runtime.health()
        available = state.service_status == ServiceStatus.READY
        body = ReadinessResponse(
            ready=available, service_status=state.service_status,
            detail="fresh scene available" if available else f"service is {state.service_status.value}",
            publication_sequence=state.publication_sequence,
            last_snapshot_age_seconds=state.last_snapshot_age_seconds,
        )
        if available:
            return body
        return JSONResponse(status_code=503, content=body.model_dump(mode="json"))

    @app.get("/api/v1/scene/latest", response_model=SceneSnapshot,
             responses={503: {"model": ApiErrorResponse}})
    def latest():
        publication = runtime.broker.latest()
        if publication is None:
            error = ApiErrorResponse(detail="no scene snapshot has been published",
                                     service_status=runtime.health().service_status)
            return JSONResponse(status_code=503, content=error.model_dump(mode="json"))
        return publication.snapshot

    @app.websocket("/api/v1/scene/stream")
    async def stream(websocket: WebSocket) -> None:
        if not origin_allowed(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        subscriber, initial = runtime.broker.subscribe(asyncio.get_running_loop())
        disconnect_task = asyncio.create_task(_watch_disconnect(websocket))
        last_sequence = 0
        try:
            if initial is not None:
                envelope = SnapshotEnvelope(
                    publication_sequence=initial.sequence, published_at=initial.published_at,
                    snapshot=initial.snapshot,
                )
                await websocket.send_json(envelope.model_dump(mode="json"))
                last_sequence = initial.sequence
            while True:
                queue_task = asyncio.create_task(subscriber.queue.get())
                try:
                    done, _ = await asyncio.wait(
                        {queue_task, disconnect_task}, timeout=heartbeat_seconds,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if disconnect_task in done:
                        break
                    if queue_task in done:
                        publication = queue_task.result()
                        if publication.sequence <= last_sequence:
                            continue
                        envelope = SnapshotEnvelope(
                            publication_sequence=publication.sequence,
                            published_at=publication.published_at,
                            snapshot=publication.snapshot,
                        )
                        await websocket.send_json(envelope.model_dump(mode="json"))
                        last_sequence = publication.sequence
                    else:
                        health = runtime.health()
                        heartbeat = HeartbeatEnvelope(
                            publication_sequence=health.publication_sequence,
                            service_status=health.service_status,
                            last_snapshot_age_seconds=health.last_snapshot_age_seconds,
                        )
                        await websocket.send_json(heartbeat.model_dump(mode="json"))
                finally:
                    if not queue_task.done():
                        queue_task.cancel()
                        try:
                            await queue_task
                        except asyncio.CancelledError:
                            pass
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
        finally:
            runtime.broker.unsubscribe(subscriber)
            disconnect_task.cancel()
            try:
                await disconnect_task
            except (asyncio.CancelledError, RuntimeError, WebSocketDisconnect):
                pass

    if serve_display is not None:
        root = serve_display.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str):
            if path == "api" or path.startswith("api/") or path in ("health", "ready"):
                raise HTTPException(status_code=404)
            candidate = (root / path).resolve()
            if not candidate.is_relative_to(root):
                raise HTTPException(status_code=404)
            if candidate.is_file():
                return FileResponse(candidate)
            if Path(path).suffix or path.startswith("assets/"):
                raise HTTPException(status_code=404)
            return FileResponse(root / "index.html")

    return app


def main(argv: list[str] | None = None) -> int:
    try:
        args = arguments(argv)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    # WebSocket protocol debug logs can contain authentication and image payloads.
    if args.browser_camera and args.log_level == "debug":
        args.log_level = "info"
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(levelname)s %(name)s: %(message)s")
    if args.host == "0.0.0.0":
        LOG.warning("binding on all interfaces; scene read endpoints have no authentication")
    runtime = BrowserCameraRuntime(args) if args.browser_camera else VisionRuntime(args)
    app = create_app(runtime, origins=args.cors_origin,
                     strict_origin=args.strict_origin,
                     heartbeat_seconds=args.heartbeat_seconds,
                     serve_display=args.serve_display, ingest_token=args.ingest_token)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level,
                workers=1, reload=False, ws_max_size=MAX_FRAME_BYTES, ws_max_queue=1,
                ws_per_message_deflate=False,
                ssl_certfile=str(args.ssl_certfile) if args.ssl_certfile else None,
                ssl_keyfile=str(args.ssl_keyfile) if args.ssl_keyfile else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
