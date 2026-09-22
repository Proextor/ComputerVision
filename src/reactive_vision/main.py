"""Command-line capture, tracking, preview and JSONL loop."""

import argparse
from collections.abc import Callable
import logging
from math import isfinite
from pathlib import Path
import sys
import time

import cv2

from .capture import FrameSource
from .behaviour import BehaviourConfig
from .pipeline import VisionSession, snapshot
from .pose_behaviour import load_pose_config
from .room_state import load_room_config
from .render import draw_preview
from .schemas import CameraStatus, SceneSnapshot
from .tracking import PoseTracker
from .zones import load_zones


LOG = logging.getLogger("reactive_vision")
WINDOW = "Reactive Vision M4 (Q / Esc to exit)"


def arguments(argv: list[str] | None = None,
              extend: Callable[[argparse.ArgumentParser], None] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose tracking and JSONL scene snapshots")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--webcam", type=int, default=None, help="webcam index (default: 0)")
    source.add_argument("--video", type=Path, help="local video file")
    if extend is not None:
        source.add_argument("--browser-camera", action="store_true", help="accept authenticated browser camera frames")
    parser.add_argument("--width", type=int, default=1280, help="requested webcam width")
    parser.add_argument("--height", type=int, default=720, help="requested webcam height")
    parser.add_argument("--imgsz", type=int, default=640, help="square inference size")
    parser.add_argument("--confidence", type=float, default=0.25, help="detection confidence threshold")
    parser.add_argument("--joint-confidence", type=float, default=0.25, help="minimum joint confidence for coordinates and overlay")
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="official model name or local pose weights path")
    parser.add_argument("--rate", type=float, default=5.0, help="maximum JSON snapshots per second")
    parser.add_argument("--zones", type=Path, help="JSON polygon zone configuration")
    parser.add_argument("--pose-config", type=Path, help="JSON pose and gesture settings")
    parser.add_argument("--room-config", type=Path, help="JSON room aggregation settings")
    parser.add_argument("--zone-debounce", type=float, default=0.35, help="seconds inside/outside before zone transition")
    parser.add_argument("--track-grace", type=float, default=0.75, help="seconds to retain an unseen track")
    parser.add_argument("--history-seconds", type=float, default=2.0, help="maximum track history duration")
    parser.add_argument("--motion-window", type=float, default=1.2, help="seconds used to classify motion")
    parser.add_argument("--min-observation", type=float, default=0.6, help="minimum seconds before motion classification")
    parser.add_argument("--motion-debounce", type=float, default=0.25, help="seconds before motion state transition")
    parser.add_argument("--movement-threshold", type=float, default=0.03, help="normalized anchor movement threshold")
    parser.add_argument("--scale-threshold", type=float, default=0.035, help="normalized box-height change threshold")
    parser.add_argument("--headless", action="store_true", help="disable preview window")
    if extend is not None:
        extend(parser)
    args = parser.parse_args(argv)
    if args.webcam is None:
        args.webcam = 0
    if (args.webcam < 0 or args.width <= 0 or args.height <= 0 or args.imgsz <= 0 or
            not isfinite(args.rate) or args.rate <= 0):
        parser.error("webcam index must be nonnegative; dimensions, imgsz and rate must be positive")
    if not 0 <= args.confidence <= 1 or not 0 <= args.joint_confidence <= 1:
        parser.error("confidence thresholds must be between 0 and 1")
    if args.video is not None and not args.video.is_file():
        parser.error(f"video file does not exist: {args.video}")
    if (args.model.endswith(".pt") or args.model.endswith(".onnx")) and ("/" in args.model or "\\" in args.model) and not Path(args.model).is_file():
        parser.error(f"model file does not exist: {args.model}")
    try:
        args.zone_config = load_zones(args.zones)
        args.pose_config = load_pose_config(args.pose_config)
        args.room_config = load_room_config(args.room_config)
        args.behaviour_config = BehaviourConfig(
            zone_debounce_seconds=args.zone_debounce,
            track_grace_seconds=args.track_grace,
            history_seconds=args.history_seconds,
            motion_window_seconds=args.motion_window,
            min_observation_seconds=args.min_observation,
            motion_debounce_seconds=args.motion_debounce,
            movement_threshold=args.movement_threshold,
            scale_threshold=args.scale_threshold,
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def emit(scene: SceneSnapshot) -> None:
    print(scene.model_dump_json(exclude_none=False), flush=True)


def run(args: argparse.Namespace) -> int:
    last_emit = float("-inf")
    preview_open = False
    session: VisionSession | None = None
    try:
        session = VisionSession(args, source_factory=FrameSource,
                                tracker_factory=PoseTracker, wait=time.sleep)
        while True:
            scene = session.step()
            if scene.camera_status != CameraStatus.OK:
                emit(session.flush_events(scene))
                if scene.camera_status != CameraStatus.EOF:
                    LOG.error("source %s: %s", scene.source, scene.camera_status.value)
                return 0 if scene.camera_status == CameraStatus.EOF else 2
            now = time.monotonic()
            if now - last_emit >= 1 / args.rate:
                emit(session.flush_events(scene))
                last_emit = now
            if not args.headless:
                frame = session.latest_frame
                assert frame is not None
                cv2.imshow(WINDOW, draw_preview(frame, scene, args.joint_confidence, args.zone_config.zones))
                preview_open = True
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    if session.pending_event_count:
                        emit(session.flush_events(scene))
                    return 0
    except KeyboardInterrupt:
        if session is not None and session.last_scene is not None and session.pending_event_count:
            emit(session.flush_events(session.last_scene))
        LOG.info("interrupted")
        return 0
    except Exception:
        if session is not None and session.last_scene is not None and session.pending_event_count:
            emit(session.flush_events(session.last_scene))
        LOG.exception("vision pipeline failed")
        return 1
    finally:
        if session is not None:
            session.close()
        if preview_open:
            cv2.destroyWindow(WINDOW)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    return run(arguments(argv))


if __name__ == "__main__":
    raise SystemExit(main())
