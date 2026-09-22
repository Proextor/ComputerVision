"""Deterministic replay through the exact live M2–M4 processing boundary."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from statistics import mean
import time
import tracemalloc

from reactive_vision.behaviour import BehaviourConfig
from reactive_vision.pipeline import TrackedFrameProcessor
from reactive_vision.pose_behaviour import PoseConfig, load_pose_config
from reactive_vision.room_state import RoomConfig, load_room_config
from reactive_vision.schemas import PersonObservation, SceneSnapshot
from reactive_vision.zones import ZoneConfig, load_zones

from .schemas import TraceFrame, TraceHeader
from .trace import percentile, validate_frames

EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def load_behaviour_config(path: Path | None) -> BehaviourConfig:
    if path is None:
        return BehaviourConfig()
    try:
        return BehaviourConfig(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"invalid behaviour configuration {path.name}: {error}") from error


@dataclass
class ReplayResult:
    scenes: list[SceneSnapshot]
    frame_seconds: list[float]
    timings: list[float]
    wall_seconds: float
    peak_replay_bytes: int | None

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.frame_seconds[-1] - self.frame_seconds[0]) if self.frame_seconds else 0.0

    def runtime(self) -> dict:
        return {
            "video_duration_seconds": self.duration_seconds,
            "frames_processed": len(self.scenes),
            "mean_replay_seconds": mean(self.timings) if self.timings else None,
            "p50_replay_seconds": percentile(self.timings, 50),
            "p95_replay_seconds": percentile(self.timings, 95),
            "effective_replay_fps": len(self.scenes) / self.wall_seconds if self.wall_seconds > 0 else None,
            "replay_real_time_factor": self.duration_seconds / self.wall_seconds if self.wall_seconds > 0 else None,
            "peak_replay_bytes": self.peak_replay_bytes,
        }


def replay(header: TraceHeader, frames: list[TraceFrame], *, session_id: str,
           zones: ZoneConfig | None = None, behaviour: BehaviourConfig | None = None,
           pose: PoseConfig | None = None, room: RoomConfig | None = None,
           real_time: bool = False, sleep=time.sleep) -> ReplayResult:
    if not frames:
        raise ValueError("cannot replay an empty trace")
    validate_frames(header, frames)
    processor = TrackedFrameProcessor(
        session_id, zones or ZoneConfig(zones=[]), behaviour or BehaviourConfig(),
        pose or PoseConfig(), room or RoomConfig(),
    )
    scenes: list[SceneSnapshot] = []
    timings: list[float] = []
    started = time.perf_counter()
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    previous_time = frames[0].time_seconds
    try:
        for frame in frames:
            if real_time:
                sleep(max(0.0, frame.time_seconds - previous_time))
            previous_time = frame.time_seconds
            observed_at = EPOCH + timedelta(seconds=frame.time_seconds)
            people = [PersonObservation.model_validate(person.model_dump()) for person in frame.people]
            beginning = time.perf_counter()
            scene = processor.process(
                frame_id=frame.frame_id, source=header.source_id,
                status=frame.camera_status, observed_at=observed_at,
                people=people, width=frame.frame_width, height=frame.frame_height,
                source_frame_index=frame.source_frame_index,
                source_time_seconds=frame.source_time_seconds,
                drain_events=True,
            )
            timings.append(time.perf_counter() - beginning)
            scenes.append(scene)
        peak = tracemalloc.get_traced_memory()[1] if not was_tracing else None
    finally:
        if not was_tracing:
            tracemalloc.stop()
    return ReplayResult(scenes=scenes, frame_seconds=[f.time_seconds for f in frames],
                        timings=timings, wall_seconds=time.perf_counter() - started,
                        peak_replay_bytes=peak)


def configs(zones: Path | None = None, pose: Path | None = None,
            room: Path | None = None, behaviour: Path | None = None):
    return (load_zones(zones), load_behaviour_config(behaviour),
            load_pose_config(pose), load_room_config(room))


def with_overrides(configuration: tuple, overrides: dict | None) -> tuple:
    zone_config, behaviour_config, pose_config, room_config = configuration
    for component, values in (overrides or {}).items():
        if component == "behaviour":
            behaviour_config = BehaviourConfig(**{**vars(behaviour_config), **values})
        elif component == "pose":
            pose_config = PoseConfig.model_validate({**pose_config.model_dump(), **values})
        elif component == "room":
            room_config = RoomConfig.model_validate({**room_config.model_dump(), **values})
        else:
            raise ValueError(f"unsupported sweep component: {component}")
    return zone_config, behaviour_config, pose_config, room_config
