"""Short-lived per-track spatial and temporal analysis, independent of YOLO."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from math import hypot, isfinite
from statistics import median

from .pose_behaviour import PoseBehaviourAnalyzer, PoseConfig, PoseTrackState
from .schemas import (
    EventType, MotionState, NormalizedPoint, PersonObservation,
    SceneEvent, TrajectoryPoint, require_utc,
)
from .zones import ZoneConfig, contains


@dataclass(frozen=True)
class BehaviourConfig:
    zone_debounce_seconds: float = 0.35
    track_grace_seconds: float = 0.75
    history_seconds: float = 2.0
    motion_window_seconds: float = 1.2
    min_observation_seconds: float = 0.6
    motion_debounce_seconds: float = 0.25
    movement_threshold: float = 0.03
    scale_threshold: float = 0.035
    smoothing_alpha: float = 0.35

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.history_seconds <= 0 or self.motion_window_seconds <= 0:
            raise ValueError("history and motion windows must be positive")
        if self.history_seconds < self.motion_window_seconds:
            raise ValueError("history window must be at least as long as motion window")
        if self.motion_window_seconds < self.min_observation_seconds:
            raise ValueError("motion window must be at least as long as minimum observation")
        if self.movement_threshold <= 0 or self.scale_threshold <= 0:
            raise ValueError("movement and scale thresholds must be positive")
        if not 0 < self.smoothing_alpha <= 1:
            raise ValueError("smoothing alpha must be in (0, 1]")


@dataclass(frozen=True)
class _Sample:
    at: datetime
    anchor: NormalizedPoint
    scale: float
    confirmed_zones: tuple[str, ...]


@dataclass
class _ZoneState:
    confirmed: bool = False
    candidate: bool | None = None
    candidate_since: datetime | None = None
    entered_at: datetime | None = None


@dataclass
class _TrackState:
    last_seen: datetime
    history: deque[_Sample] = field(default_factory=lambda: deque(maxlen=64))
    raw_anchors: deque[NormalizedPoint] = field(default_factory=lambda: deque(maxlen=3))
    raw_scales: deque[float] = field(default_factory=lambda: deque(maxlen=3))
    zones: dict[str, _ZoneState] = field(default_factory=dict)
    motion: MotionState = MotionState.UNKNOWN
    motion_candidate: MotionState | None = None
    motion_candidate_since: datetime | None = None
    pose: PoseTrackState = field(default_factory=PoseTrackState)


class BehaviourAnalyzer:
    def __init__(self, session_id: str, zones: ZoneConfig | None = None,
                 config: BehaviourConfig | None = None, pose_config: PoseConfig | None = None):
        self.zones = zones or ZoneConfig(zones=[])
        self.config = config or BehaviourConfig()
        self.pose_analyzer = PoseBehaviourAnalyzer(pose_config)
        self.reset(session_id)

    def reset(self, session_id: str) -> None:
        """Clear all track state, event sequence, and unflushed events for a source restart."""
        self.session_id = session_id
        self._tracks: dict[int, _TrackState] = {}
        self._pending: list[SceneEvent] = []
        self._sequence = 0
        self._last_timestamp: datetime | None = None

    @property
    def pending_event_count(self) -> int:
        return len(self._pending)

    def drain_events(self) -> list[SceneEvent]:
        events, self._pending = self._pending, []
        return events

    def emit_room_event(
        self, kind: EventType, at: datetime, track_id: int | None = None,
        zone_id: str | None = None,
        metadata: dict[str, str | int | float | bool | None] | None = None,
    ) -> None:
        """Append a room transition to the same ordered session event stream."""
        self._event(kind, at, track_id, zone_id, metadata)

    def _event(
        self, kind: EventType, at: datetime, track_id: int | None,
        zone_id: str | None = None,
        metadata: dict[str, str | int | float | bool | None] | None = None,
    ) -> None:
        self._sequence += 1
        self._pending.append(SceneEvent(
            event_id=self._sequence, type=kind, timestamp_utc=at,
            session_id=self.session_id, track_id=track_id,
            zone_id=zone_id, metadata=metadata or {},
        ))

    def _expire(self, now: datetime, visible_ids: set[int]) -> None:
        for track_id, track in list(self._tracks.items()):
            missing_seconds = (now - track.last_seen).total_seconds()
            if missing_seconds > self.config.track_grace_seconds:
                anchor = track.history[-1].anchor
                self._event(EventType.PERSON_DISAPPEARED, now, track_id, metadata={
                    "missing_seconds": missing_seconds,
                    "last_anchor_x": anchor.x, "last_anchor_y": anchor.y,
                })
                del self._tracks[track_id]
            elif track_id not in visible_ids:
                # An unobserved interval cannot establish an inside/outside transition.
                for zone_state in track.zones.values():
                    zone_state.candidate = None
                    zone_state.candidate_since = None

    def _update_zones(self, track_id: int, track: _TrackState, observed: set[str], now: datetime,
                      anchor: NormalizedPoint) -> None:
        for zone in self.zones.zones:
            state = track.zones.setdefault(zone.id, _ZoneState())
            inside = zone.id in observed
            if inside == state.confirmed:
                state.candidate = None
                state.candidate_since = None
                continue
            if state.candidate != inside:
                state.candidate = inside
                state.candidate_since = now
            assert state.candidate_since is not None
            if (now - state.candidate_since).total_seconds() < self.config.zone_debounce_seconds:
                continue
            state.confirmed = inside
            state.candidate = None
            state.candidate_since = None
            if inside:
                state.entered_at = now
                self._event(EventType.ZONE_ENTERED, now, track_id, zone.id,
                            {"anchor_x": anchor.x, "anchor_y": anchor.y})
            else:
                dwell = (now - state.entered_at).total_seconds() if state.entered_at is not None else 0.0
                state.entered_at = None
                self._event(EventType.ZONE_EXITED, now, track_id, zone.id,
                            {"dwell_seconds": dwell, "anchor_x": anchor.x, "anchor_y": anchor.y})

    def _movement(self, track: _TrackState, now: datetime) -> tuple[float | None, float | None, MotionState]:
        window = [sample for sample in track.history
                  if (now - sample.at).total_seconds() <= self.config.motion_window_seconds]
        if len(window) < 2:
            return None, None, MotionState.UNKNOWN
        duration = (window[-1].at - window[0].at).total_seconds()
        if duration <= 0:
            return None, None, MotionState.UNKNOWN
        vx = (window[-1].anchor.x - window[0].anchor.x) / duration
        vy = (window[-1].anchor.y - window[0].anchor.y) / duration
        if len(window) < 4 or duration < self.config.min_observation_seconds:
            return vx, vy, MotionState.UNKNOWN
        first = window[0]
        scale_change = window[-1].scale - first.scale
        deltas = [b.scale - a.scale for a, b in zip(window, window[1:])]
        positive = sum(delta > 0.002 for delta in deltas)
        negative = sum(delta < -0.002 for delta in deltas)
        directional = positive + negative
        if directional >= 2 and scale_change >= self.config.scale_threshold and positive / directional >= 0.7:
            return vx, vy, MotionState.APPROACHING
        if directional >= 2 and scale_change <= -self.config.scale_threshold and negative / directional >= 0.7:
            return vx, vy, MotionState.DEPARTING
        max_position_change = max(hypot(sample.anchor.x - first.anchor.x, sample.anchor.y - first.anchor.y)
                                  for sample in window)
        max_scale_change = max(abs(sample.scale - first.scale) for sample in window)
        if max_position_change < self.config.movement_threshold and max_scale_change < self.config.scale_threshold:
            return vx, vy, MotionState.STATIONARY
        return vx, vy, MotionState.UNKNOWN

    def _update_motion(self, track_id: int, track: _TrackState, proposed: MotionState, now: datetime, scale: float, speed: float | None) -> None:
        if proposed == track.motion:
            track.motion_candidate = None
            track.motion_candidate_since = None
            return
        if proposed != track.motion_candidate:
            track.motion_candidate = proposed
            track.motion_candidate_since = now
        assert track.motion_candidate_since is not None
        if (now - track.motion_candidate_since).total_seconds() < self.config.motion_debounce_seconds:
            return
        track.motion = proposed
        track.motion_candidate = None
        track.motion_candidate_since = None
        event_kind = {
            MotionState.APPROACHING: EventType.APPROACH_STARTED,
            MotionState.DEPARTING: EventType.DEPARTURE_STARTED,
            MotionState.STATIONARY: EventType.BECAME_STATIONARY,
        }.get(proposed)
        if event_kind is not None:
            self._event(event_kind, now, track_id, metadata={"proximity_score": scale, "speed": speed})

    def update(self, people: list[PersonObservation], now: datetime) -> tuple[list[PersonObservation], dict[str, int]]:
        """Analyze one processed frame. Event output remains queued until drained."""
        require_utc(now)
        if self._last_timestamp is not None and now < self._last_timestamp:
            raise ValueError("frame timestamps must be nondecreasing")
        self._last_timestamp = now
        visible_ids = {person.track_id for person in people if person.track_id is not None}
        self._expire(now, visible_ids)
        occupancy = {zone.id: 0 for zone in self.zones.zones}
        analyzed = []
        for person in people:
            box = person.bbox
            anchor = NormalizedPoint(x=(box.x1 + box.x2) / 2, y=box.y2)
            area = (box.x2 - box.x1) * (box.y2 - box.y1)
            scale = box.y2 - box.y1  # normalized box height: visual proximity proxy
            observed = {zone.id for zone in self.zones.zones if contains(zone, anchor)}
            for zone_id in observed:
                occupancy[zone_id] += 1
            common = {
                "anchor_point": anchor, "bbox_area_ratio": area,
                "proximity_score": scale, "zones": sorted(observed),
            }
            if person.track_id is None:
                spatial = person.model_copy(update={
                    **common, "trajectory": [TrajectoryPoint(x=anchor.x, y=anchor.y, timestamp_utc=now)],
                })
                analyzed.append(spatial.model_copy(update={"pose_analysis": self.pose_analyzer.untracked(spatial)}))
                continue
            track_id = person.track_id
            track = self._tracks.get(track_id)
            if track is None:
                track = _TrackState(last_seen=now)
                self._tracks[track_id] = track
                self._event(EventType.PERSON_APPEARED, now, track_id,
                            metadata={"anchor_x": anchor.x, "anchor_y": anchor.y})
            self._update_zones(track_id, track, observed, now, anchor)
            track.raw_anchors.append(anchor)
            track.raw_scales.append(scale)
            if len(track.raw_anchors) == 3:
                filtered_anchor = NormalizedPoint(
                    x=median(point.x for point in track.raw_anchors),
                    y=median(point.y for point in track.raw_anchors),
                )
                filtered_scale = median(track.raw_scales)
            else:
                filtered_anchor, filtered_scale = anchor, scale
            if track.history:
                previous = track.history[-1]
                alpha = self.config.smoothing_alpha
                smoothed = NormalizedPoint(
                    x=alpha * filtered_anchor.x + (1 - alpha) * previous.anchor.x,
                    y=alpha * filtered_anchor.y + (1 - alpha) * previous.anchor.y,
                )
                smooth_scale = alpha * filtered_scale + (1 - alpha) * previous.scale
            else:
                smoothed, smooth_scale = filtered_anchor, filtered_scale
            confirmed = tuple(zone.id for zone in self.zones.zones if track.zones[zone.id].confirmed)
            track.history.append(_Sample(now, smoothed, smooth_scale, confirmed))
            while len(track.history) > 1 and (now - track.history[0].at).total_seconds() > self.config.history_seconds:
                track.history.popleft()
            track.last_seen = now
            vx, vy, proposed = self._movement(track, now)
            speed = hypot(vx, vy) if vx is not None and vy is not None else None
            self._update_motion(track_id, track, proposed, now, scale, speed)
            dwell = {zone_id: (now - track.zones[zone_id].entered_at).total_seconds()
                     for zone_id in confirmed if track.zones[zone_id].entered_at is not None}
            trajectory = [TrajectoryPoint(x=sample.anchor.x, y=sample.anchor.y, timestamp_utc=sample.at)
                          for sample in list(track.history)[-16:]]
            spatial = person.model_copy(update={
                **common,
                "velocity_x": vx, "velocity_y": vy, "speed": speed,
                "motion_state": track.motion, "confirmed_zones": list(confirmed),
                "dwell_seconds": dwell, "trajectory": trajectory,
            })
            analyzed.append(spatial.model_copy(update={
                "pose_analysis": self.pose_analyzer.analyze(spatial, track.pose, now, self._event),
            }))
        return analyzed, occupancy
