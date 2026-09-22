"""Time-based room summaries from structured observations; no camera or model imports."""

from collections import deque
from collections.abc import Callable
from datetime import datetime
import json
from math import hypot, isfinite
from pathlib import Path

from pydantic import Field, ValidationError, model_validator

from .schemas import (
    ActivityLevel, CameraStatus, EventType, FocusCandidate, GestureStatus,
    HorizontalCounts, InteractionState, MotionState, NormalizedPoint,
    OccupancyState, PersonObservation, PoseQuality, RoomActivity,
    RoomGestureSummary, RoomInteraction, RoomMotionSummary, RoomOccupancy,
    RoomSpatial, RoomState, StrictModel, require_utc,
)
from .zones import ZoneConfig


class FocusWeights(StrictModel):
    recent_wave: float = Field(default=0.45, ge=0, le=1)
    both_hands_raised: float = Field(default=0.40, ge=0, le=1)
    one_hand_raised: float = Field(default=0.25, ge=0, le=1)
    interaction_zone: float = Field(default=0.20, ge=0, le=1)
    approaching: float = Field(default=0.15, ge=0, le=1)
    proximity: float = Field(default=0.10, ge=0, le=1)
    dwell: float = Field(default=0.10, ge=0, le=1)


class RoomConfig(StrictModel):
    small_group_max: int = Field(default=4, ge=2)
    crowd_min: int = Field(default=5, ge=3)
    occupied_debounce_seconds: float = Field(default=0.35, ge=0)
    empty_debounce_seconds: float = Field(default=0.75, ge=0)
    horizontal_boundaries: tuple[float, float] = (1 / 3, 2 / 3)
    history_seconds: float = Field(default=3.0, gt=0)
    stale_input_seconds: float = Field(default=5.0, gt=0)
    activity_smoothing_alpha: float = Field(default=0.35, gt=0, le=1)
    activity_reference_seconds: float = Field(default=0.2, gt=0)
    minimum_activity_coverage: float = Field(default=0.5, ge=0, le=1)
    activity_thresholds: tuple[float, float, float] = (0.12, 0.30, 0.60)
    activity_debounce_seconds: float = Field(default=0.35, ge=0)
    interaction_debounce_seconds: float = Field(default=0.35, ge=0)
    focus_weights: FocusWeights = Field(default_factory=FocusWeights)
    minimum_focus_score: float = Field(default=0.15, ge=0, le=1)
    focus_switch_margin: float = Field(default=0.15, ge=0, le=1)
    focus_loss_tolerance_seconds: float = Field(default=0.75, ge=0)
    dwell_full_score_seconds: float = Field(default=3.0, gt=0)

    @model_validator(mode="after")
    def valid_order(self) -> "RoomConfig":
        if self.crowd_min != self.small_group_max + 1:
            raise ValueError("crowd_min must equal small_group_max + 1")
        left, right = self.horizontal_boundaries
        if not 0 < left < right < 1:
            raise ValueError("horizontal boundaries must satisfy 0 < left < right < 1")
        low, moderate, high = self.activity_thresholds
        if not 0 < low < moderate < high < 1:
            raise ValueError("activity thresholds must satisfy 0 < low < moderate < high < 1")
        return self


def load_room_config(path: Path | None) -> RoomConfig:
    if path is None:
        return RoomConfig()
    try:
        with path.open(encoding="utf-8") as handle:
            return RoomConfig.model_validate(json.load(handle))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise ValueError(f"Invalid room configuration {path}: {error}") from error


RoomEventEmitter = Callable[[EventType, datetime, int | None, str | None,
                             dict[str, str | int | float | bool | None] | None], None]


class RoomStateAnalyzer:
    """Analyze every frame; event emission is delegated to the session's M2 buffer."""

    def __init__(self, zones: ZoneConfig | None = None, config: RoomConfig | None = None):
        self.zones = zones or ZoneConfig(zones=[])
        self.config = config or RoomConfig()
        self._interaction_zones = {z.id for z in self.zones.zones if z.role == "interaction"}
        self.reset()

    def reset(self) -> None:
        self._history: deque[tuple[datetime, int]] = deque(maxlen=128)
        self._last_timestamp: datetime | None = None
        self._occupancy = OccupancyState.UNKNOWN
        self._stable_count: int | None = None
        self._occupancy_candidate: OccupancyState | None = None
        self._occupancy_since: datetime | None = None
        self._activity_score: float | None = None
        self._activity_last_sample_at: datetime | None = None
        self._activity_level = ActivityLevel.UNKNOWN
        self._activity_candidate: ActivityLevel | None = None
        self._activity_since: datetime | None = None
        self._interaction = InteractionState.UNKNOWN
        self._interaction_candidate: InteractionState | None = None
        self._interaction_since: datetime | None = None
        self._focus: FocusCandidate | None = None
        self._focus_last_seen: datetime | None = None
        self._focus_last_evidence: datetime | None = None

    @property
    def history_size(self) -> int:
        return len(self._history)

    def _event(self, emit: RoomEventEmitter, kind: EventType, now: datetime,
               metadata: dict[str, str | int | float | bool | None]) -> None:
        emit(kind, now, None, None, metadata)

    def _band(self, count: int) -> OccupancyState:
        if count == 0:
            return OccupancyState.EMPTY
        if count == 1:
            return OccupancyState.SINGLE
        if count <= self.config.small_group_max:
            return OccupancyState.SMALL_GROUP
        return OccupancyState.CROWD

    def _update_occupancy(self, count: int, now: datetime, emit: RoomEventEmitter) -> None:
        proposed = self._band(count)
        if proposed == self._occupancy:
            self._stable_count = count
            self._occupancy_candidate = None
            self._occupancy_since = None
            return
        if proposed != self._occupancy_candidate:
            self._occupancy_candidate = proposed
            self._occupancy_since = now
        assert self._occupancy_since is not None
        debounce = (self.config.empty_debounce_seconds if proposed == OccupancyState.EMPTY
                    else self.config.occupied_debounce_seconds)
        if (now - self._occupancy_since).total_seconds() < debounce:
            return
        previous = self._occupancy
        previous_count = self._stable_count
        self._occupancy = proposed
        self._stable_count = count
        self._occupancy_candidate = None
        self._occupancy_since = None
        # Unknown at startup/failure is an unconfirmed baseline, not an observed transition.
        if previous == OccupancyState.UNKNOWN:
            return
        metadata = {"previous": previous.value, "new": proposed.value,
                    "previous_count": previous_count, "new_count": count}
        self._event(emit, EventType.OCCUPANCY_STATE_CHANGED, now, metadata)
        if previous == OccupancyState.EMPTY and count > 0:
            self._event(emit, EventType.ROOM_BECAME_OCCUPIED, now, metadata)
        elif previous != OccupancyState.EMPTY and proposed == OccupancyState.EMPTY:
            self._event(emit, EventType.ROOM_BECAME_EMPTY, now, metadata)

    def _spatial(self, people: list[PersonObservation], occupancy: dict[str, int]) -> RoomSpatial:
        anchors = [p.anchor_point for p in people if p.anchor_point is not None]
        left_edge, right_edge = self.config.horizontal_boundaries
        horizontal = HorizontalCounts(
            left=sum(p.x < left_edge for p in anchors),
            centre=sum(left_edge <= p.x < right_edge for p in anchors),
            right=sum(p.x >= right_edge for p in anchors),
        )
        centroid = None
        dispersion = None
        if anchors:
            centroid = NormalizedPoint(x=sum(p.x for p in anchors) / len(anchors),
                                       y=sum(p.y for p in anchors) / len(anchors))
            dispersion = (sum(hypot(p.x - centroid.x, p.y - centroid.y) ** 2
                              for p in anchors) / len(anchors)) ** 0.5
        scales = [p.proximity_score for p in people if p.proximity_score is not None]
        mean_proximity = sum(scales) / len(scales) if scales else None
        eligible = [zone_id for zone_id, count in occupancy.items() if count > 0]
        top_zone = min(eligible, key=lambda zone_id: (-occupancy[zone_id], zone_id)) if eligible else None
        return RoomSpatial(centroid=centroid, horizontal_counts=horizontal,
                           mean_proximity_score=mean_proximity, dispersion=dispersion,
                           zone_occupancy=occupancy, most_occupied_zone_id=top_zone)

    def _activity(self, people: list[PersonObservation], now: datetime,
                  emit: RoomEventEmitter) -> RoomActivity:
        weighted = []
        for person in people:
            pose = person.pose_analysis
            if pose is None or pose.activity.activity_score is None:
                continue
            weight = {PoseQuality.GOOD: 1.0, PoseQuality.PARTIAL: 0.5}.get(pose.quality.pose_quality)
            if weight is not None:
                weighted.append((pose.activity.activity_score, weight))
        coverage = len(weighted) / len(people) if people else 0.0
        if not people or not weighted or coverage < self.config.minimum_activity_coverage:
            self._activity_candidate = None
            self._activity_since = None
            self._activity_level = ActivityLevel.UNKNOWN
            self._activity_score = None
            self._activity_last_sample_at = None
            return RoomActivity(contributing_count=len(weighted), coverage_ratio=coverage)
        raw = sum(score * weight for score, weight in weighted) / sum(weight for _, weight in weighted)
        if self._activity_score is None or self._activity_last_sample_at is None:
            self._activity_score = raw
        else:
            elapsed = max(0.0, (now - self._activity_last_sample_at).total_seconds())
            alpha = 1 - (1 - self.config.activity_smoothing_alpha) ** (
                elapsed / self.config.activity_reference_seconds)
            self._activity_score = alpha * raw + (1 - alpha) * self._activity_score
        self._activity_last_sample_at = now
        low, moderate, high = self.config.activity_thresholds
        proposed = (ActivityLevel.STILL if self._activity_score < low else
                    ActivityLevel.LOW if self._activity_score < moderate else
                    ActivityLevel.MODERATE if self._activity_score < high else ActivityLevel.HIGH)
        if proposed == self._activity_level:
            self._activity_candidate = None
            self._activity_since = None
        else:
            if proposed != self._activity_candidate:
                self._activity_candidate = proposed
                self._activity_since = now
            assert self._activity_since is not None
            if (now - self._activity_since).total_seconds() >= self.config.activity_debounce_seconds:
                previous = self._activity_level
                self._activity_level = proposed
                self._activity_candidate = None
                self._activity_since = None
                if previous != ActivityLevel.UNKNOWN:
                    self._event(emit, EventType.ROOM_ACTIVITY_CHANGED, now,
                                {"previous": previous.value, "new": proposed.value,
                                 "score": self._activity_score})
        return RoomActivity(aggregate_activity_score=self._activity_score,
                            aggregate_activity_level=self._activity_level,
                            contributing_count=len(weighted), coverage_ratio=coverage)

    @staticmethod
    def _motion(people: list[PersonObservation]) -> RoomMotionSummary:
        counts = {state: sum(p.motion_state == state for p in people) for state in MotionState}
        return RoomMotionSummary(approaching=counts[MotionState.APPROACHING],
                                 departing=counts[MotionState.DEPARTING],
                                 stationary=counts[MotionState.STATIONARY],
                                 unknown=counts[MotionState.UNKNOWN])

    @staticmethod
    def _gestures(people: list[PersonObservation]) -> RoomGestureSummary:
        left = right = both = wave = unknown = 0
        for person in people:
            g = person.pose_analysis.gestures if person.pose_analysis else None
            if g is None:
                unknown += 1
                continue
            left += g.left_hand_raised == GestureStatus.ACTIVE
            right += g.right_hand_raised == GestureStatus.ACTIVE
            both += g.both_hands_raised == GestureStatus.ACTIVE
            wave += g.left_wave == GestureStatus.ACTIVE or g.right_wave == GestureStatus.ACTIVE
            unknown += (g.left_hand_raised == GestureStatus.UNKNOWN and
                        g.right_hand_raised == GestureStatus.UNKNOWN)
        return RoomGestureSummary(left_hand_raised=left, right_hand_raised=right,
                                  both_hands_raised=both, recently_waving=wave,
                                  unknown=unknown)

    def _update_interaction(self, people: list[PersonObservation], now: datetime,
                            emit: RoomEventEmitter) -> RoomInteraction:
        signal = any(p.pose_analysis and (
            p.pose_analysis.gestures.left_hand_raised == GestureStatus.ACTIVE or
            p.pose_analysis.gestures.right_hand_raised == GestureStatus.ACTIVE or
            p.pose_analysis.gestures.left_wave == GestureStatus.ACTIVE or
            p.pose_analysis.gestures.right_wave == GestureStatus.ACTIVE)
            for p in people)
        approach = any(p.motion_state == MotionState.APPROACHING or
                       bool(self._interaction_zones.intersection(p.confirmed_zones)) for p in people)
        proposed = (InteractionState.SIGNAL_DETECTED if signal else
                    InteractionState.APPROACHING if approach else
                    InteractionState.PRESENCE if people else
                    InteractionState.NONE if self._occupancy == OccupancyState.EMPTY else
                    self._interaction)
        if proposed == self._interaction:
            self._interaction_candidate = None
            self._interaction_since = None
        else:
            if proposed != self._interaction_candidate:
                self._interaction_candidate = proposed
                self._interaction_since = now
            assert self._interaction_since is not None
            if (now - self._interaction_since).total_seconds() >= self.config.interaction_debounce_seconds:
                previous = self._interaction
                self._interaction = proposed
                self._interaction_candidate = None
                self._interaction_since = None
                if previous != InteractionState.UNKNOWN:
                    self._event(emit, EventType.INTERACTION_STATE_CHANGED, now,
                                {"previous": previous.value, "new": proposed.value})
        return RoomInteraction(state=self._interaction)

    def _focus_score(self, person: PersonObservation) -> FocusCandidate | None:
        if person.track_id is None:
            return None
        g = person.pose_analysis.gestures if person.pose_analysis else None
        w = self.config.focus_weights
        score = 0.0
        evidence: list[str] = []
        def add(name: str, value: float) -> None:
            nonlocal score
            evidence.append(name)
            score += value
        if g and (g.left_wave == GestureStatus.ACTIVE or g.right_wave == GestureStatus.ACTIVE):
            add("recent_wave", w.recent_wave)
        if g and g.both_hands_raised == GestureStatus.ACTIVE:
            add("both_hands_raised", w.both_hands_raised)
        elif g and (g.left_hand_raised == GestureStatus.ACTIVE or g.right_hand_raised == GestureStatus.ACTIVE):
            add("one_hand_raised", w.one_hand_raised)
        if self._interaction_zones.intersection(person.confirmed_zones):
            add("interaction_zone", w.interaction_zone)
        if person.motion_state == MotionState.APPROACHING:
            add("approaching", w.approaching)
        # Proximity and dwell modify a cue; they never create focus from ordinary presence.
        if not evidence:
            return None
        if person.proximity_score is not None:
            score += w.proximity * person.proximity_score
            evidence.append("proximity")
        if person.dwell_seconds:
            score += w.dwell * min(1.0, max(person.dwell_seconds.values()) / self.config.dwell_full_score_seconds)
            evidence.append("dwell")
        score = min(1.0, score)
        if score < self.config.minimum_focus_score:
            return None
        return FocusCandidate(track_id=person.track_id, score=score,
                              evidence=evidence, is_visible=True)

    def _update_focus(self, people: list[PersonObservation], now: datetime,
                      emit: RoomEventEmitter) -> FocusCandidate | None:
        candidates = {candidate.track_id: candidate for person in people
                      if (candidate := self._focus_score(person)) is not None}
        visible_ids = {p.track_id for p in people if p.track_id is not None}
        previous = self._focus
        current = previous
        if current is not None and current.track_id in visible_ids:
            self._focus_last_seen = now
            observed = candidates.get(current.track_id)
            if observed is not None:
                current = observed
                self._focus_last_evidence = now
            else:
                evidence_gap = ((now - self._focus_last_evidence).total_seconds()
                                if self._focus_last_evidence else float("inf"))
                current = (current.model_copy(update={"is_visible": True})
                           if evidence_gap <= self.config.focus_loss_tolerance_seconds else None)
        elif current is not None:
            missing = (now - self._focus_last_seen).total_seconds() if self._focus_last_seen else float("inf")
            current = (current.model_copy(update={"is_visible": False})
                       if missing <= self.config.focus_loss_tolerance_seconds else None)
        ranked = sorted(candidates.values(), key=lambda c: (-c.score, c.track_id))
        challenger = next((c for c in ranked if current is None or c.track_id != current.track_id), None)
        if challenger is not None and (current is None or
                                       challenger.score > current.score + self.config.focus_switch_margin):
            current = challenger
            self._focus_last_seen = now
            self._focus_last_evidence = now
        if current is None:
            self._focus_last_seen = None
            self._focus_last_evidence = None
        self._focus = current
        old_id = previous.track_id if previous else None
        new_id = current.track_id if current else None
        if old_id != new_id:
            self._event(emit, EventType.FOCUS_CANDIDATE_CHANGED, now,
                        {"previous": old_id, "new": new_id,
                         "previous_score": previous.score if previous else None,
                         "new_score": current.score if current else None,
                         "focus_track_id": new_id,
                         "focus_score": current.score if current else None})
        return current

    def update(self, people: list[PersonObservation], camera_status: CameraStatus,
               now: datetime, zone_occupancy: dict[str, int],
               emit: RoomEventEmitter, input_age_seconds: float = 0.0) -> RoomState:
        require_utc(now)
        if self._last_timestamp is not None and now < self._last_timestamp:
            raise ValueError("room timestamps must be nondecreasing")
        if not isfinite(input_age_seconds) or input_age_seconds < 0:
            raise ValueError("input age must be finite and nonnegative")
        self._last_timestamp = now
        unavailable = camera_status != CameraStatus.OK or input_age_seconds > self.config.stale_input_seconds
        if unavailable:
            if people and camera_status != CameraStatus.OK:
                raise ValueError("unavailable camera cannot have current people")
            self.reset()
            self._last_timestamp = now
            return RoomState(
                occupancy=RoomOccupancy(visible_count=len(people)),
                spatial=self._spatial(people, zone_occupancy),
                activity=RoomActivity(contributing_count=0, coverage_ratio=0),
                motion_summary=self._motion([]), gesture_summary=self._gestures([]),
                interaction=RoomInteraction(),
            )
        self._history.append((now, len(people)))
        while self._history and (now - self._history[0][0]).total_seconds() > self.config.history_seconds:
            self._history.popleft()
        self._update_occupancy(len(people), now, emit)
        return RoomState(
            occupancy=RoomOccupancy(visible_count=len(people), stabilized_count=self._stable_count,
                                    state=self._occupancy),
            spatial=self._spatial(people, zone_occupancy),
            activity=self._activity(people, now, emit),
            motion_summary=self._motion(people),
            gesture_summary=self._gestures(people),
            interaction=self._update_interaction(people, now, emit),
            focus_candidate=self._update_focus(people, now, emit),
        )
