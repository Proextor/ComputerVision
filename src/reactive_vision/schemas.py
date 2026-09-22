"""JSON-safe scene contract. Coordinates are normalized to the source frame."""

from datetime import datetime
from enum import Enum
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CameraStatus(str, Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"
    READ_ERROR = "read_error"
    EOF = "eof"


class MotionState(str, Enum):
    APPROACHING = "approaching"
    DEPARTING = "departing"
    STATIONARY = "stationary"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    PERSON_APPEARED = "person_appeared"
    PERSON_DISAPPEARED = "person_disappeared"
    ZONE_ENTERED = "zone_entered"
    ZONE_EXITED = "zone_exited"
    APPROACH_STARTED = "approach_started"
    DEPARTURE_STARTED = "departure_started"
    BECAME_STATIONARY = "became_stationary"
    HAND_RAISE_STARTED = "hand_raise_started"
    HAND_RAISE_ENDED = "hand_raise_ended"
    BOTH_HANDS_RAISED_STARTED = "both_hands_raised_started"
    BOTH_HANDS_RAISED_ENDED = "both_hands_raised_ended"
    WAVE_DETECTED = "wave_detected"
    ACTIVITY_LEVEL_CHANGED = "activity_level_changed"
    ROOM_BECAME_OCCUPIED = "room_became_occupied"
    ROOM_BECAME_EMPTY = "room_became_empty"
    OCCUPANCY_STATE_CHANGED = "occupancy_state_changed"
    ROOM_ACTIVITY_CHANGED = "room_activity_changed"
    INTERACTION_STATE_CHANGED = "interaction_state_changed"
    FOCUS_CANDIDATE_CHANGED = "focus_candidate_changed"


class OccupancyState(str, Enum):
    UNKNOWN = "unknown"
    EMPTY = "empty"
    SINGLE = "single"
    SMALL_GROUP = "small_group"
    CROWD = "crowd"


class InteractionState(str, Enum):
    UNKNOWN = "unknown"
    NONE = "none"
    PRESENCE = "presence"
    APPROACHING = "approaching"
    SIGNAL_DETECTED = "signal_detected"


class PoseQuality(str, Enum):
    GOOD = "good"
    PARTIAL = "partial"
    POOR = "poor"
    UNAVAILABLE = "unavailable"


class GestureStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNKNOWN = "unknown"


class ActivityLevel(str, Enum):
    STILL = "still"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNKNOWN = "unknown"


class NormalizedPoint(StrictModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class TrajectoryPoint(NormalizedPoint):
    timestamp_utc: datetime

    @field_validator("timestamp_utc")
    @classmethod
    def utc_trajectory_timestamp(cls, value: datetime) -> datetime:
        return require_utc(value)


class SceneEvent(StrictModel):
    event_id: int = Field(ge=1)
    type: EventType
    timestamp_utc: datetime
    session_id: str
    track_id: int | None = Field(default=None, ge=0)
    zone_id: str | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("timestamp_utc")
    @classmethod
    def utc_event_timestamp(cls, value: datetime) -> datetime:
        return require_utc(value)

    @model_validator(mode="after")
    def zone_event_has_zone(self) -> "SceneEvent":
        if self.type in (EventType.ZONE_ENTERED, EventType.ZONE_EXITED) and self.zone_id is None:
            raise ValueError("zone event requires zone_id")
        room_types = {
            EventType.ROOM_BECAME_OCCUPIED, EventType.ROOM_BECAME_EMPTY,
            EventType.OCCUPANCY_STATE_CHANGED, EventType.ROOM_ACTIVITY_CHANGED,
            EventType.INTERACTION_STATE_CHANGED, EventType.FOCUS_CANDIDATE_CHANGED,
        }
        if self.type not in room_types and self.track_id is None:
            raise ValueError("person event requires track_id")
        return self


class BoundingBox(StrictModel):
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self) -> "BoundingBox":
        if self.x1 > self.x2 or self.y1 > self.y2:
            raise ValueError("box corners must be ordered")
        return self


class Keypoint(StrictModel):
    index: int = Field(ge=0)
    name: str
    x: float | None = Field(default=None, ge=0, le=1)
    y: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def coordinates_together(self) -> "Keypoint":
        if (self.x is None) != (self.y is None):
            raise ValueError("keypoint x and y must both be present or absent")
        return self


class PoseQualityObservation(StrictModel):
    usable_keypoint_count: int = Field(ge=0, le=17)
    usable_keypoint_ratio: float = Field(ge=0, le=1)
    pose_confidence: float | None = Field(default=None, ge=0, le=1)
    pose_quality: PoseQuality


class ActivityObservation(StrictModel):
    activity_score: float | None = Field(default=None, ge=0, le=1)
    activity_level: ActivityLevel = ActivityLevel.UNKNOWN
    contributing_joint_count: int = Field(default=0, ge=0, le=12)


class GestureObservation(StrictModel):
    left_hand_raised: GestureStatus = GestureStatus.UNKNOWN
    right_hand_raised: GestureStatus = GestureStatus.UNKNOWN
    both_hands_raised: GestureStatus = GestureStatus.UNKNOWN
    left_wave: GestureStatus = GestureStatus.UNKNOWN
    right_wave: GestureStatus = GestureStatus.UNKNOWN


class PoseAnalysisObservation(StrictModel):
    quality: PoseQualityObservation
    activity: ActivityObservation
    gestures: GestureObservation


class PersonObservation(StrictModel):
    # ByteTrack IDs are temporary identifiers within one camera session.
    track_id: int | None = Field(default=None, ge=0)
    bbox: BoundingBox
    confidence: float | None = Field(default=None, ge=0, le=1)
    keypoints: list[Keypoint] | None = None
    anchor_point: NormalizedPoint | None = None
    bbox_area_ratio: float | None = Field(default=None, ge=0, le=1)
    proximity_score: float | None = Field(default=None, ge=0, le=1)
    velocity_x: float | None = None
    velocity_y: float | None = None
    speed: float | None = Field(default=None, ge=0)
    motion_state: MotionState = MotionState.UNKNOWN
    zones: list[str] = Field(default_factory=list)
    confirmed_zones: list[str] = Field(default_factory=list)
    dwell_seconds: dict[str, float] = Field(default_factory=dict)
    trajectory: list[TrajectoryPoint] = Field(default_factory=list)
    pose_analysis: PoseAnalysisObservation | None = None


class RoomOccupancy(StrictModel):
    visible_count: int = Field(ge=0)
    stabilized_count: int | None = Field(default=None, ge=0)
    state: OccupancyState = OccupancyState.UNKNOWN


class HorizontalCounts(StrictModel):
    left: int = Field(ge=0)
    centre: int = Field(ge=0)
    right: int = Field(ge=0)


class RoomSpatial(StrictModel):
    centroid: NormalizedPoint | None = None
    horizontal_counts: HorizontalCounts
    mean_proximity_score: float | None = Field(default=None, ge=0, le=1)
    dispersion: float | None = Field(default=None, ge=0)
    zone_occupancy: dict[str, int] = Field(default_factory=dict)
    most_occupied_zone_id: str | None = None


class RoomActivity(StrictModel):
    aggregate_activity_score: float | None = Field(default=None, ge=0, le=1)
    aggregate_activity_level: ActivityLevel = ActivityLevel.UNKNOWN
    contributing_count: int = Field(ge=0)
    coverage_ratio: float = Field(ge=0, le=1)


class RoomMotionSummary(StrictModel):
    approaching: int = Field(ge=0)
    departing: int = Field(ge=0)
    stationary: int = Field(ge=0)
    unknown: int = Field(ge=0)


class RoomGestureSummary(StrictModel):
    left_hand_raised: int = Field(ge=0)
    right_hand_raised: int = Field(ge=0)
    both_hands_raised: int = Field(ge=0)
    recently_waving: int = Field(ge=0)
    unknown: int = Field(ge=0)


class RoomInteraction(StrictModel):
    state: InteractionState = InteractionState.UNKNOWN


class FocusCandidate(StrictModel):
    track_id: int = Field(ge=0)
    score: float = Field(ge=0, le=1)
    evidence: list[str]
    is_visible: bool


class RoomState(StrictModel):
    occupancy: RoomOccupancy
    spatial: RoomSpatial
    activity: RoomActivity
    motion_summary: RoomMotionSummary
    gesture_summary: RoomGestureSummary
    interaction: RoomInteraction
    focus_candidate: FocusCandidate | None = None


class SceneSnapshot(StrictModel):
    schema_version: str = "4.0"
    session_id: str
    frame_id: int = Field(ge=0)
    timestamp_utc: datetime
    source: str
    source_frame_index: int | None = Field(default=None, ge=0)
    source_time_seconds: float | None = Field(default=None, ge=0)
    frame_width: int | None = Field(default=None, gt=0)
    frame_height: int | None = Field(default=None, gt=0)
    camera_status: CameraStatus
    processing_fps: float | None = Field(default=None, ge=0)
    visible_count: int = Field(ge=0)
    people: list[PersonObservation]
    zone_occupancy: dict[str, int] = Field(default_factory=dict)
    room_state: RoomState
    events: list[SceneEvent] = Field(default_factory=list)

    @field_validator("timestamp_utc")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        return require_utc(value)

    @model_validator(mode="after")
    def count_matches_people(self) -> "SceneSnapshot":
        if self.visible_count != len(self.people):
            raise ValueError("visible_count must match current-frame people")
        if self.camera_status != CameraStatus.OK and self.people:
            raise ValueError("non-OK camera status cannot contain people")
        if any(count < 0 for count in self.zone_occupancy.values()):
            raise ValueError("zone occupancy counts cannot be negative")
        if any(event.session_id != self.session_id for event in self.events):
            raise ValueError("event session ID must match snapshot")
        if self.room_state is not None:
            if self.room_state.occupancy.visible_count != self.visible_count:
                raise ValueError("room visible count must match current-frame people")
            if self.room_state.spatial.zone_occupancy != self.zone_occupancy:
                raise ValueError("room zone occupancy must match scene zone occupancy")
            if self.camera_status != CameraStatus.OK and self.room_state.occupancy.state != OccupancyState.UNKNOWN:
                raise ValueError("failed camera requires unknown room occupancy")
            if self.camera_status != CameraStatus.OK and self.room_state.interaction.state != InteractionState.UNKNOWN:
                raise ValueError("failed camera requires unknown room interaction")
        return self


def require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must have a timezone")
    if value.utcoffset().total_seconds() != 0:
        raise ValueError("timestamp must be UTC")
    return value


def finite_or_none(value: object) -> float | None:
    """Convert tensor/NumPy scalars to a finite Python float, or explicit null."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def normalized_box(xyxy: object, width: int, height: int) -> BoundingBox | None:
    """Convert pixel xyxy to frame fractions; reject invalid or degenerate boxes."""
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    try:
        values = [finite_or_none(x) for x in xyxy]  # type: ignore[union-attr]
    except TypeError:
        return None
    if len(values) != 4 or any(x is None for x in values):
        return None
    x1, y1, x2, y2 = values
    assert x1 is not None and y1 is not None and x2 is not None and y2 is not None
    if x2 <= x1 or y2 <= y1:
        return None
    box = BoundingBox(
        x1=clamp01(x1 / width), y1=clamp01(y1 / height),
        x2=clamp01(x2 / width), y2=clamp01(y2 / height),
    )
    if box.x1 == box.x2 or box.y1 == box.y2:
        return None
    return box
