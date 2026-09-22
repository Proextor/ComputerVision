"""Versioned, strict portable validation contracts."""

from pathlib import Path
import re
from typing import Literal

from pydantic import Field, model_validator, field_validator

from reactive_vision.schemas import (
    BoundingBox, CameraStatus, EventType, Keypoint, StrictModel,
    MotionState, OccupancyState, InteractionState, ActivityLevel,
)

TRACE_VERSION = "1.0"
MANIFEST_VERSION = "1.0"
ANNOTATION_VERSION = "1.0"


class TracePerson(StrictModel):
    track_id: int | None = Field(default=None, ge=0)
    bbox: BoundingBox
    confidence: float | None = Field(default=None, ge=0, le=1)
    keypoints: list[Keypoint] | None = None


class TraceFrame(StrictModel):
    type: Literal["frame"] = "frame"
    trace_version: Literal["1.0"] = TRACE_VERSION
    source_id: str = Field(min_length=1)
    frame_id: int = Field(ge=0)
    time_seconds: float = Field(ge=0)
    frame_width: int | None = Field(default=None, gt=0)
    frame_height: int | None = Field(default=None, gt=0)
    camera_status: CameraStatus
    people: list[TracePerson]
    source_frame_index: int | None = Field(default=None, ge=0)
    source_time_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def consistent(self):
        if self.camera_status != CameraStatus.OK and self.people:
            raise ValueError("non-OK trace frame cannot contain people")
        if self.camera_status == CameraStatus.OK and (self.frame_width is None or self.frame_height is None):
            raise ValueError("OK trace frame requires dimensions")
        return self


class TraceHeader(StrictModel):
    type: Literal["metadata"] = "metadata"
    trace_version: Literal["1.0"] = TRACE_VERSION
    source_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_sha256: str | None = None
    video_sha256: str | None = None
    dependencies: dict[str, str]
    detector_settings: dict[str, str | int | float | bool | None]
    extraction_runtime: dict[str, float | int | None] = Field(default_factory=dict)

    @field_validator("model_sha256", "video_sha256")
    @classmethod
    def hash_is_valid(cls, value: str | None):
        if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("SHA-256 must be 64 lowercase hex characters")
        return value


class Scenario(StrictModel):
    id: str
    description: str
    split: Literal["calibration", "validation", "diagnostic"]
    trace: str
    annotations: str
    video: str | None = None
    zones: str | None = None
    pose_config: str | None = None
    room_config: str | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", value):
            raise ValueError("scenario ID must be an alphanumeric identifier")
        return value

    def resolve(self, manifest: Path, field: str) -> Path | None:
        value = getattr(self, field)
        return (manifest.parent / value).resolve() if value is not None else None


class Manifest(StrictModel):
    manifest_version: Literal["1.0"] = MANIFEST_VERSION
    scenarios: list[Scenario] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [scenario.id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate scenario IDs")
        return self


class ExpectedEvent(StrictModel):
    type: EventType
    time_seconds: float | None = Field(default=None, ge=0)
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    tolerance_seconds: float = Field(default=0.3, ge=0, le=30)
    actor: str | None = None
    hand_side: Literal["left", "right"] | None = None
    zone_id: str | None = None
    excluded: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def time_definition(self):
        if self.time_seconds is None and (self.start_seconds is None or self.end_seconds is None):
            raise ValueError("event needs time_seconds or a complete time window")
        if self.time_seconds is not None and (self.start_seconds is not None or self.end_seconds is not None):
            raise ValueError("event must use either a point or a window")
        if self.start_seconds is not None and self.end_seconds < self.start_seconds:
            raise ValueError("event window end precedes start")
        return self


class StateInterval(StrictModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    visible_count: int | None = Field(default=None, ge=0)
    occupancy_state: OccupancyState | None = None
    interaction_state: InteractionState | None = None
    motion_state: MotionState | None = None
    activity_level: ActivityLevel | None = None
    zone_membership: list[str] | None = None
    actor: str | None = None
    brief_occlusion: bool = False
    excluded: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def valid_interval(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError("state interval must have positive duration")
        if all(getattr(self, key) is None for key in (
            "visible_count", "occupancy_state", "interaction_state",
            "motion_state", "activity_level", "zone_membership",
        )) and not self.brief_occlusion:
            raise ValueError("state interval needs a state or brief_occlusion")
        if (self.motion_state is not None or self.zone_membership is not None) and self.actor is None:
            raise ValueError("person state interval requires an anonymous actor")
        return self


class Annotations(StrictModel):
    annotation_version: Literal["1.0"] = ANNOTATION_VERSION
    scenario_id: str
    actor_tracks: dict[str, int] = Field(default_factory=dict)
    scored_event_types: list[EventType] | None = None
    events: list[ExpectedEvent] = Field(default_factory=list)
    states: list[StateInterval] = Field(default_factory=list)

    @model_validator(mode="after")
    def actors_are_anonymous(self):
        labels = set(self.actor_tracks)
        labels.update(event.actor for event in self.events if event.actor)
        labels.update(interval.actor for interval in self.states if interval.actor)
        if any(not re.fullmatch(r"person_[a-z0-9_]+", label) for label in labels):
            raise ValueError("use anonymous labels such as person_a")
        if any(track < 0 for track in self.actor_tracks.values()):
            raise ValueError("temporary track IDs must be nonnegative")
        if self.scored_event_types is not None:
            if len(self.scored_event_types) != len(set(self.scored_event_types)):
                raise ValueError("scored_event_types contains duplicates")
            if any(event.type not in self.scored_event_types for event in self.events if not event.excluded):
                raise ValueError("a scored expected event is outside scored_event_types")
        return self


class Criteria(StrictModel):
    minimum_event_f1: float | None = Field(default=None, ge=0, le=1)
    maximum_false_events_per_minute: float | None = Field(default=None, ge=0)
    maximum_median_latency_seconds: float | None = Field(default=None, ge=0)
    minimum_occupancy_accuracy: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def at_least_one(self):
        if not self.model_dump(exclude_none=True):
            raise ValueError("criteria file must specify at least one target")
        return self


class ValidationReport(StrictModel):
    report_version: Literal["1.0"] = "1.0"
    run_timestamp_utc: str
    git_commit: str | None
    manifest_sha256: str
    scenarios: list[dict]
    groups: dict[str, dict]
    criteria: dict | None = None
    limitations: list[str]
