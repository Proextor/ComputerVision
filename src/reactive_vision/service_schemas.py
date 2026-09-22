"""M5 transport contract, separate from SceneSnapshot schema 4.0."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, field_validator

from .schemas import CameraStatus, SceneSnapshot, StrictModel, require_utc


class ServiceStatus(str, Enum):
    STARTING = "starting"
    READY = "ready"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    EOF = "eof"
    ERROR = "error"
    STOPPING = "stopping"


class HealthResponse(StrictModel):
    service_status: ServiceStatus
    worker_alive: bool
    session_id: str | None
    camera_status: CameraStatus | None
    latest_frame_id: int | None = Field(default=None, ge=0)
    publication_sequence: int = Field(ge=0)
    last_snapshot_age_seconds: float | None = Field(default=None, ge=0)
    connected_websocket_clients: int = Field(ge=0)
    schema_version: Literal["4.0"] = "4.0"
    source: str
    last_worker_error: str | None = None


class ReadinessResponse(StrictModel):
    ready: bool
    service_status: ServiceStatus
    detail: str
    publication_sequence: int = Field(ge=0)
    last_snapshot_age_seconds: float | None = Field(default=None, ge=0)


class ApiErrorResponse(StrictModel):
    code: Literal["snapshot_not_available"] = "snapshot_not_available"
    detail: str
    service_status: ServiceStatus


class SnapshotEnvelope(StrictModel):
    type: Literal["snapshot"] = "snapshot"
    publication_sequence: int = Field(ge=1)
    published_at: datetime
    snapshot: SceneSnapshot

    @field_validator("published_at")
    @classmethod
    def utc_publication(cls, value: datetime) -> datetime:
        return require_utc(value)


class HeartbeatEnvelope(StrictModel):
    type: Literal["heartbeat"] = "heartbeat"
    publication_sequence: int = Field(ge=0)
    service_status: ServiceStatus
    last_snapshot_age_seconds: float | None = Field(default=None, ge=0)
