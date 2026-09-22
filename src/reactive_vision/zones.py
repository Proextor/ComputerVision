"""Validated normalized polygon zones, without a geometry dependency."""

import json
from pathlib import Path
import re

from pydantic import Field, ValidationError, field_validator, model_validator

from .schemas import NormalizedPoint, StrictModel


ZONE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def _segments_intersect(a: tuple[float, float], b: tuple[float, float],
                        c: tuple[float, float], d: tuple[float, float]) -> bool:
    def orientation(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> bool:
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    return any((abs(o) < 1e-12 and on_segment(p, q, r)) for o, p, q, r in (
        (o1, a, c, b), (o2, a, d, b), (o3, c, a, d), (o4, c, b, d),
    ))


class Zone(StrictModel):
    id: str
    label: str
    role: str | None = None
    polygon: list[tuple[float, float]] = Field(min_length=3)

    @field_validator("id", "role")
    @classmethod
    def valid_identifier(cls, value: str | None) -> str | None:
        if value is not None and not ZONE_ID.fullmatch(value):
            raise ValueError("must start with a letter and contain only letters, digits, '_' or '-'")
        return value

    @field_validator("label")
    @classmethod
    def valid_label(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("label must not be empty")
        return value

    @field_validator("polygon")
    @classmethod
    def valid_polygon(cls, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for x, y in points:
            if not 0 <= x <= 1 or not 0 <= y <= 1:
                raise ValueError("polygon coordinates must be between 0 and 1")
        if len(set(points)) != len(points):
            raise ValueError("polygon vertices must be distinct")
        twice_area = sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
        if abs(twice_area) < 1e-8:
            raise ValueError("polygon must enclose a nonzero area")
        count = len(points)
        for i in range(count):
            for j in range(i + 1, count):
                if j == i + 1 or (i == 0 and j == count - 1):
                    continue
                if _segments_intersect(points[i], points[(i + 1) % count],
                                       points[j], points[(j + 1) % count]):
                    raise ValueError("polygon edges must not self-intersect")
        return points


class ZoneConfig(StrictModel):
    zones: list[Zone]

    @model_validator(mode="after")
    def unique_ids(self) -> "ZoneConfig":
        ids = [zone.id for zone in self.zones]
        if len(ids) != len(set(ids)):
            raise ValueError("zone IDs must be unique")
        return self


def load_zones(path: Path | None) -> ZoneConfig:
    if path is None:
        return ZoneConfig(zones=[])
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return ZoneConfig.model_validate(value)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise ValueError(f"Invalid zone configuration {path}: {error}") from error


def contains(zone: Zone, point: NormalizedPoint) -> bool:
    """Ray cast with polygon edges included as inside."""
    x, y = point.x, point.y
    inside = False
    polygon = zone.polygon
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= 1e-10 and min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10 and min(y1, y2) - 1e-10 <= y <= max(y1, y2) + 1e-10:
            return True
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside
