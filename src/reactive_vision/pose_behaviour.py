"""Rule-based pose quality, hand gestures, waves, and movement intensity.

All state belongs to an M2 track. This component does not own track expiry or event IDs.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from math import hypot
from pathlib import Path
from statistics import mean, median
from typing import Callable

from pydantic import Field, ValidationError, model_validator

from .schemas import (
    ActivityLevel, ActivityObservation, EventType, GestureObservation,
    GestureStatus, Keypoint, NormalizedPoint, PersonObservation,
    PoseAnalysisObservation, PoseQuality, PoseQualityObservation, StrictModel,
)


class PoseConfig(StrictModel):
    min_keypoint_confidence: float = Field(default=0.45, ge=0, le=1)
    gesture_activation_seconds: float = Field(default=0.30, ge=0)
    gesture_release_seconds: float = Field(default=0.35, ge=0)
    missing_tolerance_seconds: float = Field(default=0.50, ge=0)
    raised_margin_ratio: float = Field(default=0.12, gt=0, le=1)
    wave_window_seconds: float = Field(default=2.0, gt=0)
    wave_min_amplitude_ratio: float = Field(default=0.15, gt=0)
    wave_min_reversals: int = Field(default=2, ge=2)
    wave_cooldown_seconds: float = Field(default=2.0, ge=0)
    wave_display_seconds: float = Field(default=0.80, ge=0)
    activity_smoothing_alpha: float = Field(default=0.35, gt=0, le=1)
    activity_thresholds: tuple[float, float, float] = (0.12, 0.30, 0.60)
    activity_debounce_seconds: float = Field(default=0.35, ge=0)

    @model_validator(mode="after")
    def valid_thresholds(self) -> "PoseConfig":
        a, b, c = self.activity_thresholds
        if not 0 < a < b < c < 1:
            raise ValueError("activity_thresholds must be three increasing values between 0 and 1")
        if self.wave_display_seconds > self.wave_cooldown_seconds and self.wave_cooldown_seconds > 0:
            raise ValueError("wave_display_seconds must not exceed wave_cooldown_seconds")
        return self


def load_pose_config(path: Path | None) -> PoseConfig:
    if path is None:
        return PoseConfig()
    try:
        with path.open(encoding="utf-8") as handle:
            return PoseConfig.model_validate(json.load(handle))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise ValueError(f"Invalid pose configuration {path}: {error}") from error


@dataclass
class _HandState:
    status: GestureStatus = GestureStatus.UNKNOWN
    candidate: bool | None = None
    candidate_since: datetime | None = None
    candidate_count: int = 0
    last_valid: datetime | None = None


@dataclass(frozen=True)
class _WaveSample:
    at: datetime
    x: float


@dataclass
class _WaveState:
    samples: deque[_WaveSample] = field(default_factory=lambda: deque(maxlen=64))
    raw_x: deque[float] = field(default_factory=lambda: deque(maxlen=3))
    last_valid: datetime | None = None
    detected_at: datetime | None = None
    cooldown_until: datetime | None = None
    latched: bool = False
    last_qualified_at: datetime | None = None


@dataclass(frozen=True)
class _ActivityFrame:
    at: datetime
    anchor: NormalizedPoint
    scale: float
    joints: dict[int, tuple[float, float]]


@dataclass
class PoseTrackState:
    left_hand: _HandState = field(default_factory=_HandState)
    right_hand: _HandState = field(default_factory=_HandState)
    left_wave: _WaveState = field(default_factory=_WaveState)
    right_wave: _WaveState = field(default_factory=_WaveState)
    both_hands: GestureStatus = GestureStatus.UNKNOWN
    previous_activity: _ActivityFrame | None = None
    activity_ewma: float | None = None
    activity_level: ActivityLevel = ActivityLevel.UNKNOWN
    activity_candidate: ActivityLevel | None = None
    activity_candidate_since: datetime | None = None


EventSink = Callable[..., None]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class PoseBehaviourAnalyzer:
    def __init__(self, config: PoseConfig | None = None):
        self.config = config or PoseConfig()

    def _usable(self, person: PersonObservation) -> tuple[dict[int, Keypoint], PoseQualityObservation]:
        available: dict[int, Keypoint] = {}
        for point in person.keypoints or []:
            if (0 <= point.index < 17 and point.x is not None and point.y is not None
                    and point.confidence is not None
                    and point.confidence >= self.config.min_keypoint_confidence):
                available[point.index] = point
        count = len(available)
        if not person.keypoints:
            quality = PoseQuality.UNAVAILABLE
        elif count >= 12:
            quality = PoseQuality.GOOD
        elif count >= 6:
            quality = PoseQuality.PARTIAL
        else:
            quality = PoseQuality.POOR
        return available, PoseQualityObservation(
            usable_keypoint_count=count,
            usable_keypoint_ratio=count / 17,
            pose_confidence=mean(point.confidence for point in available.values()) if available else None,
            pose_quality=quality,
        )

    def untracked(self, person: PersonObservation) -> PoseAnalysisObservation:
        _, quality = self._usable(person)
        return PoseAnalysisObservation(
            quality=quality, activity=ActivityObservation(), gestures=GestureObservation(),
        )

    def _hand_evidence(self, points: dict[int, Keypoint], side: str, margin: float) -> bool | None:
        shoulder_id, elbow_id, wrist_id = (5, 7, 9) if side == "left" else (6, 8, 10)
        shoulder, wrist = points.get(shoulder_id), points.get(wrist_id)
        if shoulder is None or wrist is None:
            return None
        assert shoulder.y is not None and wrist.y is not None
        above = shoulder.y - wrist.y
        if above >= margin:
            elbow = points.get(elbow_id)
            if elbow is not None and elbow.y is not None and wrist.y > elbow.y + margin * 0.5:
                return None
            return True
        if above <= -margin * 0.5:
            return False
        return None

    def _hand(
        self, state: _HandState, evidence: bool | None, now: datetime,
        side: str, track_id: int, wrist_confidence: float | None, emit: EventSink,
    ) -> GestureStatus:
        if evidence is None:
            state.candidate = None
            state.candidate_since = None
            state.candidate_count = 0
            if (state.status != GestureStatus.ACTIVE or state.last_valid is None
                    or (now - state.last_valid).total_seconds() > self.config.missing_tolerance_seconds):
                state.status = GestureStatus.UNKNOWN
            return state.status
        state.last_valid = now
        desired = GestureStatus.ACTIVE if evidence else GestureStatus.INACTIVE
        if desired == state.status:
            state.candidate = None
            state.candidate_since = None
            state.candidate_count = 0
            return state.status
        if state.status == GestureStatus.UNKNOWN and desired == GestureStatus.INACTIVE:
            state.status = GestureStatus.INACTIVE
            return state.status
        if state.candidate != evidence:
            state.candidate = evidence
            state.candidate_since = now
            state.candidate_count = 1
        else:
            state.candidate_count += 1
        assert state.candidate_since is not None
        duration = self.config.gesture_activation_seconds if evidence else self.config.gesture_release_seconds
        if state.candidate_count < 2 or (now - state.candidate_since).total_seconds() < duration:
            return state.status
        previous = state.status
        state.status = desired
        state.candidate = None
        state.candidate_since = None
        state.candidate_count = 0
        if desired == GestureStatus.ACTIVE:
            emit(EventType.HAND_RAISE_STARTED, now, track_id,
                 metadata={"side": side, "confidence": wrist_confidence})
        elif previous == GestureStatus.ACTIVE:
            emit(EventType.HAND_RAISE_ENDED, now, track_id,
                 metadata={"side": side, "confidence": wrist_confidence})
        return state.status

    def _wave_pattern(self, samples: deque[_WaveSample]) -> tuple[bool, float, int]:
        if len(samples) < 5 or (samples[-1].at - samples[0].at).total_seconds() < 0.4:
            return False, 0.0, 0
        values = [sample.x for sample in samples]
        amplitude = max(values) - min(values)
        if amplitude < self.config.wave_min_amplitude_ratio:
            return False, amplitude, 0
        step = max(0.035, self.config.wave_min_amplitude_ratio * 0.22)
        direction = 0
        pivot = extreme = values[0]
        reversals = 0
        for value in values[1:]:
            if direction == 0:
                if value - pivot >= step:
                    direction, extreme = 1, value
                elif pivot - value >= step:
                    direction, extreme = -1, value
            elif direction > 0:
                if value > extreme:
                    extreme = value
                elif extreme - value >= step:
                    reversals += 1
                    direction, pivot, extreme = -1, extreme, value
            else:
                if value < extreme:
                    extreme = value
                elif value - extreme >= step:
                    reversals += 1
                    direction, pivot, extreme = 1, extreme, value
        return reversals >= self.config.wave_min_reversals, amplitude, reversals

    def _wave(
        self, state: _WaveState, points: dict[int, Keypoint], side: str,
        scale: float, margin: float, now: datetime, track_id: int, emit: EventSink,
    ) -> GestureStatus:
        shoulder_id, wrist_id = (5, 9) if side == "left" else (6, 10)
        shoulder, wrist = points.get(shoulder_id), points.get(wrist_id)
        evidence_valid = shoulder is not None and wrist is not None
        around_shoulder = False
        if evidence_valid:
            assert shoulder.x is not None and shoulder.y is not None
            assert wrist.x is not None and wrist.y is not None
            state.last_valid = now
            around_shoulder = wrist.y <= shoulder.y + margin * 0.5
            if around_shoulder:
                relative_x = (wrist.x - shoulder.x) / scale
                state.raw_x.append(relative_x)
                filtered = median(state.raw_x) if len(state.raw_x) == 3 else relative_x
                smoothed = filtered if not state.samples else 0.65 * filtered + 0.35 * state.samples[-1].x
                state.samples.append(_WaveSample(now, smoothed))
            else:
                state.samples.clear()
                state.raw_x.clear()
                state.latched = False
        while state.samples and (now - state.samples[0].at).total_seconds() > self.config.wave_window_seconds:
            state.samples.popleft()
        if evidence_valid and around_shoulder:
            qualified, amplitude, reversals = self._wave_pattern(state.samples)
            if qualified:
                state.last_qualified_at = now
                if not state.latched and (state.cooldown_until is None or now >= state.cooldown_until):
                    state.detected_at = now
                    state.cooldown_until = now + timedelta(seconds=self.config.wave_cooldown_seconds)
                    state.latched = True
                    emit(EventType.WAVE_DETECTED, now, track_id, metadata={
                        "side": side, "amplitude_ratio": amplitude,
                        "reversals": reversals, "confidence": wrist.confidence,
                    })
            elif (state.last_qualified_at is not None
                  and (now - state.last_qualified_at).total_seconds() > 0.5):
                state.latched = False
        if state.detected_at is not None and (now - state.detected_at).total_seconds() <= self.config.wave_display_seconds:
            return GestureStatus.ACTIVE
        if not evidence_valid:
            return GestureStatus.UNKNOWN
        if not around_shoulder:
            return GestureStatus.INACTIVE
        return GestureStatus.INACTIVE if len(state.samples) >= 3 else GestureStatus.UNKNOWN

    def _activity_frame(self, points: dict[int, Keypoint], anchor: NormalizedPoint,
                        scale: float, now: datetime) -> _ActivityFrame | None:
        body = {index: point for index, point in points.items() if 5 <= index <= 16}
        if len(body) < 6:
            return None
        torso = [body[index] for index in (5, 6, 11, 12) if index in body]
        center_x = mean(point.x for point in torso) if len(torso) >= 2 else anchor.x
        center_y = mean(point.y for point in torso) if len(torso) >= 2 else anchor.y
        joints = {index: ((point.x - center_x) / scale, (point.y - center_y) / scale)
                  for index, point in body.items()}
        return _ActivityFrame(now, anchor, scale, joints)

    def _activity(self, state: PoseTrackState, points: dict[int, Keypoint],
                  anchor: NormalizedPoint, scale: float, now: datetime,
                  track_id: int, emit: EventSink) -> ActivityObservation:
        current = self._activity_frame(points, anchor, scale, now)
        if current is None:
            state.previous_activity = None
            state.activity_candidate = None
            state.activity_candidate_since = None
            return ActivityObservation()
        previous = state.previous_activity
        state.previous_activity = current
        if previous is None:
            return ActivityObservation()
        elapsed = (now - previous.at).total_seconds()
        common = sorted(current.joints.keys() & previous.joints.keys())
        if elapsed <= 0 or elapsed > 1.0 or len(common) < 6:
            return ActivityObservation(contributing_joint_count=len(common))
        speeds = [min(1.5, hypot(current.joints[index][0] - previous.joints[index][0],
                                 current.joints[index][1] - previous.joints[index][1]) / elapsed)
                  for index in common]
        strongest = sorted(speeds, reverse=True)[:4]
        pose_component = _clamp01(mean(strongest) / 0.7)
        anchor_rate = hypot(anchor.x - previous.anchor.x, anchor.y - previous.anchor.y) / elapsed
        scale_rate = abs(scale - previous.scale) / elapsed
        raw_score = 0.72 * pose_component + 0.18 * _clamp01(anchor_rate / 0.35) + 0.10 * _clamp01(scale_rate / 0.35)
        alpha = self.config.activity_smoothing_alpha
        state.activity_ewma = raw_score if state.activity_ewma is None else alpha * raw_score + (1 - alpha) * state.activity_ewma
        score = _clamp01(state.activity_ewma)
        still, low, moderate = self.config.activity_thresholds
        proposed = (ActivityLevel.STILL if score < still else
                    ActivityLevel.LOW if score < low else
                    ActivityLevel.MODERATE if score < moderate else ActivityLevel.HIGH)
        if proposed == state.activity_level:
            state.activity_candidate = None
            state.activity_candidate_since = None
        else:
            if proposed != state.activity_candidate:
                state.activity_candidate = proposed
                state.activity_candidate_since = now
            assert state.activity_candidate_since is not None
            if (now - state.activity_candidate_since).total_seconds() >= self.config.activity_debounce_seconds:
                previous_level = state.activity_level
                state.activity_level = proposed
                state.activity_candidate = None
                state.activity_candidate_since = None
                emit(EventType.ACTIVITY_LEVEL_CHANGED, now, track_id, metadata={
                    "from": previous_level.value, "to": proposed.value, "activity_score": score,
                })
        return ActivityObservation(
            activity_score=score, activity_level=state.activity_level,
            contributing_joint_count=len(common),
        )

    def analyze(self, person: PersonObservation, state: PoseTrackState, now: datetime,
                emit: EventSink) -> PoseAnalysisObservation:
        """Analyze a visible tracked person, using only M2-owned per-track state."""
        assert person.track_id is not None and person.anchor_point is not None
        points, quality = self._usable(person)
        track_id = person.track_id
        scale = max(person.bbox.y2 - person.bbox.y1, 1e-6)
        margin = self.config.raised_margin_ratio * scale
        left_wrist, right_wrist = points.get(9), points.get(10)
        left = self._hand(state.left_hand, self._hand_evidence(points, "left", margin), now,
                          "left", track_id, left_wrist.confidence if left_wrist else None, emit)
        right = self._hand(state.right_hand, self._hand_evidence(points, "right", margin), now,
                           "right", track_id, right_wrist.confidence if right_wrist else None, emit)
        both = (GestureStatus.ACTIVE if left == right == GestureStatus.ACTIVE else
                GestureStatus.INACTIVE if GestureStatus.INACTIVE in (left, right) else GestureStatus.UNKNOWN)
        old_both = state.both_hands
        state.both_hands = both
        if both == GestureStatus.ACTIVE and old_both != GestureStatus.ACTIVE:
            emit(EventType.BOTH_HANDS_RAISED_STARTED, now, track_id,
                 metadata={"left_confidence": left_wrist.confidence if left_wrist else None,
                           "right_confidence": right_wrist.confidence if right_wrist else None})
        elif old_both == GestureStatus.ACTIVE and both == GestureStatus.INACTIVE:
            emit(EventType.BOTH_HANDS_RAISED_ENDED, now, track_id, metadata={})
        left_wave = self._wave(state.left_wave, points, "left", scale, margin, now, track_id, emit)
        right_wave = self._wave(state.right_wave, points, "right", scale, margin, now, track_id, emit)
        activity = self._activity(state, points, person.anchor_point, scale, now, track_id, emit)
        return PoseAnalysisObservation(
            quality=quality, activity=activity,
            gestures=GestureObservation(
                left_hand_raised=left, right_hand_raised=right, both_hands_raised=both,
                left_wave=left_wave, right_wave=right_wave,
            ),
        )
