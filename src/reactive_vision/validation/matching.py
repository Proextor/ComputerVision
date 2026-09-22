"""Deterministic one-to-one event scoring without identity assumptions."""

from dataclasses import dataclass

from reactive_vision.schemas import SceneEvent

from .replay import EPOCH
from .schemas import Annotations, ExpectedEvent


def event_time(event: SceneEvent) -> float:
    return (event.timestamp_utc - EPOCH).total_seconds()


def expected_centre(expected: ExpectedEvent) -> float:
    if expected.time_seconds is not None:
        return expected.time_seconds
    assert expected.start_seconds is not None and expected.end_seconds is not None
    return (expected.start_seconds + expected.end_seconds) / 2


def compatible(expected: ExpectedEvent, predicted: SceneEvent, actors: dict[str, int],
               *, check_time: bool = True) -> bool:
    if predicted.type != expected.type:
        return False
    if expected.hand_side is not None and predicted.metadata.get("side") != expected.hand_side:
        return False
    if expected.zone_id is not None and predicted.zone_id != expected.zone_id:
        return False
    if expected.actor in actors and predicted.track_id != actors[expected.actor]:
        return False
    if not check_time:
        return True
    time = event_time(predicted)
    start = expected.time_seconds if expected.time_seconds is not None else expected.start_seconds
    end = expected.time_seconds if expected.time_seconds is not None else expected.end_seconds
    assert start is not None and end is not None
    return start - expected.tolerance_seconds <= time <= end + expected.tolerance_seconds


@dataclass(frozen=True)
class MatchedEvent:
    expected_index: int
    predicted_index: int
    offset_seconds: float


@dataclass
class MatchResult:
    matched: list[MatchedEvent]
    false_positive: list[int]
    false_negative: list[int]
    ignored_predictions: list[int]
    ignored_unscored: list[int]
    scenario_level_actors: list[str]


def match_events(annotations: Annotations, predictions: list[SceneEvent]) -> MatchResult:
    used: set[int] = set()
    matches: list[MatchedEvent] = []
    misses: list[int] = []
    order = sorted(range(len(annotations.events)),
                   key=lambda i: (expected_centre(annotations.events[i]), i))
    for index in order:
        expected = annotations.events[index]
        if expected.excluded:
            continue
        candidates = [i for i, prediction in enumerate(predictions)
                      if i not in used and compatible(expected, prediction, annotations.actor_tracks)]
        if not candidates:
            misses.append(index)
            continue
        nearest = min(candidates, key=lambda i: (
            abs(event_time(predictions[i]) - expected_centre(expected)),
            event_time(predictions[i]), predictions[i].event_id, i,
        ))
        used.add(nearest)
        matches.append(MatchedEvent(index, nearest,
                                     event_time(predictions[nearest]) - expected_centre(expected)))
    ignored_unscored = [i for i, prediction in enumerate(predictions)
                        if annotations.scored_event_types is not None and
                        prediction.type not in annotations.scored_event_types]
    ignored_excluded = [i for i, prediction in enumerate(predictions) if i not in used and
               i not in ignored_unscored and
               any(e.excluded and compatible(e, prediction, annotations.actor_tracks)
                   for e in annotations.events)]
    ignored = sorted(ignored_unscored + ignored_excluded)
    false_positive = [i for i in range(len(predictions)) if i not in used and i not in ignored]
    scenario_level = sorted({e.actor for e in annotations.events
                             if e.actor is not None and e.actor not in annotations.actor_tracks})
    return MatchResult(matches, false_positive, misses, ignored, ignored_unscored, scenario_level)
