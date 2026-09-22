"""Event, state and stability metrics on source-relative time."""

from bisect import bisect_right
from collections import Counter, defaultdict
from statistics import median

from reactive_vision.schemas import SceneEvent

from .matching import MatchResult, event_time, expected_centre
from .replay import ReplayResult
from .schemas import Annotations, StateInterval
from .trace import percentile

SAMPLE_SECONDS = 0.1  # Fixed 10 Hz timeline, latest replay scene at or before sample.


def ratios(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def event_metrics(annotations: Annotations, predictions: list[SceneEvent],
                  matches: MatchResult, duration: float) -> dict:
    kinds = sorted({e.type.value for e in annotations.events if not e.excluded} |
                   {e.value for e in annotations.scored_event_types or []} |
                   {predictions[i].type.value for i in matches.false_positive})

    def group(kind: str | None) -> dict:
        selected = [m for m in matches.matched
                    if kind is None or annotations.events[m.expected_index].type.value == kind]
        fp = [i for i in matches.false_positive if kind is None or predictions[i].type.value == kind]
        fn = [i for i in matches.false_negative if kind is None or annotations.events[i].type.value == kind]
        absolute = [abs(m.offset_seconds) for m in selected]
        return {**ratios(len(selected), len(fp), len(fn)),
                "false_events_per_minute": len(fp) * 60 / duration if duration > 0 else None,
                "median_offset_seconds": median([m.offset_seconds for m in selected]) if selected else None,
                "median_absolute_error_seconds": median(absolute) if absolute else None,
                "p95_absolute_error_seconds": percentile(absolute, 95) if len(absolute) >= 5 else None,
                "timing_offsets_seconds": [m.offset_seconds for m in selected]}

    return {"overall": group(None), "by_type": {kind: group(kind) for kind in kinds}}


def _value(scene, field: str, interval: StateInterval, actors: dict[str, int]):
    if field == "visible_count":
        return scene.visible_count
    if field == "occupancy_state":
        return scene.room_state.occupancy.state.value
    if field == "interaction_state":
        return scene.room_state.interaction.state.value
    if field == "activity_level":
        return scene.room_state.activity.aggregate_activity_level.value
    # Person state is only scored with an explicitly supplied actor-to-track mapping.
    track = actors.get(interval.actor) if interval.actor else None
    person = next((p for p in scene.people if p.track_id == track), None) if track is not None else None
    if field == "motion_state":
        return person.motion_state.value if person else "unknown"
    if field == "zone_membership":
        return tuple(sorted(person.confirmed_zones)) if person else ("unknown",)
    raise ValueError(f"unsupported state metric {field}")


def state_metrics(annotations: Annotations, replay: ReplayResult) -> dict:
    counts: dict[str, Counter] = defaultdict(Counter)
    confusion: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    delays: dict[str, list[float]] = defaultdict(list)
    transitions: dict[str, Counter] = defaultdict(Counter)
    skipped = Counter()
    times = replay.frame_seconds
    if not times:
        return {"sample_interval_seconds": SAMPLE_SECONDS, "by_field": {}, "skipped_unmapped": {}}
    # Sample in [first, last], with each interval end exclusive.
    sample_count = int((times[-1] - times[0]) / SAMPLE_SECONDS) + 1
    fields = ("visible_count", "occupancy_state", "interaction_state",
              "motion_state", "activity_level", "zone_membership")
    previous_expected: dict[tuple[str, str | None], tuple[object, float]] = {}
    for interval in sorted(annotations.states, key=lambda item: item.start_seconds):
        if interval.excluded:
            continue
        for field in fields:
            expected = getattr(interval, field)
            if expected is None:
                continue
            if field in ("motion_state", "zone_membership") and interval.actor not in annotations.actor_tracks:
                skipped[field] += 1
                continue
            expected_value = tuple(sorted(expected)) if field == "zone_membership" else (
                expected.value if hasattr(expected, "value") else expected)
            expected_label = str(expected_value)
            sample_values: list[tuple[float, object]] = []
            for tick in range(sample_count):
                at = times[0] + tick * SAMPLE_SECONDS
                if not interval.start_seconds <= at < interval.end_seconds:
                    continue
                scene = replay.scenes[bisect_right(times, at) - 1]
                predicted = _value(scene, field, interval, annotations.actor_tracks)
                label = str(predicted)
                counts[field]["samples"] += 1
                counts[field]["correct"] += predicted == expected_value
                counts[field]["unknown"] += predicted == "unknown" or predicted == ("unknown",)
                confusion[field][expected_label][label] += 1
                sample_values.append((at, predicted))
            key = field, interval.actor if field in ("motion_state", "zone_membership") else None
            previous = previous_expected.get(key)
            if previous is not None and previous[0] != expected_value and sample_values:
                transitions[field]["annotated"] += 1
                first = next((at for at, value in sample_values if value == expected_value), None)
                if first is not None:
                    delays[field].append(max(0.0, first - interval.start_seconds))
                else:
                    transitions[field]["missed"] += 1
            previous_expected[key] = expected_value, interval.end_seconds
    return {"sample_interval_seconds": SAMPLE_SECONDS,
            "by_field": {field: {
                "accuracy": value["correct"] / value["samples"] if value["samples"] else None,
                "samples": value["samples"],
                "unknown_coverage": value["unknown"] / value["samples"] if value["samples"] else None,
                "confusion": {expected: dict(predicted) for expected, predicted in confusion[field].items()},
                "transition_delays_seconds": delays[field],
                "median_transition_delay_seconds": median(delays[field]) if delays[field] else None,
                "transitions_annotated": transitions[field]["annotated"],
                "transitions_missed": transitions[field]["missed"],
            } for field, value in counts.items()},
            "skipped_unmapped": dict(skipped)}


def stability_metrics(annotations: Annotations, replay: ReplayResult,
                      predictions: list[SceneEvent]) -> dict:
    duration = replay.duration_seconds
    repeated = 0
    seen: dict[tuple, float] = {}
    reappeared = Counter()
    first_appearance: dict[int, float] = {}
    short_tracks = 0
    for event in predictions:
        at = event_time(event)
        key = (event.type.value, event.track_id, event.zone_id, event.metadata.get("side"))
        if key in seen and at - seen[key] <= 1:
            repeated += 1
        seen[key] = at
        if event.type.value == "person_appeared":
            reappeared[event.track_id] += 1
            if event.track_id is not None:
                first_appearance[event.track_id] = at
        if event.type.value == "person_disappeared" and event.track_id in first_appearance:
            short_tracks += at - first_appearance.pop(event.track_id) <= 1
    occlusions = [interval for interval in annotations.states
                  if interval.brief_occlusion and not interval.excluded]
    disappearance = sum(
        event.type.value == "person_disappeared" and
        any(interval.start_seconds <= event_time(event) <= interval.end_seconds and
            (interval.actor not in annotations.actor_tracks or
             event.track_id == annotations.actor_tracks[interval.actor])
            for interval in occlusions)
        for event in predictions
    )
    occupancies = [(time, scene.room_state.occupancy.state.value)
                   for time, scene in zip(replay.frame_seconds, replay.scenes)]
    changes = [(t, value) for i, (t, value) in enumerate(occupancies)
               if i and value != occupancies[i - 1][1]]
    transitions = occupancies[:1] + changes
    flicker = sum(transitions[i][1] == transitions[i - 2][1] and
                  transitions[i][0] - transitions[i - 1][0] <= 1
                  for i in range(2, len(transitions)))
    activity = [scene.room_state.activity.aggregate_activity_level.value for scene in replay.scenes]
    return {"repeated_duplicate_transitions": repeated,
            "false_disappearances_during_brief_occlusion": disappearance,
            "focus_switches_per_minute": sum(
                e.type.value == "focus_candidate_changed" and
                e.metadata.get("previous") is not None and e.metadata.get("new") is not None and
                e.metadata.get("previous") != e.metadata.get("new")
                for e in predictions) * 60 / duration if duration else None,
            "occupancy_flicker_count": flicker,
            "activity_level_transition_count": sum(a != b for a, b in zip(activity, activity[1:])),
            "repeated_track_start_indicators": sum(count - 1 for count in reappeared.values() if count > 1),
            "short_lived_track_indicators": short_tracks,
            "tracking_accuracy": None}
