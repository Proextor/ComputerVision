"""Manifest validation and scenario evaluation using recorded observations."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import subprocess
import tracemalloc

from pydantic import ValidationError

from .matching import event_time, expected_centre, match_events
from .metrics import event_metrics, ratios, stability_metrics, state_metrics
from .replay import configs, replay, with_overrides
from .schemas import Annotations, Criteria, Manifest, Scenario, TraceHeader, ValidationReport
from .trace import hash_file, load_trace, percentile


def load_manifest(path: Path) -> Manifest:
    try:
        manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise ValueError(f"invalid manifest {path.name}: {error}") from error
    missing = [(scenario.id, field, str(scenario.resolve(path, field)))
               for scenario in manifest.scenarios
               for field in ("trace", "annotations", "video", "zones", "pose_config", "room_config")
               if scenario.resolve(path, field) is not None and not scenario.resolve(path, field).is_file()]
    if missing:
        raise ValueError("missing scenario files:\n" + "\n".join(
            f"  {scenario} {field}: {location}" for scenario, field, location in missing))
    return manifest


def load_annotations(path: Path, scenario_id: str) -> Annotations:
    try:
        annotation = Annotations.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise ValueError(f"invalid annotations {path.name}: {error}") from error
    if annotation.scenario_id != scenario_id:
        raise ValueError(f"annotation scenario ID differs from {scenario_id}")
    return annotation


def ensure_split_isolation(path: Path, manifest: Manifest) -> None:
    """Reject the same tracked trace or source video in tuning and held-out splits."""
    seen: dict[tuple[str, str], tuple[str, str]] = {}
    for scenario in manifest.scenarios:
        if scenario.split not in ("calibration", "validation"):
            continue
        trace_path = scenario.resolve(path, "trace")
        assert trace_path is not None
        with trace_path.open(encoding="utf-8") as handle:
            header = TraceHeader.model_validate_json(handle.readline())
        keys = [("trace", hash_file(trace_path))]
        if header.video_sha256 is not None:
            keys.append(("video", header.video_sha256))
        video_path = scenario.resolve(path, "video")
        if video_path is not None:
            keys.append(("video", hash_file(video_path)))
        for key in keys:
            previous = seen.get(key)
            if previous is not None and previous[0] != scenario.split:
                raise ValueError(f"split leakage: {previous[1]} ({previous[0]}) and "
                                 f"{scenario.id} ({scenario.split}) reuse the same {key[0]}")
            seen[key] = scenario.split, scenario.id


def evaluate_scenario(manifest_path: Path, scenario: Scenario, *, overrides: dict | None = None) -> dict:
    trace_path = scenario.resolve(manifest_path, "trace")
    annotation_path = scenario.resolve(manifest_path, "annotations")
    assert trace_path is not None and annotation_path is not None
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    try:
        header, frames = load_trace(trace_path)
        trace_peak_bytes = tracemalloc.get_traced_memory()[1] if not tracing else None
    finally:
        if not tracing:
            tracemalloc.stop()
    video_path = scenario.resolve(manifest_path, "video")
    if video_path is not None and header.video_sha256 != hash_file(video_path):
        raise ValueError(f"video hash differs from trace metadata for {scenario.id}; re-extract this scenario")
    annotation = load_annotations(annotation_path, scenario.id)
    zone_path = scenario.resolve(manifest_path, "zones")
    pose_path = scenario.resolve(manifest_path, "pose_config")
    room_path = scenario.resolve(manifest_path, "room_config")
    zone_config, behaviour_config, pose_config, room_config = with_overrides(
        configs(zone_path, pose_path, room_path), overrides)
    result = replay(header, frames, session_id=f"replay-{scenario.id}",
                    zones=zone_config, behaviour=behaviour_config, pose=pose_config, room=room_config)
    predictions = [event for scene in result.scenes for event in scene.events]
    matched = match_events(annotation, predictions)
    metrics = event_metrics(annotation, predictions, matched, result.duration_seconds)
    states = state_metrics(annotation, result)
    stability = stability_metrics(annotation, result, predictions)
    comparison = {
        "matched": [{"type": annotation.events[m.expected_index].type.value,
                     "expected_seconds": expected_centre(annotation.events[m.expected_index]),
                     "predicted_seconds": event_time(predictions[m.predicted_index]),
                     "offset_seconds": m.offset_seconds,
                     "prediction_event_id": predictions[m.predicted_index].event_id}
                    for m in matched.matched],
        "false_positive": [{"type": predictions[i].type.value, "time_seconds": event_time(predictions[i]),
                            "event_id": predictions[i].event_id} for i in matched.false_positive],
        "false_negative": [{"type": annotation.events[i].type.value,
                            "time_seconds": expected_centre(annotation.events[i])} for i in matched.false_negative],
        "ignored_predictions": len(matched.ignored_predictions),
        "ignored_unscored_predictions": len(matched.ignored_unscored),
        "scored_event_types": [kind.value for kind in annotation.scored_event_types]
        if annotation.scored_event_types is not None else "all",
        "actors_scored_at_scenario_level": matched.scenario_level_actors,
    }
    effective_config = {"behaviour": vars(behaviour_config),
                        "pose": pose_config.model_dump(), "room": room_config.model_dump(),
                        "zones": zone_config.model_dump()}
    effective_hash = sha256(json.dumps(effective_config, sort_keys=True,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"id": scenario.id, "description": scenario.description,
            "split": scenario.split, "tags": scenario.tags,
            "input_hashes": {"trace_sha256": hash_file(trace_path),
                             "annotations_sha256": hash_file(annotation_path),
                             "zones_sha256": hash_file(zone_path) if zone_path else None,
                             "pose_sha256": hash_file(pose_path) if pose_path else None,
                             "room_sha256": hash_file(room_path) if room_path else None,
                             "video_sha256": header.video_sha256,
                             "effective_config_sha256": effective_hash},
            "extraction": {"model": header.model_name, "model_sha256": header.model_sha256,
                           "dependencies": header.dependencies, "settings": header.detector_settings,
                           "runtime": header.extraction_runtime},
            "effective_config": effective_config,
            "events": metrics, "states": states, "stability": stability,
            "replay_runtime": {**result.runtime(), "trace_file_bytes": trace_path.stat().st_size,
                               "peak_trace_load_bytes": trace_peak_bytes},
            "comparison": comparison}


def aggregate(scenarios: list[dict]) -> dict:
    duration = sum(s["replay_runtime"]["video_duration_seconds"] for s in scenarios)
    kinds = sorted({kind for s in scenarios for kind in s["events"]["by_type"]})

    def event_group(kind: str | None) -> dict:
        rows = [s["events"]["overall" if kind is None else "by_type"].get(kind)
                if kind is not None else s["events"]["overall"] for s in scenarios]
        rows = [row for row in rows if row is not None]
        tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
        offsets = [offset for row in rows for offset in row["timing_offsets_seconds"]]
        absolute = [abs(value) for value in offsets]
        return {**ratios(tp, fp, fn),
                "false_events_per_minute": fp * 60 / duration if duration > 0 else None,
                "median_offset_seconds": median(offsets) if offsets else None,
                "median_absolute_error_seconds": median(absolute) if absolute else None,
                "p95_absolute_error_seconds": percentile(absolute, 95) if len(absolute) >= 5 else None,
                "timing_offsets_seconds": offsets}

    fields: dict[str, dict] = {}
    for field in sorted({name for s in scenarios for name in s["states"]["by_field"]}):
        rows = [s["states"]["by_field"][field] for s in scenarios if field in s["states"]["by_field"]]
        count = sum(row["samples"] for row in rows)
        confusion: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            for expected, predictions in row["confusion"].items():
                confusion[expected].update(predictions)
        correct = sum(confusion[label][label] for label in confusion)
        unknown = sum(sum(n for predicted, n in values.items() if predicted in ("unknown", "('unknown',)"))
                      for values in confusion.values())
        delays = [value for row in rows for value in row["transition_delays_seconds"]]
        fields[field] = {"samples": count, "accuracy": correct / count if count else None,
                         "unknown_coverage": unknown / count if count else None,
                         "confusion": {key: dict(value) for key, value in confusion.items()},
                         "transition_delays_seconds": delays,
                         "median_transition_delay_seconds": median(delays) if delays else None,
                         "transitions_annotated": sum(row["transitions_annotated"] for row in rows),
                         "transitions_missed": sum(row["transitions_missed"] for row in rows)}
    return {"scenario_count": len(scenarios), "duration_seconds": duration,
            "events": {"overall": event_group(None),
                       "by_type": {kind: event_group(kind) for kind in kinds}},
            "states": {"sample_interval_seconds": 0.1, "by_field": fields}}


def criteria_results(criteria: Criteria, group: dict) -> dict:
    event = group["events"]["overall"]
    occupancy = group["states"]["by_field"].get("occupancy_state", {})
    observed = {"minimum_event_f1": event["f1"],
                "maximum_false_events_per_minute": event["false_events_per_minute"],
                "maximum_median_latency_seconds": event["median_absolute_error_seconds"],
                "minimum_occupancy_accuracy": occupancy.get("accuracy")}
    checked = {}
    for name, target in criteria.model_dump(exclude_none=True).items():
        value = observed[name]
        passed = value is not None and (value >= target if name.startswith("minimum") else value <= target)
        checked[name] = {"target": target, "observed": value, "passed": passed,
                         "reason": None if value is not None else "metric unavailable"}
    return {"passed": all(value["passed"] for value in checked.values()), "checks": checked}


def evaluate(manifest_path: Path, *, scenario_ids: set[str] | None = None,
             overrides: dict | None = None, criteria: Criteria | None = None) -> ValidationReport:
    manifest = load_manifest(manifest_path)
    ensure_split_isolation(manifest_path, manifest)
    selected = [scenario for scenario in manifest.scenarios
                if scenario_ids is None or scenario.id in scenario_ids]
    if not selected:
        raise ValueError("no matching scenarios")
    scenarios = [evaluate_scenario(manifest_path, item, overrides=overrides) for item in selected]
    groups = {f"split:{split}": aggregate([s for s in scenarios if s["split"] == split])
              for split in ("calibration", "validation", "diagnostic")
              if any(s["split"] == split for s in scenarios)}
    groups["headline"] = aggregate([s for s in scenarios if s["split"] != "diagnostic"])
    for tag in sorted({tag for s in scenarios for tag in s["tags"]}):
        groups[f"tag:{tag}"] = aggregate([s for s in scenarios if tag in s["tags"]])
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], capture_output=True,
                                text=True, check=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    target = "split:validation" if "split:validation" in groups else "split:calibration"
    criteria_outcome = ({"target_group": target,
                         **criteria_results(criteria, groups[target])}
                        if criteria is not None and target in groups else None)
    if criteria is not None and criteria_outcome is None:
        criteria_outcome = {"target_group": None, "passed": False,
                            "checks": {}, "reason": "no scored calibration or validation scenarios"}
    return ValidationReport(
        run_timestamp_utc=datetime.now(timezone.utc).isoformat(), git_commit=commit or None,
        manifest_sha256=hash_file(manifest_path), scenarios=scenarios, groups=groups,
        criteria=criteria_outcome,
        limitations=["No accuracy claim without consented, annotated real recordings.",
                     "Unmapped anonymous actors are scored at scenario level; track IDs are temporary.",
                     "Frame identity tracking accuracy is not measured.",
                     "State accuracy samples the latest trace scene every 0.1 seconds.",
                     "Sparse or narrow scenarios cannot establish generalization."],
    )
