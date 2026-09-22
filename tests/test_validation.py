"""M7 contracts, replay, scoring and calibration without footage or model downloads."""

from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from reactive_vision.behaviour import BehaviourAnalyzer
from reactive_vision.capture import CaptureResult
from reactive_vision.schemas import BoundingBox, CameraStatus, EventType, PersonObservation, SceneEvent
from reactive_vision.validation.cli import main
from reactive_vision.validation.annotate import template
from reactive_vision.validation.matching import match_events
from reactive_vision.validation.metrics import event_metrics, ratios, stability_metrics, state_metrics
from reactive_vision.validation.replay import EPOCH, replay
from reactive_vision.validation.report import markdown, save_report
from reactive_vision.validation.runner import evaluate, load_annotations, load_manifest
from reactive_vision.validation.schemas import (
    Annotations, Criteria, ExpectedEvent, Manifest, Scenario,
    StateInterval, TraceFrame, TraceHeader, TracePerson,
)
from reactive_vision.validation.sweep import SweepDefinition, load_sweep, run_sweep
from reactive_vision.validation.trace import extract, hash_file, load_trace, write_trace


def header():
    return TraceHeader(source_id="synthetic", model_name="synthetic-pose.pt",
                       dependencies={"test": "1"}, detector_settings={"confidence": 0.25})


def person(track=1):
    return TracePerson(track_id=track, bbox=BoundingBox(x1=.3, y1=.3, x2=.5, y2=.9), confidence=.9)


def frames():
    return [TraceFrame(source_id="synthetic", frame_id=i, time_seconds=i * .2,
                       source_frame_index=i, source_time_seconds=i * .2,
                       frame_width=320, frame_height=240, camera_status=CameraStatus.OK,
                       people=[person()] if 2 <= i <= 5 else []) for i in range(11)]


def prediction(kind, seconds, event_id, *, side=None, zone=None, track=1):
    return SceneEvent(type=kind, timestamp_utc=EPOCH + timedelta(seconds=seconds),
                      session_id="replay-s", track_id=track, event_id=event_id,
                      zone_id=zone, metadata={"side": side} if side else {})


def collection(tmp_path: Path, *, validation=True):
    trace = tmp_path / "trace.jsonl"
    write_trace(trace, header(), frames())
    annotation = tmp_path / "truth.json"
    annotation.write_text(Annotations(scenario_id="sample", events=[
        ExpectedEvent(type="person_appeared", time_seconds=.4, tolerance_seconds=.1),
    ], states=[StateInterval(start_seconds=.4, end_seconds=.8, visible_count=1)]).model_dump_json(),
        encoding="utf-8")
    scenarios = [dict(id="sample", description="tiny no-person recording", split="calibration",
                      trace="trace.jsonl", annotations="truth.json", tags=["entry"])]
    if validation:
        held_frames = frames()
        held_frames[2] = held_frames[2].model_copy(update={"frame_id": 2,
            "people": [TracePerson(track_id=1, bbox=BoundingBox(x1=.31, y1=.3, x2=.51, y2=.9))]})
        write_trace(tmp_path / "held_trace.jsonl", header(), held_frames)
        second = tmp_path / "truth_held.json"
        second.write_text(Annotations(scenario_id="held", events=[
            ExpectedEvent(type="person_disappeared", time_seconds=1.8, tolerance_seconds=.3),
        ]).model_dump_json(), encoding="utf-8")
        scenarios.append(dict(id="held", description="untuned holdout", split="validation",
                              trace="held_trace.jsonl", annotations="truth_held.json", tags=["occlusion"]))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"manifest_version": "1.0", "scenarios": scenarios}), encoding="utf-8")
    return path


def test_trace_roundtrip_order_version_and_pixel_rejection(tmp_path):
    path = tmp_path / "trace.jsonl"
    write_trace(path, header(), frames())
    got_header, got_frames = load_trace(path)
    assert got_header.source_id == "synthetic" and got_frames == frames()
    assert "image" not in path.read_text(encoding="utf-8")
    with pytest.raises(ValidationError):
        TraceFrame.model_validate({**frames()[0].model_dump(), "image_base64": "unwanted"})
    text = path.read_text(encoding="utf-8").replace('"trace_version":"1.0"', '"trace_version":"2.0"')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible trace"):
        load_trace(path)


def test_trace_nonmonotonic_and_missing_timestamp(tmp_path):
    changed = frames()
    changed[3] = changed[3].model_copy(update={"time_seconds": .1})
    with pytest.raises(ValueError, match="timestamps"):
        write_trace(tmp_path / "bad.jsonl", header(), changed)
    changed[3] = changed[3].model_copy(update={"time_seconds": .6, "frame_id": 1})
    with pytest.raises(ValueError, match="IDs"):
        write_trace(tmp_path / "bad.jsonl", header(), changed)


def test_extraction_no_pixels_and_metadata_with_fake_tracker(tmp_path):
    video = tmp_path / "input.mp4"; video.write_bytes(b"synthetic video placeholder")
    weights = tmp_path / "weights.pt"; weights.write_bytes(b"model bytes")
    class Source:
        def __init__(self, *_):
            self.items = iter([CaptureResult(CameraStatus.OK, np.zeros((8, 8, 3), dtype=np.uint8), 0, 0),
                               CaptureResult(CameraStatus.OK, np.zeros((8, 8, 3), dtype=np.uint8), 1, .2),
                               CaptureResult(CameraStatus.EOF)])
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self): return next(self.items)
    class Tracker:
        def __init__(self, *_): pass
        def process(self, _): return [PersonObservation(track_id=1, bbox=person().bbox, confidence=.9)]
    target = tmp_path / "trace.jsonl"
    extracted = extract(video, target, model=str(weights), source_factory=Source, tracker_factory=Tracker)
    _, observations = load_trace(target)
    assert extracted.video_sha256 == hash_file(video)
    assert extracted.model_sha256 == hash_file(weights)
    assert extracted.model_name == "weights.pt"
    assert extracted.detector_settings["tracker"] == "bytetrack.yaml"
    assert [f.time_seconds for f in observations] == [0, .2, .2]
    assert observations[-1].source_frame_index == 1
    assert "image" not in target.read_text(encoding="utf-8")


def test_extraction_rejects_unreadable_video_without_leaving_trace(tmp_path):
    video = tmp_path / "input.mp4"; video.write_bytes(b"not a valid recording")
    class Unavailable:
        def __init__(self, *_): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self): return CaptureResult(CameraStatus.UNAVAILABLE)
    trace = tmp_path / "trace.jsonl"
    with pytest.raises(ValueError, match="no frames"):
        extract(video, trace, source_factory=Unavailable)
    assert not trace.exists()


def test_replay_determinism_reset_and_real_analyzers(monkeypatch):
    calls = 0
    original = BehaviourAnalyzer.update
    def observed(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)
    monkeypatch.setattr(BehaviourAnalyzer, "update", observed)
    first = replay(header(), frames(), session_id="replay-s")
    second = replay(header(), frames(), session_id="replay-s")
    assert calls == 2 * len(frames())
    assert [s.model_dump(exclude={"processing_fps"}) for s in first.scenes] == [
        s.model_dump(exclude={"processing_fps"}) for s in second.scenes]
    assert first.scenes[2].events[0].type == EventType.PERSON_APPEARED
    assert first.scenes[2].events[0].event_id == 1
    assert second.scenes[2].events[0].event_id == 1
    assert first.scenes[2].timestamp_utc == EPOCH + timedelta(seconds=.4)


def test_manifest_paths_splits_duplicate_and_missing(tmp_path):
    path = collection(tmp_path)
    manifest = load_manifest(path)
    assert manifest.scenarios[0].resolve(path, "trace") == (tmp_path / "trace.jsonl").resolve()
    assert manifest.scenarios[1].split == "validation"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["scenarios"][1]["id"] = "sample"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate scenario IDs"):
        load_manifest(path)
    data["scenarios"][1]["id"] = "held"
    data["scenarios"][1]["video"] = "missing.mp4"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="held video"):
        load_manifest(path)


def test_split_isolation_rejects_reused_trace(tmp_path):
    path = collection(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["scenarios"][1]["trace"] = "trace.jsonl"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="split leakage"):
        evaluate(path)


def test_annotations_validate_actor_and_time(tmp_path):
    with pytest.raises(ValidationError):
        ExpectedEvent(type="wave_detected")
    with pytest.raises(ValidationError):
        ExpectedEvent(type="wave_detected", start_seconds=2, end_seconds=1)
    with pytest.raises(ValidationError, match="anonymous"):
        Annotations(scenario_id="x", events=[ExpectedEvent(type="wave_detected", time_seconds=1, actor="Alice")])
    with pytest.raises(ValidationError):
        StateInterval(start_seconds=1, end_seconds=1.1, motion_state="approaching")
    target = tmp_path / "blank.json"
    template("demo", target)
    assert Annotations.model_validate_json(target.read_text(encoding="utf-8")).scenario_id == "demo"
    with pytest.raises(FileExistsError):
        template("demo", target)


def test_one_to_one_nearest_hand_zone_exclusion_and_zero_denominator():
    annotations = Annotations(scenario_id="s", events=[
        ExpectedEvent(type="wave_detected", time_seconds=1, tolerance_seconds=.2,
                      actor="person_a", hand_side="left"),
        ExpectedEvent(type="wave_detected", time_seconds=1.08, tolerance_seconds=.15,
                      actor="person_a", hand_side="left"),
        ExpectedEvent(type="zone_entered", start_seconds=2, end_seconds=2.4, zone_id="left"),
        ExpectedEvent(type="wave_detected", time_seconds=3, excluded=True),
    ])
    predicted = [prediction("wave_detected", 1.07, 1, side="left"),
                 prediction("wave_detected", 1.01, 2, side="right"),
                 prediction("wave_detected", 1.12, 3, side="left"),
                 prediction("zone_entered", 2.2, 4, zone="right"),
                 prediction("zone_entered", 2.3, 5, zone="left"),
                 prediction("wave_detected", 3, 6)]
    matched = match_events(annotations, predicted)
    assert {m.predicted_index for m in matched.matched} == {0, 2, 4}
    assert matched.false_positive == [1, 3]
    assert matched.false_negative == [] and matched.ignored_predictions == [5]
    assert matched.scenario_level_actors == ["person_a"]
    metrics = event_metrics(annotations, predicted, matched, 5)
    assert metrics["overall"]["tp"] == 3 and metrics["overall"]["fp"] == 2
    assert metrics["overall"]["median_absolute_error_seconds"] == pytest.approx(.07)
    assert metrics["overall"]["p95_absolute_error_seconds"] is None
    missed = match_events(Annotations(scenario_id="s", events=[
        ExpectedEvent(type="wave_detected", time_seconds=4, tolerance_seconds=.1)]), predicted)
    assert missed.false_negative == [0] and len(missed.false_positive) == len(predicted)
    assert ratios(0, 0, 0)["f1"] is None
    assert ratios(0, 0, 1)["f1"] == 0
    assert ratios(0, 1, 1)["f1"] == 0
    mapped = annotations.model_copy(update={"actor_tracks": {"person_a": 99}})
    assert len(match_events(mapped, predicted).matched) == 1


def test_explicit_event_scope_ignores_unannotated_classes():
    annotations = Annotations(scenario_id="s", scored_event_types=[EventType.WAVE_DETECTED],
        events=[ExpectedEvent(type="wave_detected", time_seconds=1)])
    predicted = [prediction("wave_detected", 1, 1), prediction("person_appeared", 1, 2)]
    matched = match_events(annotations, predicted)
    assert len(matched.matched) == 1 and matched.ignored_unscored == [1]
    assert event_metrics(annotations, predicted, matched, 2)["overall"]["f1"] == 1
    with pytest.raises(ValidationError, match="outside scored_event_types"):
        Annotations(scenario_id="s", scored_event_types=[EventType.WAVE_DETECTED],
                    events=[ExpectedEvent(type="person_appeared", time_seconds=1)])


def test_states_confusion_unknown_coverage_and_occlusion():
    data = frames()
    result = replay(header(), data, session_id="replay-s")
    annotation = Annotations(scenario_id="s", states=[
        StateInterval(start_seconds=0, end_seconds=.4, visible_count=0, occupancy_state="empty"),
        StateInterval(start_seconds=.4, end_seconds=.8, visible_count=1),
        StateInterval(start_seconds=1.2, end_seconds=1.6, brief_occlusion=True),
        StateInterval(start_seconds=.4, end_seconds=.8, actor="person_a", motion_state="stationary"),
    ])
    states = state_metrics(annotation, result)
    assert states["by_field"]["visible_count"]["accuracy"] == 1
    assert states["by_field"]["occupancy_state"]["unknown_coverage"] > 0
    assert states["by_field"]["occupancy_state"]["confusion"]
    assert states["skipped_unmapped"]["motion_state"] == 1
    disappeared = prediction("person_disappeared", 1.4, 9)
    assert stability_metrics(annotation, result, [disappeared])[
        "false_disappearances_during_brief_occlusion"] == 1


def test_runner_report_and_criteria_exit(tmp_path):
    path = collection(tmp_path)
    report = evaluate(path)
    assert set(report.groups) >= {"split:calibration", "split:validation", "tag:entry", "headline"}
    assert report.scenarios[0]["events"]["overall"]["tp"] == 1
    assert report.scenarios[0]["replay_runtime"]["frames_processed"] == len(frames())
    assert "Matched" not in markdown(report) or "Scenario evidence" in markdown(report)
    output = tmp_path / "reports" / "baseline"
    save_report(output, report, csv_table=True)
    assert all(output.with_suffix(s).is_file() for s in (".json", ".md", ".csv"))
    assert json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["manifest_sha256"] == hash_file(path)
    pass_criteria = tmp_path / "pass.json"
    pass_criteria.write_text('{"maximum_false_events_per_minute":1000}', encoding="utf-8")
    fail_criteria = tmp_path / "fail.json"
    fail_criteria.write_text('{"minimum_occupancy_accuracy":0.9}', encoding="utf-8")
    assert main(["evaluate", "--manifest", str(path), "--output", str(tmp_path / "pass"),
                 "--criteria", str(pass_criteria)]) == 0
    assert main(["evaluate", "--manifest", str(path), "--output", str(tmp_path / "fail"),
                 "--criteria", str(fail_criteria)]) == 3
    assert evaluate(path, criteria=Criteria(minimum_occupancy_accuracy=.9)).criteria["checks"][
        "minimum_occupancy_accuracy"]["reason"] == "metric unavailable"


def test_sweep_limit_split_ranking_and_preserved_defaults(tmp_path, monkeypatch):
    path = collection(tmp_path)
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(json.dumps({"parameters": {"behaviour.track_grace_seconds": [.3, .75]}}), encoding="utf-8")
    assert load_sweep(sweep_path).size == 2
    default = (Path(__file__).parents[1] / "pose.example.json").read_bytes()
    from reactive_vision.validation import sweep as module, runner as runner_module
    real = module.evaluate_scenario
    real_runner = runner_module.evaluate_scenario
    held = []
    def observe(manifest, scenario, **kwargs):
        if scenario.split == "validation":
            held.append(scenario.id)
        return real(manifest, scenario, **kwargs)
    monkeypatch.setattr(module, "evaluate_scenario", observe)
    def observe_runner(manifest, scenario, **kwargs):
        if scenario.split == "validation":
            held.append(scenario.id)
        return real_runner(manifest, scenario, **kwargs)
    monkeypatch.setattr(runner_module, "evaluate_scenario", observe_runner)
    output = tmp_path / "reports" / "calibrate"
    results = run_sweep(path, sweep_path, output)
    assert results["held_out_evaluated_once"] is True
    assert held == ["held"]  # held-out is replayed only after selection
    assert results["ranked"] == sorted(results["ranked"], key=lambda row: (
        row["objective"]["score"] is None, -(row["objective"]["score"] or 0),
        sum(len(values) for values in row["overrides"].values()), json.dumps(row["overrides"], sort_keys=True)))
    assert output.with_name(output.name + ".validation.json").is_file()
    assert output.with_name(output.name + ".recommended.json").is_file()
    assert (Path(__file__).parents[1] / "pose.example.json").read_bytes() == default
    large = tmp_path / "large.json"
    large.write_text(json.dumps({"parameters": {
        "behaviour.zone_debounce_seconds": list(range(8)),
        "behaviour.track_grace_seconds": list(range(8)),
        "pose.wave_min_amplitude_ratio": [.1, .2],
    }}), encoding="utf-8")
    with pytest.raises(ValueError, match="exceed limit"):
        load_sweep(large)
    with pytest.raises(ValidationError, match="unsupported sweep"):
        SweepDefinition(parameters={"unsupported": [1]})
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"parameters": {"pose.wave_min_reversals": [1]}}), encoding="utf-8")
    with pytest.raises(ValidationError):
        run_sweep(path, invalid, tmp_path / "reports" / "invalid")


def test_sweep_criteria_require_held_out_data(tmp_path):
    path = collection(tmp_path, validation=False)
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text('{"parameters":{"room.focus_switch_margin":[0.15]}}', encoding="utf-8")
    result = run_sweep(path, sweep_path, tmp_path / "reports" / "no_holdout",
                       criteria=Criteria(minimum_event_f1=.5))
    assert result["held_out_evaluated_once"] is False
    assert result["held_out_criteria"]["passed"] is False
