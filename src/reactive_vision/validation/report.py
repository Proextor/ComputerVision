"""Portable JSON, Markdown and optional CSV evidence reports."""

import csv
from pathlib import Path

from .schemas import ValidationReport


def _number(value: object) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def markdown(report: ValidationReport) -> str:
    lines = ["# Reactive Vision validation evidence", "",
             f"Run (UTC): {report.run_timestamp_utc}  ",
             f"Git commit: {report.git_commit or 'unavailable'}  ",
             f"Manifest SHA-256: `{report.manifest_sha256}`", "",
             "Metrics are evidence from the supplied annotations, not a general accuracy claim.", "",
             "## Split and tag summary", "",
             "| Group | Scenarios | Duration (s) | TP | FP | FN | Precision | Recall | F1 | False/min | Occupancy accuracy |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, group in report.groups.items():
        events = group["events"]["overall"]
        occupancy = group["states"]["by_field"].get("occupancy_state", {})
        values = [name, group["scenario_count"], group["duration_seconds"],
                  events["tp"], events["fp"], events["fn"], events["precision"],
                  events["recall"], events["f1"], events["false_events_per_minute"],
                  occupancy.get("accuracy")]
        lines.append("| " + " | ".join(_number(v) for v in values) + " |")
    if report.criteria is not None:
        lines += ["", "## User criteria", "",
                  f"Target group: {report.criteria['target_group'] or 'unavailable'}; "
                  f"all satisfied: {report.criteria['passed']}", "",
                  "| Criterion | Target | Observed | Satisfied |",
                  "| --- | ---: | ---: | --- |"]
        for name, result in report.criteria["checks"].items():
            lines.append(f"| {name} | {_number(result['target'])} | "
                         f"{_number(result['observed'])} | {result['passed']} |")
    headline = report.groups["headline"]
    lines += ["", "## Per-event metrics (headline; diagnostic split excluded)", "",
              "| Type | TP | FP | FN | Precision | Recall | F1 | False/min | Median absolute error (s) | P95 error (s) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind, row in headline["events"]["by_type"].items():
        values = [kind, row["tp"], row["fp"], row["fn"], row["precision"],
                  row["recall"], row["f1"], row["false_events_per_minute"],
                  row["median_absolute_error_seconds"], row["p95_absolute_error_seconds"]]
        lines.append("| " + " | ".join(_number(v) for v in values) + " |")
    for split in ("calibration", "validation"):
        group = report.groups.get(f"split:{split}")
        if group is None:
            continue
        lines += ["", f"### {split} event detail", "",
                  "| Type | TP | FP | FN | F1 | False/min | Median absolute error (s) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for kind, row in group["events"]["by_type"].items():
            values = [kind, row["tp"], row["fp"], row["fn"], row["f1"],
                      row["false_events_per_minute"], row["median_absolute_error_seconds"]]
            lines.append("| " + " | ".join(_number(v) for v in values) + " |")
    lines += ["", "## State confusion matrices (headline; 10 Hz source-time samples)", ""]
    for field, state in headline["states"]["by_field"].items():
        lines += [f"### {field}", "",
                  f"Accuracy: {_number(state['accuracy'])}; unknown coverage: "
                  f"{_number(state['unknown_coverage'])}; samples: {state['samples']}; "
                  f"median transition delay (s): {_number(state['median_transition_delay_seconds'])}; "
                  f"missed transitions: {state['transitions_missed']}/{state['transitions_annotated']}", "",
                  "| Expected | Predicted | Samples |", "| --- | --- | ---: |"]
        for expected, predictions in state["confusion"].items():
            for predicted, count in predictions.items():
                lines.append(f"| {expected} | {predicted} | {count} |")
        lines.append("")
    lines += ["## Scenario evidence", ""]
    worst = sorted(report.scenarios,
                   key=lambda s: (s["events"]["overall"]["f1"] is not None,
                                  s["events"]["overall"]["f1"] if s["events"]["overall"]["f1"] is not None else 0,
                                  s["id"]))
    lines += ["Worst or unscorable scenarios first: " + ", ".join(s["id"] for s in worst[:5]), ""]
    for scenario in report.scenarios:
        events = scenario["events"]["overall"]
        lines += [f"### {scenario['id']} ({scenario['split']})", "",
                  scenario["description"], "",
                  f"Tags: {', '.join(scenario['tags']) or 'none'}; F1: {_number(events['f1'])}; "
                  f"TP/FP/FN: {events['tp']}/{events['fp']}/{events['fn']}; "
                  f"median timing offset (s): {_number(events['median_offset_seconds'])}", "",
                  f"Runtime: {scenario['replay_runtime']['frames_processed']} frames, "
                  f"{_number(scenario['replay_runtime']['video_duration_seconds'])} s of source, "
                  f"mean replay {_number(scenario['replay_runtime']['mean_replay_seconds'])} s/frame, "
                  f"P95 {_number(scenario['replay_runtime']['p95_replay_seconds'])} s/frame, "
                  f"effective {_number(scenario['replay_runtime']['effective_replay_fps'])} FPS, "
                  f"real-time factor {_number(scenario['replay_runtime']['replay_real_time_factor'])}, "
                  f"peak replay memory {_number(scenario['replay_runtime']['peak_replay_bytes'])} bytes; "
                  f"trace file {scenario['replay_runtime']['trace_file_bytes']} bytes, "
                  f"peak trace load {_number(scenario['replay_runtime']['peak_trace_load_bytes'])} bytes", "",
                  f"Extraction model: {scenario['extraction']['model']}; SHA-256: "
                  f"{scenario['extraction']['model_sha256'] or 'unavailable'}; "
                  f"versions: {scenario['extraction']['dependencies']}; "
                  f"settings: {scenario['extraction']['settings']}; "
                  f"extraction runtime: {scenario['extraction']['runtime']}", "",
                  f"Input/configuration hashes: {scenario['input_hashes']}", "",
                  f"Stability: {scenario['stability']}", "",
                  "| Result | Type | Expected (s) | Predicted (s) | Offset (s) |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for match in scenario["comparison"]["matched"]:
            lines.append(f"| match | {match['type']} | {_number(match['expected_seconds'])} | "
                         f"{_number(match['predicted_seconds'])} | {_number(match['offset_seconds'])} |")
        for item in scenario["comparison"]["false_positive"]:
            lines.append(f"| false positive | {item['type']} | — | {_number(item['time_seconds'])} | — |")
        for item in scenario["comparison"]["false_negative"]:
            lines.append(f"| missed expectation | {item['type']} | {_number(item['time_seconds'])} | — | — |")
        lines += ["", f"Scored event types: {scenario['comparison']['scored_event_types']}; "
                  f"unscored predictions ignored: {scenario['comparison']['ignored_unscored_predictions']}. "
                  f"Unmapped actor labels scored at scenario level: "
                  f"{scenario['comparison']['actors_scored_at_scenario_level']}; "
                  f"total ignored predictions: {scenario['comparison']['ignored_predictions']}", ""]
    lines += ["## Limits", ""] + [f"- {item}" for item in report.limitations]
    annotated = sum(s["events"]["overall"]["tp"] + s["events"]["overall"]["fn"]
                    for s in report.scenarios if s["split"] == "validation")
    if annotated < 20 or report.groups.get("split:validation", {}).get("scenario_count", 0) < 3:
        lines += ["", "Held-out event samples are too few for a reliable conclusion; "
                  "record and annotate more independent validation scenarios."]
    return "\n".join(lines) + "\n"


def save_report(prefix: Path, report: ValidationReport, *, csv_table: bool = False) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    Path(f"{prefix}.md").write_text(markdown(report), encoding="utf-8")
    if csv_table:
        with Path(f"{prefix}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "split", "tags", "tp", "fp", "fn", "f1", "occupancy_accuracy"])
            for scenario in report.scenarios:
                row = scenario["events"]["overall"]
                writer.writerow([scenario["id"], scenario["split"], ",".join(scenario["tags"]),
                                 row["tp"], row["fp"], row["fn"], row["f1"],
                                 scenario["states"]["by_field"].get("occupancy_state", {}).get("accuracy")])
