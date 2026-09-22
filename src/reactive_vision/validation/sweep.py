"""Small, explicit threshold grid; held-out recordings never select a candidate."""

from itertools import product
import json
from pathlib import Path

from pydantic import model_validator

from .runner import aggregate, ensure_split_isolation, evaluate, evaluate_scenario, load_manifest
from .schemas import Criteria, StrictModel
from .report import save_report
from .replay import configs, with_overrides

MAX_COMBINATIONS = 64
ALLOWED = {
    "behaviour.zone_debounce_seconds", "behaviour.track_grace_seconds",
    "behaviour.motion_window_seconds", "behaviour.scale_threshold",
    "pose.gesture_activation_seconds", "pose.gesture_release_seconds",
    "pose.raised_margin_ratio", "pose.wave_min_amplitude_ratio",
    "pose.wave_min_reversals", "pose.wave_window_seconds",
    "pose.activity_thresholds", "room.activity_thresholds",
    "room.occupied_debounce_seconds", "room.empty_debounce_seconds",
    "room.focus_switch_margin",
}


class SweepDefinition(StrictModel):
    sweep_version: str = "1.0"
    parameters: dict[str, list[float | int | tuple[float, float, float]]]

    @model_validator(mode="after")
    def bounded(self):
        if self.sweep_version != "1.0":
            raise ValueError("unsupported sweep version")
        if not self.parameters:
            raise ValueError("sweep needs selected parameters")
        for name, values in self.parameters.items():
            if name not in ALLOWED:
                raise ValueError(f"unsupported sweep parameter: {name}")
            if not 1 <= len(values) <= 8:
                raise ValueError(f"{name} requires 1–8 candidate values")
            if len({json.dumps(value) for value in values}) != len(values):
                raise ValueError(f"{name} has duplicate candidates")
            if name.endswith("wave_min_reversals") and any(type(value) is not int for value in values):
                raise ValueError("wave reversal counts must be integers")
            if "activity_thresholds" in name and any(not isinstance(value, tuple) or len(value) != 3 for value in values):
                raise ValueError("activity thresholds must be triples")
        return self

    @property
    def size(self) -> int:
        count = 1
        for values in self.parameters.values():
            count *= len(values)
        return count


def load_sweep(path: Path, allow_large: bool = False) -> SweepDefinition:
    try:
        definition = SweepDefinition.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid sweep definition {path.name}: {error}") from error
    if definition.size > MAX_COMBINATIONS and not allow_large:
        raise ValueError(f"{definition.size} combinations exceed limit {MAX_COMBINATIONS}; use --allow-large explicitly")
    return definition


def objective(group: dict) -> dict:
    rows = [row for row in group["events"]["by_type"].values() if row["tp"] + row["fn"] > 0]
    macro = sum(row["f1"] or 0 for row in rows) / len(rows) if rows else None
    fp_rate = group["events"]["overall"]["false_events_per_minute"]
    covered = [row["unknown_coverage"] for row in group["states"]["by_field"].values()
               if row["unknown_coverage"] is not None]
    unknown = sum(covered) / len(covered) if covered else None
    latency = group["events"]["overall"]["median_absolute_error_seconds"]
    # No annotated positive events means F1 is unmeasurable; never invent a score.
    score = (macro - 0.02 * (fp_rate or 0) - 0.1 * (unknown or 0) - 0.05 * (latency or 0)
             if macro is not None else None)
    return {"score": score, "macro_event_f1": macro,
            "false_events_per_minute": fp_rate, "mean_unknown_coverage": unknown,
            "median_absolute_latency_seconds": latency}


def overrides_for(names: list[str], values: tuple) -> dict:
    result: dict[str, dict] = {}
    for name, value in zip(names, values):
        component, field = name.split(".", 1)
        result.setdefault(component, {})[field] = value
    return result


def run_sweep(manifest_path: Path, sweep_path: Path, output: Path, *,
              allow_large: bool = False, criteria: Criteria | None = None) -> dict:
    manifest = load_manifest(manifest_path)
    ensure_split_isolation(manifest_path, manifest)
    calibration = [s for s in manifest.scenarios if s.split == "calibration"]
    validation = [s for s in manifest.scenarios if s.split == "validation"]
    if not calibration:
        raise ValueError("sweep requires calibration scenarios")
    definition = load_sweep(sweep_path, allow_large)
    names = sorted(definition.parameters)
    candidates = [{}] + [overrides_for(names, values)
                         for values in product(*(definition.parameters[name] for name in names))]
    # Baseline may already be present in the grid; evaluate it once.
    candidates = list({json.dumps(value, sort_keys=True): value for value in candidates}.values())
    # Reject invalid candidate combinations before reading and replaying long traces.
    for scenario in calibration:
        baseline = configs(scenario.resolve(manifest_path, "zones"),
                           scenario.resolve(manifest_path, "pose_config"),
                           scenario.resolve(manifest_path, "room_config"))
        for candidate in candidates:
            with_overrides(baseline, candidate)
    ranked: list[dict] = []
    for candidate in candidates:
        results = [evaluate_scenario(manifest_path, scenario, overrides=candidate)
                   for scenario in calibration]
        group = aggregate(results)
        ranked.append({"overrides": candidate, "objective": objective(group),
                       "calibration": group})
    ranked.sort(key=lambda row: (
        row["objective"]["score"] is None,
        -(row["objective"]["score"] or 0),
        sum(len(values) for values in row["overrides"].values()),
        json.dumps(row["overrides"], sort_keys=True),
    ))
    chosen = ranked[0]["overrides"]
    chosen_calibration = evaluate(manifest_path,
                                  scenario_ids={s.id for s in calibration}, overrides=chosen)
    held_out = (evaluate(manifest_path, scenario_ids={s.id for s in validation},
                         overrides=chosen, criteria=criteria) if validation else None)
    unavailable_criteria = ({"target_group": "split:validation", "passed": False,
                             "checks": {name: {"target": target, "observed": None,
                                               "passed": False, "reason": "no held-out validation scenarios"}
                                        for name, target in criteria.model_dump(exclude_none=True).items()}}
                            if criteria is not None and not validation else None)
    summary = {"sweep_version": "1.0", "selection_split": "calibration",
               "maximum_default_combinations": MAX_COMBINATIONS,
               "grid_combinations": definition.size,
               "candidates_evaluated": len(ranked), "ranked": ranked,
               "recommended_overrides": chosen,
               "held_out_evaluated_once": bool(validation),
               "held_out_validation": held_out.groups.get("split:validation") if held_out else None,
               "held_out_criteria": held_out.criteria if held_out else unavailable_criteria,
               "warning": "No held-out validation scenarios were supplied; generalization is untested."
               if not validation else "Held-out results describe only these supplied scenarios."}
    output.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{output}.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    Path(f"{output}.recommended.json").write_text(
        json.dumps({"selected_on": "calibration", "overrides": chosen}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    save_report(output.with_name(output.name + ".calibration"), chosen_calibration)
    if held_out:
        save_report(output.with_name(output.name + ".validation"), held_out)
    rows = ["# Bounded calibration sweep", "",
            f"Grid: {definition.size} combinations plus default baseline; selection: calibration only.", "",
            "| Rank | Overrides | Objective | Macro F1 | False/min | Unknown | Median latency |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for index, entry in enumerate(ranked, 1):
        score = entry["objective"]
        rows.append(f"| {index} | `{json.dumps(entry['overrides'], sort_keys=True)}` | "
                    f"{score['score']} | {score['macro_event_f1']} | "
                    f"{score['false_events_per_minute']} | {score['mean_unknown_coverage']} | "
                    f"{score['median_absolute_latency_seconds']} |")
    rows += ["", f"Recommended overrides: `{json.dumps(chosen, sort_keys=True)}`", "",
             f"Held-out validation evaluated once: {bool(validation)}. {summary['warning']}", "",
             "Objective = macro event F1 − 0.02 × false events/min − 0.1 × unknown coverage "
             "− 0.05 × median absolute timing error (seconds). Missing macro F1 remains unscorable. "
             "Ties prefer fewer changes, then lexical parameter order. Settings are not applied automatically."]
    Path(f"{output}.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return summary
