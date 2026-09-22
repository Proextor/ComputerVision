"""M7 offline validation commands. No command records or uploads footage."""

import argparse
from pathlib import Path
import sys

from pydantic import ValidationError

from .annotate import annotate, template
from .replay import configs, replay
from .report import save_report
from .runner import evaluate, load_annotations, load_manifest
from .schemas import Criteria
from .sweep import run_sweep
from .trace import extract, load_trace


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Offline validation of supplied recordings")
    sub = command.add_subparsers(dest="action", required=True)
    check = sub.add_parser("validate-manifest", help="check all manifest paths and IDs")
    check.add_argument("--manifest", type=Path, required=True)
    annotation = sub.add_parser("validate-annotations", help="validate one annotation sidecar")
    annotation.add_argument("--annotations", type=Path, required=True)
    annotation.add_argument("--scenario-id", required=True)
    source = sub.add_parser("extract", help="extract no-pixel trace from supplied video")
    source.add_argument("--video", type=Path, required=True)
    source.add_argument("--trace", type=Path, required=True)
    source.add_argument("--model", default="yolo26n-pose.pt")
    source.add_argument("--imgsz", type=int, default=640)
    source.add_argument("--confidence", type=float, default=0.25)
    source.add_argument("--joint-confidence", type=float, default=0.25)
    source.add_argument("--device", default="cpu")
    source.add_argument("--source-id")
    playback = sub.add_parser("replay", help="reprocess a trace without YOLO")
    playback.add_argument("--trace", type=Path, required=True)
    playback.add_argument("--output", type=Path, required=True)
    playback.add_argument("--session-id", default="offline-replay")
    playback.add_argument("--zones", type=Path)
    playback.add_argument("--pose-config", type=Path)
    playback.add_argument("--room-config", type=Path)
    playback.add_argument("--behaviour-config", type=Path)
    playback.add_argument("--real-time", action="store_true")
    edit = sub.add_parser("annotate", help="local video annotation GUI")
    edit.add_argument("--video", type=Path, required=True)
    edit.add_argument("--output", type=Path, required=True)
    edit.add_argument("--scenario-id")
    edit.add_argument("--zone-id")
    blank = sub.add_parser("template", help="create strict blank annotation sidecar")
    blank.add_argument("--scenario-id", required=True)
    blank.add_argument("--output", type=Path, required=True)
    assessment = sub.add_parser("evaluate", help="generate JSON/Markdown/CSV evidence")
    assessment.add_argument("--manifest", type=Path, required=True)
    assessment.add_argument("--output", type=Path, required=True)
    assessment.add_argument("--criteria", type=Path)
    assessment.add_argument("--csv", action="store_true")
    sweep = sub.add_parser("sweep", help="rank calibration thresholds and evaluate held-out once")
    sweep.add_argument("--manifest", type=Path, required=True)
    sweep.add_argument("--sweep", type=Path, required=True)
    sweep.add_argument("--output", type=Path, required=True)
    sweep.add_argument("--allow-large", action="store_true")
    sweep.add_argument("--criteria", type=Path)
    return command


def _criteria(path: Path | None) -> Criteria | None:
    return Criteria.model_validate_json(path.read_text(encoding="utf-8")) if path else None


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.action == "validate-manifest":
            manifest = load_manifest(args.manifest)
            print(f"{len(manifest.scenarios)} scenario(s); manifest and referenced files valid")
        elif args.action == "validate-annotations":
            annotations = load_annotations(args.annotations, args.scenario_id)
            print(f"{len(annotations.events)} event(s), {len(annotations.states)} state interval(s) valid")
        elif args.action == "extract":
            header = extract(args.video, args.trace, model=args.model,
                             imgsz=args.imgsz, confidence=args.confidence,
                             joint_confidence=args.joint_confidence, device=args.device,
                             source_id=args.source_id)
            print(f"trace written: {args.trace}; extraction: {header.extraction_runtime}")
        elif args.action == "replay":
            header, frames = load_trace(args.trace)
            zone, behaviour, pose, room = configs(args.zones, args.pose_config,
                                                   args.room_config, args.behaviour_config)
            result = replay(header, frames, session_id=args.session_id,
                            zones=zone, behaviour=behaviour, pose=pose, room=room,
                            real_time=args.real_time)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as handle:
                for scene in result.scenes:
                    handle.write(scene.model_dump_json(exclude_none=False) + "\n")
            print(f"{len(result.scenes)} scenes replayed; runtime: {result.runtime()}")
        elif args.action == "annotate":
            annotate(args.video, args.output, scenario_id=args.scenario_id, zone_id=args.zone_id)
        elif args.action == "template":
            template(args.scenario_id, args.output)
            print(f"annotation template written: {args.output}")
        elif args.action == "evaluate":
            report = evaluate(args.manifest, criteria=_criteria(args.criteria))
            save_report(args.output, report, csv_table=args.csv)
            print(f"evidence written: {args.output.with_suffix('.json')} and .md")
            if report.criteria is not None and not report.criteria["passed"]:
                print(f"criteria unmet: {report.criteria['checks']}", file=sys.stderr)
                return 3
        elif args.action == "sweep":
            summary = run_sweep(args.manifest, args.sweep, args.output,
                                allow_large=args.allow_large, criteria=_criteria(args.criteria))
            print(f"{summary['candidates_evaluated']} candidates; recommended overrides: "
                  f"{summary['recommended_overrides']}; held-out evaluated: "
                  f"{summary['held_out_evaluated_once']}")
            if summary["held_out_criteria"] is not None and not summary["held_out_criteria"]["passed"]:
                print(f"held-out criteria unmet: {summary['held_out_criteria']['checks']}", file=sys.stderr)
                return 3
        return 0
    except (ValueError, OSError, ValidationError, FileExistsError) as error:
        print(f"validation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
