# M7 validation, trace replay and calibration

This is an offline workflow for videos explicitly supplied by their owner. Extraction runs the existing Ultralytics pose model and ByteTrack once, then writes a **versioned JSONL no-pixel trace**. Replay feeds tracked observations into the same `TrackedFrameProcessor` as the CLI and service, using the video source times and a fixed UTC epoch. It resets all M2–M4 state for each scenario and does not sleep unless `replay --real-time` is selected. No threshold sweep calls YOLO again.

The repository contains only a tiny non-person synthetic diagnostic fixture. It demonstrates the commands and **does not establish model accuracy**. The example criteria are illustrative user targets, not scientific defaults.

## PowerShell commands

From the root of this workspace, set up Python 3.11 if needed:

```powershell
cd 'G:\web apps\ComputerVision'
python -m pip install --user uv
python -m uv python install 3.11
python -m uv venv --python 3.11 .venv
python -m uv pip install --python .venv\Scripts\python.exe -e ".[test]"
```

Check the supplied synthetic manifest and produce a diagnostic report:

```powershell
.\.venv\Scripts\python.exe -m reactive_vision.validation.cli validate-manifest `
  --manifest .\validation\manifest.example.json
.\.venv\Scripts\python.exe -m reactive_vision.validation.cli evaluate `
  --manifest .\validation\manifest.example.json `
  --output .\validation\reports\synthetic --csv
```

For a consented recording, create `validation/manifest.json` from the example and add your own scenario entries. Use split `calibration`, `validation` or `diagnostic` deliberately. Place the local video under `validation/data/`; paths in the manifest are relative to `validation/manifest.json`. A trace-only scenario omits `video`; a declared missing video, trace, annotation or config fails validation before a long run.
The same trace or source-video SHA-256 cannot appear in both calibration and validation. Keep held-out recordings independent from tuning material.

```powershell
# Run detection/tracking once; local weights avoid a model download.
.\.venv\Scripts\python.exe -m reactive_vision.validation.cli extract `
  --video .\validation\data\wave_01.mp4 `
  --trace .\validation\traces\wave_01.jsonl `
  --model .\yolo26n-pose.pt --device cpu

# Make and check an annotation template, or use the local video GUI.
.\.venv\Scripts\python.exe -m reactive_vision.validation.cli template `
  --scenario-id wave_01 --output .\validation\annotations\wave_01.json
.\.venv\Scripts\python.exe -m reactive_vision.validation.cli annotate `
  --video .\validation\data\wave_01.mp4 `
  --output .\validation\annotations\wave_01.json `
  --scenario-id wave_01 --zone-id interaction
.\.venv\Scripts\python.exe -m reactive_vision.validation.cli validate-annotations `
  --annotations .\validation\annotations\wave_01.json --scenario-id wave_01

# Inspect deterministic schema 4.0 JSONL without a model invocation.
.\.venv\Scripts\python.exe -m reactive_vision.validation.cli replay `
  --trace .\validation\traces\wave_01.jsonl `
  --output .\validation\reports\wave_01.scenes.jsonl `
  --zones .\zones.example.json `
  --pose-config .\pose.example.json `
  --room-config .\room.example.json

# Evaluate manifest scenarios and produce JSON, Markdown and CSV.
.\.venv\Scripts\python.exe -m reactive_vision.validation.cli evaluate `
  --manifest .\validation\manifest.json `
  --output .\validation\reports\baseline --csv

# Small explicit grid; calibration split selects, validation split evaluated once.
.\.venv\Scripts\python.exe -m reactive_vision.validation.cli sweep `
  --manifest .\validation\manifest.json `
  --sweep .\validation\sweep.example.json `
  --output .\validation\reports\sweep
```

`extract` accepts `--imgsz`, `--confidence`, `--joint-confidence`, `--device`, `--source-id` and `--model`. A model name that is not a local weights path may trigger Ultralytics' pretrained weight download; no training occurs. `replay` accepts `--behaviour-config` with the M2 `BehaviourConfig` field names as a JSON object; scenario manifests can select zones, pose and room config files independently. The vision service and the display run as before.

To apply optional target criteria, add `--criteria .\validation\criteria.example.json` to `evaluate` or `sweep`. `evaluate` returns 0 after generating evidence regardless of low measured accuracy when no criteria file is supplied; it returns 3 when a requested criterion fails or its metric is missing. It returns 2 for invalid input. A criterion is checked against the held-out validation split if present, otherwise against calibration and labelled accordingly. The sweep writes ranked `.json`/`.md`, selected `.calibration.json`/`.md`, held-out `.validation.json`/`.md` when available, and `.recommended.json`. It never modifies production configs. Copy recommended values into a new reviewed pose/room configuration or M2 settings intentionally.

## Contracts and scoring

`trace_version: "1.0"` metadata is the first JSONL line and contains a source identifier, model basename, local model SHA-256 when present, input-video SHA-256, dependency versions, detector settings (including ByteTrack) and extraction runtime. Subsequent `frame` lines include strictly increasing frame IDs, nondecreasing source-relative seconds, dimensions, source status, source video position and only current tracked people (`track_id`, normalized `bbox`, detection confidence and available keypoints/confidence). Null means unavailable. Invalid versions, nonfinite values, out-of-order time, extra image payloads and frames after terminal status are rejected. Traces contain body-coordinate data and still deserve care.

`manifest_version: "1.0"` identifies each unique scenario, split, trace, annotation, optional source video and configs. Missing referenced files are reported up front. The video hash is checked against trace metadata when the video is listed. `annotation_version: "1.0"` accepts point `time_seconds` or inclusive `[start_seconds,end_seconds]` event windows, tolerance, optional hand side and zone, anonymous actor label, exclusion flag and notes. Set optional `scored_event_types` to an explicit list (for example `["wave_detected"]`) if only those event classes were fully annotated; absent means score **all** predicted classes. Out-of-scope predictions are counted separately as ignored. State intervals use `[start_seconds,end_seconds)` and can specify visible count, occupancy, interaction, motion, activity, zone membership or a brief full-body occlusion. `actor_tracks` explicitly maps an anonymous actor such as `person_a` to a **temporary session track ID** only when the annotator has adequate evidence; otherwise event matching is scenario-level and person-state intervals are unscorable. Never infer a person’s identity from ByteTrack IDs.

Predicted events are matched one-to-one to compatible expectations in expected-time order. Each chooses the nearest unmatched candidate inside its point tolerance or expanded window, with deterministic tie-breaking. Hand side and zone must agree when annotated. Window timing offset is relative to the window midpoint. Excluded expectations remove matching unclaimed predictions from scoring. Duplicates remain false positives; unmatched expectations are false negatives. Reports show each timing offset and all unmatched timestamps.

Event metrics give TP/FP/FN, precision, recall, F1, false events per source-video minute, median signed and absolute error and P95 absolute error for at least five matches. Undefined denominators produce `null`, never an invented score. State accuracy compares the latest replay scene on a **fixed 0.1-second timeline** within annotated intervals. Reports include confusion counts, unknown coverage, observed transition delays and missed annotated transitions. Stability fields include repeated transitions, disappearance during labelled brief occlusion, genuine focus switches/minute, occupancy flicker, activity transitions, repeated starts and very short tracks. These are indicators, not formal identity-tracking accuracy. Extraction and replay runtime figures are separate; peak Python allocation measurements exclude native model memory. Very short clips can have an undefined false-event rate.

The sweep accepts only listed M2–M4 fields, 1–8 values per field and at most **64 grid combinations** unless `--allow-large` is explicit. It also compares the current baseline. Objective = macro F1 over annotated positive event types − 0.02 × false events/min − 0.1 × mean unknown coverage − 0.05 × median absolute timing error in seconds. If no positive event is annotated, objective remains unscorable; ranking prefers the existing baseline. Ties prefer fewer override values, then lexical order. The validation split never participates in candidate ranking and is replayed **once** for the selected candidate. A few recordings cannot demonstrate generalization.

## OpenCV annotation keyboard reference

Press `?` to show or hide this reference in the video window. The video remains local and the source file is never changed.

| Key | Action |
| --- | --- |
| Space | Play/pause |
| `J` / `L` (lowercase) | Seek backward/forward 0.25 seconds |
| Shift+`J` / Shift+`L` | Seek backward/forward 2 seconds |
| `,` / `.` | Select expected event type |
| `A` / `H` (lowercase) | Cycle anonymous actor (`person_a` etc.) / hand side |
| `E` | Mark point event at displayed source time |
| `B` then `N` | Mark event window start and end |
| `T` / `V` (lowercase) | Choose state field / expected value |
| `[` then `]` | Mark state interval start and end |
| `U` | Undo the last state interval, then the last event |
| `S` | Save validated sidecar; if output exists, press `Y` to confirm overwrite |
| `Q` / Escape | Close the annotation window |

The simple interface offers a few common actor labels and state values; edit the JSON sidecar for additional anonymous actors, larger visible counts, other zone combinations, exclusions or notes, then run `validate-annotations`. Annotations are never automatically saved on exit.

## Consent, retention and repository handling

Obtain consent from each recorded participant. Record short, purpose-specific takes, store videos and derived traces in a controlled local folder, and agree on a retention period before filming. `.gitignore` excludes `validation/data/`, `validation/traces/`, `validation/reports/`, `validation/annotations/` and common validation video formats. Keep only synthetic examples and templates in the repository. No command uploads anything. Avoid personal names and avoid sharing body-coordinate traces as if they were anonymous.

To delete local material after your agreed retention period, review the paths first and then delete the specific directories you created, for example `Remove-Item -LiteralPath '.\validation\data\wave_01.mp4'`, `Remove-Item -LiteralPath '.\validation\traces\wave_01.jsonl'`, and the corresponding annotation/report files. This tool never deletes source recordings automatically.

## Recording checklist

Use a fixed camera and log its position, resolution, fps, lighting, zone polygons, model weights and configuration hashes. Capture multiple takes with consenting people, varied clothing, positions, speeds and lighting. Label point or window events and expected states immediately after each take. Partition by recording/person/location before tuning, keeping the validation split untouched. Negative and stress cases matter as much as positives.

| # | Scenario | Role | Annotation cues |
| ---: | --- | --- | --- |
| 1 | Empty room | Negative | Stable empty and visible count 0 |
| 2 | One person enters and remains still | Positive | Appearance, occupancy, stationary |
| 3 | Walk toward camera | Positive | Approach event, motion and room interaction |
| 4 | Walk away | Positive | Departure event and motion |
| 5 | Cross a configured zone boundary | Positive | Entry/exit and zone interval |
| 6 | Stand near boundary with small jitter | Confuser | No repeated zone transitions |
| 7 | Left hand raised and held | Positive | Left start/end, held state |
| 8 | Right hand raised and held | Positive | Right start/end, held state |
| 9 | Both hands raised | Positive | Both-hand start/end |
| 10 | Raised hand held still | Confuser | No wave |
| 11 | One horizontal arm movement | Confuser | No repeated wave |
| 12 | Repeated side-to-side wave | Positive | Wave onset and hand side |
| 13 | Walk sideways without waving | Confuser | No wave from body translation |
| 14 | Brief full-body occlusion | Stress | Brief occlusion interval, no false disappearance |
| 15 | Brief wrist occlusion | Stress | Held hand and no false end |
| 16 | Leave frame | Positive | Disappearance and empty debounce |
| 17 | Two people enter | Stress | Two visible, group occupancy |
| 18 | Two people cross paths | Stress | ID switches/fragmentation indicators |
| 19 | One person signals while another is closer | Stress | Focus changes and possible occlusion |
| 20 | Lighting and partial cropping variation | Stress | Pose coverage, unknown states and missed signals |

No physical-camera validation or measured real-world accuracy is supplied with this repository. Run `python -m pytest -q` from the root and `npm run test`, `npm run lint`, `npm run typecheck`, `npm run build` in `display/` after changes.
