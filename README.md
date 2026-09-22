# Reactive Vision — M9 live camera overlay

This Python 3.11 application reads a webcam, local video or an explicitly connected browser camera. It uses a pretrained Ultralytics YOLO pose model with **ByteTrack** (`tracker="bytetrack.yaml"`, `persist=True`), then adds short-lived spatial analysis, movement intensity, rule-based hand gestures and a stabilized room summary. The original CLI prints JSONL scenes and shows an OpenCV preview. M5 serves scenes over HTTP/WebSocket; M7 provides [offline validation](validation/README.md). **M9 adds authenticated browser-camera ingestion and a compact original SVG reaction overlay over full-screen local video**, with optional same-origin static serving and TLS. See the [display setup and phone guide](display/README.md) and [M9 validation record](display/M9_VALIDATION.md). Verification remains demo-only. There is no automatic recording, database, face or identity recognition, emotion inference, or hardware output.

## Windows PowerShell setup and run

From the project directory:

```powershell
cd 'G:\web apps\ComputerVision'
python -m pip install --user uv
python -m uv python install 3.11
python -m uv venv --python 3.11 .venv
python -m uv pip install --python .venv\Scripts\python.exe -e ".[test]"
```

## Display setup and demo mode

The browser display is a separate Vite project. Its visual-only demo uses only
synthetic fixtures: it does not request a camera or connect to the vision
service.

```powershell
cd .\display
npm ci
npm run dev
```

Open `http://localhost:3000/?demo=1`. For the explicitly labeled local-camera
demo (no frame upload), open `http://localhost:3000/?demo=1&camera=1` and press
**Start camera**. See [display/.env.example](display/.env.example) for public
URL placeholders and [display/README.md](display/README.md) for live-camera
setup; never place an ingest token in a Vite environment file.

Run a webcam **without** configured zones:

```powershell
.\.venv\Scripts\python.exe -m reactive_vision.main --webcam 0
```

Run the same webcam with the supplied polygon zone:

```powershell
.\.venv\Scripts\python.exe -m reactive_vision.main --webcam 0 --zones .\zones.example.json
```

Run with zone and pose settings together:

```powershell
.\.venv\Scripts\python.exe -m reactive_vision.main --webcam 0 --zones .\zones.example.json --pose-config .\pose.example.json
```

Run M4 with the optional room settings and preview:

```powershell
.\.venv\Scripts\python.exe -m reactive_vision.main --webcam 0 --zones .\zones.example.json --pose-config .\pose.example.json --room-config .\room.example.json
```

Run the M5 service instead of the preview CLI when another program needs live data:

```powershell
.\.venv\Scripts\python.exe -m reactive_vision.server `
  --webcam 0 `
  --zones .\zones.example.json `
  --pose-config .\pose.example.json `
  --room-config .\room.example.json `
  --cors-origin http://localhost:3000
```

The minimal service command is `.\.venv\Scripts\python.exe -m reactive_vision.server --webcam 0`. It binds to `127.0.0.1:8000`, runs headless, and publishes up to 5 snapshots per second. The CLI and service each own their own camera session; run one at a time.

Run a local file without a preview, or capture structured output only:

```powershell
.\.venv\Scripts\python.exe -m reactive_vision.main --video "C:\path\to\clip.mp4" --zones .\zones.example.json --headless
.\.venv\Scripts\python.exe -m reactive_vision.main --webcam 0 --zones .\zones.example.json --headless 1> scenes.jsonl
```

Diagnostics go to stderr. Press **Q** or **Escape** in the preview, or Ctrl+C in the terminal, to release the capture. `--width` and `--height` request webcam resolution; the frame's actual size is reported. `--imgsz`, `--confidence`, `--joint-confidence`, `--device`, `--model`, and `--rate` retain their M1 meanings. CPU is the default; `--device 0` requests the first CUDA GPU if PyTorch supports it. For offline use, supply a local pose weights file, for example `--model ".\weights\yolo26n-pose.pt"`. Otherwise Ultralytics downloads the lightweight pretrained `yolo26n-pose.pt` on first use. No training occurs. Existing M1 and M2 run commands remain valid; the pose config file is optional.

## M5 live data service

The CLI and server use the same `VisionSession` for synchronous capture, YOLO/ByteTrack inference, M2–M4 analysis and buffered events. The server runs that blocking session in one dedicated worker thread; FastAPI's event loop handles requests separately. Each publication is an immutable JSON copy of a complete schema 4.0 snapshot. A thread-safe broker tracks a service-wide publication sequence and monotonic receipt time. Every WebSocket client gets at most one pending snapshot; newer publications replace older pending data. There is no application frame queue and no second camera owner.

| Endpoint | Response |
| --- | --- |
| `GET /health` | Always HTTP 200 while the web service responds. Reports `starting`, `ready`, `stale`, `unavailable`, `eof`, `error` or `stopping`, worker state, session, source/camera status, last frame and publication, age, subscriber count and a sanitized worker error summary. |
| `GET /ready` | HTTP 200 only for a fresh `ok` scene; otherwise 503. Both responses contain the same typed readiness body. |
| `GET /api/v1/scene/latest` | Most recently published `SceneSnapshot` 4.0. Before the first publication, HTTP 503 with a typed `snapshot_not_available` error. An unavailable or EOF snapshot remains retrievable as meaningful source state. |
| `WS /api/v1/scene/stream` | Latest snapshot immediately on connection, then new snapshots or idle heartbeats. |

Check the running service from another PowerShell terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
Invoke-RestMethod http://127.0.0.1:8000/api/v1/scene/latest
```

WebSocket messages have distinct `type` values. The publication sequence increases only for snapshots and is independent of `frame_id`, `session_id` and event IDs:

```json
{"type":"snapshot","publication_sequence":42,"published_at":"2026-09-20T12:00:00Z","snapshot":{"schema_version":"4.0"}}
{"type":"heartbeat","publication_sequence":42,"service_status":"stale","last_snapshot_age_seconds":3.2}
```

The first line shows the envelope shape; real `snapshot` messages contain the **complete** schema 4.0 scene. A heartbeat never contains a scene. The idle heartbeat interval defaults to **1 s**. Reconnecting clients receive the current latest snapshot, with no historical replay. Socket input is ignored in M5; there are no control commands.

JavaScript client for a future frontend served from the explicitly allowed `http://localhost:3000` origin:

```javascript
const socket = new WebSocket("ws://127.0.0.1:8000/api/v1/scene/stream");
socket.onmessage = ({ data }) => {
  const message = JSON.parse(data);
  if (message.type === "snapshot") {
    console.log(message.publication_sequence, message.snapshot.room_state);
  } else if (message.type === "heartbeat") {
    console.log("service", message.service_status, message.last_snapshot_age_seconds);
  }
};
```

Python client (`websockets` is installed by the setup command):

```python
import asyncio
import json
from websockets.asyncio.client import connect

async def main():
    async with connect("ws://127.0.0.1:8000/api/v1/scene/stream") as socket:
        async for text in socket:
            message = json.loads(text)
            if message["type"] == "snapshot":
                print(message["publication_sequence"], message["snapshot"]["room_state"])
            else:
                print("heartbeat", message["service_status"])

asyncio.run(main())
```

To allow a browser frontend, pass its exact origin with repeatable `--cors-origin` options, for example `--cors-origin http://localhost:3000 --cors-origin http://127.0.0.1:3000`. HTTP CORS and WebSocket `Origin` checks use that list. Unlisted browser origins are rejected for the socket; non-browser clients without an `Origin` header are accepted unless `--strict-origin` is set. Wildcard CORS and credentialed CORS are disabled.

Defaults: bind **127.0.0.1:8000**; freshness timeout **3 s**; heartbeat **1 s**; shutdown wait **2 s**; webcam retry **2 s**. `--rate` remains **5 Hz**. The corresponding flags are `--host`, `--port`, `--freshness-timeout`, `--heartbeat-interval`, `--shutdown-timeout`, and `--webcam-retry`. `--log-level` sets server logging. `--debug-jsonl` explicitly mirrors published snapshots to stdout; otherwise the server does not print scene JSONL. A bind to `0.0.0.0` logs a warning.

Health uses monotonic time since service start or the last publication. A slow model load, slow inference or blocked camera read never blocks HTTP or WebSocket handling; after the freshness timeout, health reports `stale` and `/ready` returns 503. The service does not invent a replacement scene. An unavailable webcam publishes `unavailable`, waits the retry interval, and opens a fresh session on recovery. Model initialization failures put the worker in `error` without an uncontrolled retry loop. Video EOF publishes `eof` and stops; `--loop-video` explicitly restarts the file with a new session. Each restart resets ByteTrack, gesture, room and event state. Shutdown signals the worker, attempts capture release and waits at most the configured timeout. A misbehaving native driver may not be interruptible immediately; the daemon worker can outlive that bound until the read returns.

M5 has **no authentication**. Keep the default loopback binding; do not expose it directly to an untrusted network. Frames and video are processed in memory and are not stored by default, while snapshots contain people locations and pose-derived observations. Apply appropriate access controls outside this milestone before sharing the service beyond a trusted local setup.

## Zone configuration

[zones.example.json](zones.example.json) contains one trapezoid suitable as a starting point for a fixed, front-facing webcam. Adjust its corners to your camera view. Add multiple objects to `zones` for overlapping or separate areas. Every zone needs a unique ID, nonempty label, and at least three distinct vertices forming a simple polygon. A zone with `"role":"interaction"` contributes an M4 approaching cue after M2 confirms the person is inside it; other roles remain descriptive. Invalid JSON, IDs, coordinates outside `[0,1]`, zero-area polygons, and self-intersecting polygons fail at startup with an error.

The image coordinate origin `(0,0)` is at the **top left**. x increases right, y increases down, and `(1,1)` is the bottom right. Bounding boxes use normalized `x1,y1,x2,y2`; the `anchor_point` is the normalized **bottom centre** of each box: `((x1+x2)/2, y2)`. Zone membership uses that anchor, including polygon edges. A person can occupy more than one overlapping zone.

`zones` on each person lists zones containing the anchor **on this frame**. `zone_occupancy` counts currently visible people whose anchors fall in each zone, including untracked detections. `confirmed_zones` lists memberships that passed the enter/exit debounce. `dwell_seconds` reports elapsed time since each confirmed entry. A brief occlusion retains confirmed membership and dwell time, but it does not count as an observed zone exit.

## Spatial behaviour and events

`bbox_area_ratio` is normalized box width times height. `proximity_score` is simply normalized **box height**, from 0 to 1. It is a non-metric visual proxy; it is **not** metres or physical distance. A larger box can indicate approach to a fixed camera, but crouching, camera movement, partial occlusion, and truncation at the frame edge can also change it. `approaching` and `departing` are camera-relative estimates.

For temporary tracked IDs, the analyzer retains up to 64 samples and 2 seconds of timestamped, smoothed anchor/scale history. A three-observation median filter removes isolated spikes before exponential smoothing. It calculates `velocity_x`, `velocity_y` (normalized image displacement per second) and `speed` from smoothed anchors. Motion stays `unknown` until at least four samples span 0.6 seconds. Over a rolling 1.2-second window, consistent proxy increase or decrease drives `approaching` or `departing`; small position and scale changes drive `stationary`. An additional 0.25-second motion debounce prevents a one-frame classification from changing the reported state. The `trajectory` field contains up to 16 recent smoothed anchor points with UTC timestamps. People without an assigned track ID still receive current-frame spatial fields and `unknown` motion, but no historical claims or identity events.

Events have a session-local integer `event_id`, type, UTC timestamp, session ID, temporary track ID, optional zone ID, and small metadata object. Supported types are `person_appeared`, `person_disappeared`, `zone_entered`, `zone_exited`, `approach_started`, `departure_started`, and `became_stationary`. Transitions fire once when confirmed. A track that is unseen for more than the grace period produces one `person_disappeared` event; this means tracking was lost, **not** that the person physically left. It does not generate a zone exit. Events from processed frames are buffered until the next emitted JSON line, including the final EOF/error line. `visible_count` always counts only current-frame detections.

## Pose quality, activity, and gestures

Each visible person has a nested `pose_analysis` object with `quality`, `activity`, and `gestures`. M3 uses structured COCO keypoints and temporary ByteTrack IDs; it does not call Ultralytics directly or infer identity or emotion. For an untracked detection, pose quality is still measured, while history-based activity and gestures remain `unknown`.

`quality` reports `usable_keypoint_count`, `usable_keypoint_ratio` out of 17 COCO joints, mean `pose_confidence` of usable joints (or `null`), and `pose_quality`: `good` for at least 12 usable joints, `partial` for 6–11, `poor` for fewer, or `unavailable` when no keypoints were supplied. A usable joint needs normalized coordinates and confidence at or above `min_keypoint_confidence`. Missing or weak shoulders/wrists make the corresponding gesture `unknown`; coordinates are never fabricated. COCO left/right labels are anatomical and may appear reversed in a mirrored preview.

`gestures` contains `left_hand_raised`, `right_hand_raised`, `both_hands_raised`, `left_wave`, and `right_wave`. Values are `active`, `inactive`, or `unknown`. A wrist must be sufficiently above its shoulder relative to box height; a reliable elbow supports raised-hand evidence. Activation and observed lowering are debounced. Missing wrist evidence holds an already active hand briefly, then changes it to `unknown` without a false end event. Both hands become active only when both sides are confirmed active.

A wave requires a hand around or above shoulder level, enough reliable wrist observations, normalized horizontal travel relative to its shoulder, and at least two direction reversals within a two-second window. The wrist path uses median and exponential smoothing. A detected wave is shown briefly, and cooldown plus a continuous-wave latch prevent repeat events. A static raised hand or simple sideways walk is not a wave.

`activity` reports `activity_score` from 0–1 (or `null`), `activity_level` (`unknown`, `still`, `low`, `moderate`, `high`), and `contributing_joint_count`. The score combines **72% body-relative joint movement**, **18% anchor movement**, and **10% box-scale change**, all using elapsed seconds; at least six reliable body joints shared across two observations are required. Joint displacement is normalized by visible body height and calculated around the torso centre. An EWMA smooths the score, and level changes are debounced. Activity describes observed movement, not happiness, excitement, anger, or any other emotion. A person can remain spatially `stationary` in M2 while waving and increasing M3 activity.

New buffered events are `hand_raise_started`, `hand_raise_ended`, `both_hands_raised_started`, `both_hands_raised_ended`, `wave_detected`, and `activity_level_changed`. They use the existing M2 session event sequence and JSONL buffer. Hand and wave events include the side and relevant confidence or amplitude; activity events include the prior and new levels and score. Missing keypoints or a brief tracking dropout do not create observed-lowering events. Track expiry or a new session removes all pose history.

### Pose configuration

Use `--pose-config .\pose.example.json` to load the complete [pose.example.json](pose.example.json). You can supply a smaller JSON object containing only values you want to override. Invalid values and unknown keys fail at startup. `--joint-confidence` controls the M1 adapter's coordinate masking; keep it at or below `min_keypoint_confidence` if you want M3 to evaluate all qualifying joints.

| JSON key | Default | Reason |
| --- | ---: | --- |
| `min_keypoint_confidence` | 0.45 | Reject weak or cropped joints before gesture rules. |
| `gesture_activation_seconds` / `gesture_release_seconds` | 0.30 / 0.35 s | Require sustained evidence; lowering has slightly longer tolerance. |
| `missing_tolerance_seconds` | 0.50 s | Preserve an active hand across brief wrist loss without inventing its position. |
| `raised_margin_ratio` | 0.12 | Require wrist-to-shoulder height separation relative to visible body height. |
| `wave_window_seconds` | 2.0 s | Bound a plausible wave episode and its memory. |
| `wave_min_amplitude_ratio` / `wave_min_reversals` | 0.15 / 2 | Require noticeable side-to-side travel and repeated turns. |
| `wave_cooldown_seconds` / `wave_display_seconds` | 2.0 / 0.80 s | Avoid duplicate wave events while keeping a short visible indication. |
| `activity_smoothing_alpha` | 0.35 | Dampen isolated pose jitter. |
| `activity_thresholds` | `[0.12, 0.30, 0.60]` | Divide still, low, moderate, and high scores. |
| `activity_debounce_seconds` | 0.35 s | Prevent a single score spike from changing level. |

Increase confidence, margins, amplitude, debounce, or activity thresholds if noisy detections create false positives. Lower them cautiously if real gestures are missed. These are starting values for a fixed front-facing webcam, not calibrated physical measurements.

## Spatial defaults and tuning

| Option | Default | Meaning |
| --- | ---: | --- |
| `--zone-debounce` | 0.35 s | Continuous observed inside/outside time before zone enter/exit. About two or three observations at common CPU frame rates suppress boundary jitter. |
| `--track-grace` | 0.75 s | Retain a missing temporary ID through a brief one- or two-frame dropout. |
| `--history-seconds` | 2.0 s | Upper age of per-track samples; the buffer also caps at 64. |
| `--motion-window` | 1.2 s | Recent samples used for camera-relative motion. |
| `--min-observation` | 0.6 s | Minimum span before classifying motion. |
| `--motion-debounce` | 0.25 s | Sustained candidate time before a motion-state transition. |
| `--movement-threshold` | 0.03 | Maximum normalized anchor displacement for stationary classification. |
| `--scale-threshold` | 0.035 | Minimum normalized box-height change for approaching/departing. |

For a noisy camera, increase the debounce values and thresholds. For slower movement, lower `--scale-threshold` carefully; small values can turn pose jitter into false approach/departure. For example:

```powershell
.\.venv\Scripts\python.exe -m reactive_vision.main --webcam 0 --zones .\zones.example.json --zone-debounce 0.5 --track-grace 1.0 --movement-threshold 0.04 --scale-threshold 0.05
```

## M4 room state

Every JSON snapshot has a typed `room_state` object:

| Field | Meaning |
| --- | --- |
| `occupancy` | Current `visible_count`, debounced `stabilized_count` (or `null` before confirmation), and `unknown`, `empty`, `single`, `small_group`, or `crowd`. |
| `spatial` | Current normalized anchor centroid, left/centre/right counts, mean bounding-box-height proximity proxy, RMS image-space dispersion, current zone occupancy and most occupied zone ID. |
| `activity` | Quality-filtered, smoothed score and debounced level, plus contributing person count and current-frame coverage ratio. |
| `motion_summary` | Current approaching, departing, stationary and unknown person counts. |
| `gesture_summary` | Current left/right/both raised, recently waving and gesture-unknown person counts. `both_hands_raised` counts people once. |
| `interaction` | Debounced `unknown`, `none`, `presence`, `approaching` or `signal_detected`. |
| `focus_candidate` | `null` or one temporary track ID, 0–1 score, evidence strings and `is_visible`. |

The room analyzer runs for **every analyzed frame**, including frames skipped by the JSON output rate. `visible_count` always describes detections in that frame. `stabilized_count` and occupancy state retain the last confirmed band through brief dropouts: a newly occupied band needs 0.35 s of continuous evidence, and empty needs 0.75 s. Count changes within a confirmed band update immediately. At startup the state is `unknown` until a band is confirmed; this first confirmation emits no occupancy transition events. A failed read, unavailable camera, stale input or EOF gives `unknown` occupancy and interaction, with no `room_became_empty` event. The analyzer's history is limited to 3 seconds and 128 samples and resets with the session. A stale input is one whose supplied age exceeds 5 seconds; the synchronous capture loop normally reports a failed read or EOF instead of aging a queued frame.

Spatial values use **current** bottom-centre anchors, so a missing person is not included even while occupancy remains stabilized. The centroid is their arithmetic mean. Dispersion is root-mean-square Euclidean distance from the centroid in normalized image coordinates. Empty centroid, proximity and dispersion are `null`. Horizontal boundaries default to one-third and two-thirds; the left bound belongs to centre, and the right bound belongs to right. Overlapping zones each count a person; the most occupied positive-count zone uses lexical zone ID as the tie breaker. Proximity and spread are camera-relative image measurements, not metres, physical density or ground-plane movement.

Activity uses each current person's M3 score only when pose quality is `good` (weight 1) or `partial` (weight 0.5). `poor` and `unavailable` contribute nothing. Coverage counts valid contributors divided by currently visible people; below 0.5 coverage the aggregate is `unknown` with a `null` score. The weighted mean naturally reduces one active person's effect among several valid people. A time-adjusted EWMA with alpha 0.35 per 0.2 s smooths the mean regardless of processed frame rate; score bands are still below 0.12, low below 0.30, moderate below 0.60 and high otherwise. Levels require 0.35 s of continuous candidate evidence. The score describes observable image movement only.

Interaction precedence is `signal_detected` (confirmed M3 raised hand or recent wave), then `approaching` (confirmed M2 camera-relative approach or membership in a confirmed zone with role `interaction`), then `presence`, then `none` when occupancy is confirmed empty. Changes require 0.35 s of sustained evidence. The state is an observable cue, not intent or emotion.

Focus scoring uses one recent wave (0.45), both hands raised (0.40) **or** one hand raised (0.25), interaction-zone membership (0.20), and approaching motion (0.15). Once at least one such cue exists, proximity adds up to 0.10 and confirmed-zone dwell adds up to 0.10 (full credit at 3 s). The score is capped at 1; the minimum is 0.15. Ordinary presence, proximity and dwell alone cannot create focus. Ties go to the lower temporary track ID. A challenger must exceed the current score by **more than** 0.15 to switch; the focused ID survives 0.75 s of invisibility with `is_visible:false`, or 0.75 s without the original cue while still visible, then clears. The retained score/evidence are the last confirmed values. This is a short-lived display candidate, not identity, recognition, attention or emotion.

Room transitions share the M2/M3 session event IDs and buffer: `occupancy_state_changed`, `room_became_occupied`, `room_became_empty`, `room_activity_changed`, `interaction_state_changed`, and `focus_candidate_changed`. They include previous/new values; focus changes also include previous/new IDs and scores. Room events have `track_id:null`. Initial occupancy, activity and interaction confirmation from `unknown` sets a baseline without a transition event. Lost activity coverage produces `unknown` immediately without a behavioral transition event; restored evidence establishes a new baseline. Focus first appears only with a meaningful cue and emits one candidate change. Person events from a frame precede room events from that frame. All events between JSONL emissions arrive on the next emitted line, including terminal lines. A brief missing focus emits no change unless another candidate clears the switch margin.

### Room configuration

Use `--room-config .\room.example.json` to load [room.example.json](room.example.json), or omit it for the defaults above. Partial JSON objects may override only selected keys. The file covers occupancy bands, debounce times, horizontal boundaries, room history/stale age, activity smoothing and its reference interval/coverage/thresholds, interaction debounce, and all focus weights, score threshold, switch margin and loss tolerance. Durations must be nonnegative; thresholds and weights must be finite and in range; horizontal and activity boundaries must increase; and `crowd_min` must equal `small_group_max + 1` so no count falls between bands. Invalid JSON or keys fail at startup.

## JSON contract

The schema version is **`4.0`** because `SceneSnapshot.room_state` and room transition event types extend the contract. Existing M1–M3 fields retain their meanings. Each stdout line is a complete JSON object with no NumPy values or non-finite numbers. Missing measurements are `null` or `unknown`; a camera failure has `camera_status:"unavailable"` or `"read_error"`, whereas an empty processed frame has `camera_status:"ok"` and `visible_count:0`. Video `eof` is a terminal line. A new run starts a new session and resets ByteTrack, spatial, pose and room history, event sequence, and pending events.

Webcam analysis uses UTC observation time. For video, available source seconds are mapped onto a UTC playback timeline so motion thresholds follow recorded frame spacing even when inference is slower. `source_frame_index` and `source_time_seconds` remain available as separate file-position fields. `processing_fps` is the rolling inference and analysis throughput, excluding preview and deliberate video pacing. JSON output is limited by `--rate` (default 5 Hz); frames are processed synchronously without an application backlog. Events from frames omitted by rate limiting are delivered on the next JSON line.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
python -m uv pip check --python .venv\Scripts\python.exe
```

Validated on Windows with Python **3.11.16**, Ultralytics **8.4.156**, `opencv-python` **5.0.0.93** (OpenCV reports 5.0.0), Pydantic **2.13.5**, PyTorch **2.14.0+cpu**, NumPy **2.4.6**, `lap` **0.5.13**, pytest **9.1.1**, FastAPI **0.141.1**, Uvicorn **0.53.0**, Starlette **1.6.0**, `websockets` **16.1.1** and `httpx` **0.28.1**. `lap` is declared explicitly because this ByteTrack path requires it; `websockets` supplies Uvicorn's socket protocol and the example Python client. Synthetic tests cover the broker, endpoints, sockets and runtime without a camera or weights download. A separate real-model headless video smoke check verified both the CLI and localhost service, including HTTP, WebSocket and EOF. No physical webcam was tested. The current FastAPI TestClient stack reports two upstream deprecation warnings about `httpx` and AnyIO while tests pass.

Manual webcam checklist:

1. Stand still with both arms down; check for inactive hands and low activity.
2. Raise one hand and hold it; check one `hand_raise_started` event and a sustained active state.
3. Lower it; check one `hand_raise_ended` event after the release debounce.
4. Raise both hands; check `both_hands_raised` and one start event.
5. Hold one hand still above the shoulder; check that no wave is reported.
6. Wave that hand side to side; check one side-specific `wave_detected` and a short active indication.
7. Walk sideways without waving; check that body translation alone does not cause a wave.
8. Move your arms actively while staying roughly in one place; check that activity rises even if M2 motion stays `stationary`.
9. Briefly hide a raised wrist; check active-hand tolerance followed by `unknown` if hidden longer, without a false end event.
10. Leave and re-enter the frame; check that an expired track starts with fresh gesture state. Press Q or Escape and check camera release.

M4 room checklist:

1. Start with an empty scene; after the empty debounce, check `empty` and null geometry.
2. Enter alone and remain still; compare current visible count with confirmed `single` and low activity.
3. Walk toward the camera; check approaching flow and the interaction cue if M2 confirms it.
4. Raise one hand; check `signal_detected` after debounce and a focus candidate.
5. Wave; check recently waving and focus evidence without repeated events each frame.
6. Add a second person; check `small_group`, spatial counts and centroid.
7. Have both people move actively; check activity coverage, score and level.
8. Have one person signal while the other remains closer; check that focus follows the configured cues and switch margin.
9. Briefly occlude the focus candidate; check `is_visible:false` during the grace interval, then recovery or timeout.
10. Have everyone leave; check that a short detection gap stays occupied and sustained absence confirms empty once.
11. Disconnect or block camera input; check `camera_status` and `unknown` room state rather than `empty`.

M5 service checklist:

1. Start the service with the webcam while a second terminal calls `/health`, `/ready` and `/api/v1/scene/latest`; confirm they answer during model loading.
2. Subscribe with a browser or Python WebSocket client; confirm an immediate snapshot, later snapshots and idle heartbeats with separate `type` values.
3. Enter, leave and briefly occlude people; confirm service data matches the existing CLI's scene semantics.
4. Disconnect and reconnect the socket; confirm the new connection receives only the current latest scene.
5. Try a browser origin absent from `--cors-origin`; confirm the socket is rejected, then allow it explicitly.
6. Disconnect and reconnect the webcam; confirm `unavailable`, then a new session ID on recovery without stale track IDs.
7. Run a local video to EOF; confirm `eof` persists unless `--loop-video` is supplied.
8. Stop the service with Ctrl+C; confirm the camera is released. A driver blocked in a native read may exceed the configured wait and needs platform-specific verification.

Known limits: gestures are rules based on 2D pose, so occlusion, cropping, poor lighting, camera movement, mirrored display, and ByteTrack ID switches can cause missed or false results. Room focus follows temporary session track IDs and may change after an ID switch. Activity is an image-based movement score, not physical energy or emotion. Bbox height is sensitive to pose and cropping; image-plane velocity, centroid and dispersion are not ground-plane physical measures. Unknown video time metadata falls back to processing time. Physical-webcam preview, thresholds, and GPU use still require device-specific verification.

API references: [Ultralytics tracking and `persist=True`](https://docs.ultralytics.com/modes/track), [pose models and COCO keypoints](https://docs.ultralytics.com/tasks/pose), [result boxes/keypoints](https://docs.ultralytics.com/reference/engine/results), [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/), [FastAPI WebSocket testing](https://fastapi.tiangolo.com/advanced/testing-websockets/), and [Uvicorn settings](https://www.uvicorn.org/settings/).
