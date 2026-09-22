# M9 — Live camera overlay

Aura is a browser camera surface with an original compact SVG reaction character. The video fills the viewport; smoked-glass optics, titanium rims, cyan pupils and a layered warm metallic thumb sit above it. No external art, fonts or animation service is used. The existing schema 4.0 scene pipeline and sanitized M8 verification boundary remain in use.

## Windows PowerShell setup

Python 3.11 and Node.js 22.12+ are required. From a fresh checkout:

```powershell
cd 'G:\web apps\ComputerVision'
python -m pip install --user uv
python -m uv python install 3.11
python -m uv venv --python 3.11 .venv
python -m uv pip install --python .venv\Scripts\python.exe -e ".[test]"
cd display
npm ci
npm run build
cd ..
```

Do not set `VITE_VISION_HTTP_URL` or `VITE_VISION_WS_URL` for a same-origin build. Remove old overrides from `.env.local` before building. Vite variables are public build-time configuration; never put an ingest token there.

### One-command local run after setup

```powershell
.\.venv\Scripts\python.exe -m reactive_vision.server --browser-camera --ingest-token "temporary-review-token" --serve-display .\display\dist --host 127.0.0.1 --port 8000
```

Open **http://localhost:8000**. Press **Start camera**, then **Connect vision** and enter the same token. Starting the camera alone only previews locally. Connecting explicitly starts upload. Disconnect vision to keep local preview; Diagnostics → Stop camera releases the camera and disconnects. Switching cameras stops old tracks and upload; reconnect explicitly for the new camera.

The default lightweight `yolo26n-pose.pt` is downloaded by Ultralytics if absent. To avoid a download, add `--model 'C:\models\yolo26n-pose.pt'`. No training occurs. CPU is the default; optional `--device 0` requires a working compatible GPU/PyTorch installation. Existing `--imgsz`, `--confidence`, `--rate`, zone/pose/room configuration and local webcam/video modes remain available. `--rate` limits scene publication (default 5 Hz), separately from camera upload (10 Hz).

### Physical phone on the LAN: user-supplied trusted HTTPS

A phone opening a laptop LAN address needs a **trusted HTTPS origin** for camera access; plain LAN HTTP is insufficient. Obtain a development certificate with the laptop IP/hostname in its subject alternative names, issued by a CA that the phone trusts. Configure that trust yourself using your certificate provider's instructions. This project does not generate or install certificates. Place the certificate chain and private key in `certs` (gitignored), then run:

```powershell
cd 'G:\web apps\ComputerVision'
.\.venv\Scripts\python.exe -m reactive_vision.server `
  --browser-camera `
  --host 0.0.0.0 `
  --port 8000 `
  --ingest-token "temporary-review-token" `
  --serve-display .\display\dist `
  --ssl-certfile .\certs\review-cert.pem `
  --ssl-keyfile .\certs\review-key.pem
```

Open **https://LAPTOP_IP:8000** on the phone using the address covered by the certificate. Allow the private-LAN firewall port as needed. The page, ingestion socket and scene stream use that same origin. Do not expose this service publicly: ingestion is authenticated, but scene read/health endpoints retain M5's unauthenticated local-review design. Use a fresh token for review; CLI tokens are visible in process arguments. Alternatively set `VISION_INGEST_TOKEN` in the server environment and omit `--ingest-token`. Never put the token in a URL or commit it.

### Frontend development and explicit demos

```powershell
cd 'G:\web apps\ComputerVision\display'
npm run dev
```

- **http://localhost:3000/?demo=1** — neutral visual-only sample; no camera request, no production socket.
- **http://localhost:3000/?demo=1&camera=1** — press Start camera for local video under the scripted 18-step demo; **no frame upload**, no production socket.
- **http://localhost:3000/** — live camera mode. For a separate backend use `--cors-origin http://localhost:3000`; this Vite port defaults to the backend at `http://127.0.0.1:8000`.

Demo steps: quiet room, arrival, wake, approach, verification start/success, thumb, hand raise, wave, both hands, activity, second person, focus change, departure, sleep, source failure, recovery, retry. The DEMO badge stays visible. Open Review sequence for previous/next/play/pause/restart/scenario selection and Preview verification. The sheet is collapsed by default. Diagnostics includes reduced motion, fullscreen and stop camera. Keyboard: D diagnostics, M reduced motion, F fullscreen, Escape close sheet.

## Capture and transport architecture

`camera.ts` owns `getUserMedia({video: ..., audio: false})`, front/rear preference, generation guards, track events and cleanup. Front video is mirrored via CSS; rear is not. Frames sent to inference are always unmirrored. The video is muted, autoplay, playsInline, `object-fit: cover`. Permission denial, missing device, busy device, interruption and unsupported/insecure context have separate states. Permission begins only on a user action.

`ingest.ts` draws into a private, reusable canvas and encodes JPEG quality 0.78 at approximately 10 Hz, fitting the original aspect ratio into 640×360 by default. `VITE_CAPTURE_WIDTH`/`VITE_CAPTURE_HEIGHT` change the size envelope. There is at most one pending encoder operation; congestion above 256 KiB `bufferedAmount`, hidden pages and unavailable video cause drops. Encoding completions from a previous connection are discarded. Rendering local video does not wait for inference. Nothing is recorded, saved or sent until authenticated connection readiness.

Protocol at **WS /api/v1/camera/ingest**:

1. Server accepts a socket and waits at most 5 seconds for a text control message `{"type":"auth","token":"..."}` (maximum 1 KiB).
2. Constant-time token comparison; token must be 16–256 characters at CLI startup. Token is never in the socket URL. UI clears it after submission/closing and retains no persistent copy.
3. One authorized producer receives `{"type":"ready","session_id":"...","max_frame_bytes":524288,"max_fps":12}`. A second producer is rejected (1013); invalid authentication/rate is 1008.
4. Subsequent messages are binary JPEGs. Maximum 512 KiB/frame, 12 frames per rolling second, each dimension ≤1280, total pixels ≤921600. JPEG dimensions are checked before OpenCV allocation. Malformed frames close with 1003; oversized with 1009.
5. `browser_camera.py` decodes each message and retains only the newest pending frame. A dedicated worker passes it through **the existing VisionSession, PoseTracker/ByteTrack, spatial/gesture/room analyzers and SnapshotBroker**. No parallel recognition implementation exists. Uvicorn uses a bounded socket queue and disables compression.
6. Each producer gets a fresh session/tracker/temporal state. Disconnection clears pending input, wakes the worker and publishes unavailable. In-flight results from an old producer cannot enter a new session. Missing frames produce stale health. The display consumes the existing M5 scene stream and ignores other producer sessions.

No application logging or storage includes frames or tokens. Browser-camera mode caps server debug logging at info because WebSocket protocol debug output can contain payloads. Frames exist only transiently in memory. `--browser-camera`, `--webcam` and `--video` are mutually exclusive. `--serve-display` is opt-in: API/health/ready/WebSocket routes remain reserved; only extensionless frontend paths receive SPA fallback, with path containment checks.

## Placement and reactions

`placement.ts` maps normalized unmirrored boxes through the `object-fit: cover` scale `max(viewWidth/sourceWidth, viewHeight/sourceHeight)`, subtracts centered crop offsets, then mirrors the displayed X coordinate for a front camera. Rotation recomputes dimensions. The focus candidate must be visible and current. The sticker is above its box when possible, otherwise beside it; missing/stale anchors fall back to upper-middle. Safe margins include CSS safe-area padding and controls. Exponential smoothing with a 260 px/s speed bound prevents jumps. Physical face occlusion still needs human camera review.

`Character.tsx` provides original inline SVG masks, gradients and layered vectors. `renderer.ts` changes SVG attributes on animation frames without React rerendering each frame. Aperture masks blink without squashing the eye housing. Wake has a thin trace and short Hello pill; approach subtly widens/warms the optics; hand raise acknowledges once; wave has restrained arcs; both hands create a 1.4-second symmetric light burst. Verified closes/reopens apertures, brightens pupils, slides the thumb in and expands a ring once for 1.8 seconds. Retry uses a neutral tilt and Try again. Departure fades to a light line. Reduced motion keeps clear static symbols and labels without bounce/wave. Animation pauses in hidden tabs and cleans up its observers and RAF.

**There is no real verifier.** Only explicit demo mode instantiates the simulated verification source. Live detection, gestures and temporary ByteTrack IDs never imply identity or Verified. M8 freshness/expiry/deduplication and accepted-pending-session checks are preserved. A future provider must supply sanitized verification events through `verification.ts`; the reaction renderer receives no images, embeddings, names or biometric data. The capture/upload path is separate from that boundary.

## Verification and manual checklist

```powershell
cd 'G:\web apps\ComputerVision\display'
npm run test
npm run lint
npm run typecheck
npm run build
cd ..
.\.venv\Scripts\python.exe -m pytest -q
```

Automated tests use fake camera streams and tiny generated JPEGs, never a physical webcam or a model download. See [M9 validation record](M9_VALIDATION.md) for exact results and versions. The development-only `test-support/camera.html` fixture renders a labeled synthetic canvas stream through the real video/capture path; it is not included in the production bundle.

Manual physical-device checks still required:

- Trusted phone HTTPS and camera permission; denial/no-device/busy/interruption messages.
- Front mirror, unmirrored rear, switch stops old tracks, new connection/session, portrait/landscape rotation.
- Stand/walk/raise hand/wave; verify person-relative alignment, latency and face visibility on real people.
- Leave frame: visible count reaches zero; disconnect or stop camera: unavailable is distinct from empty.
- Network slowdown drops frames without accumulating delay; resume/reconnect remains responsive.
- Close page/stop camera: OS camera indicator clears. Check mobile suspend/resume and performance.
- Both demo modes remain clearly labeled; camera demo never uploads; live mode never claims verification.
