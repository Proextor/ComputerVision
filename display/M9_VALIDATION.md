# M9 validation record

Validated on Windows, 2026-09-22. No physical camera or physical phone was used.

## Automated results

- Complete Python suite: **110 passed**, 2 upstream deprecation warnings (Starlette/httpx and AnyIO alias). Includes the prior 98 regression tests and 12 browser-ingestion cases.
- Frontend: **71 passed**, 7 test files.
- `npm run lint`: passed.
- `npm run typecheck`: passed.
- `npm run build`: passed. Production output contains index.html and bundled JS/CSS; the synthetic camera fixture is excluded. JS approximately 291 kB, 86 kB gzip; CSS approximately 9.3 kB, 2.9 kB gzip.

Coverage includes permission/cleanup/switching/mirroring, authentication readiness, capture throttling/backpressure, cancellation after disconnect, cover/mirrored/portrait/landscape mapping, bounded smoothing/clamping, inward-side mirrored placement, demo without ingest, verification freshness/deduplication/once-only success, retry without thumb and reduced-motion symbols. Backend covers invalid auth, second producer, size/rate/image validation, bounded newest-frame buffer, reconnect sessions, stale/disconnected health, shared pipeline boundary, no image writes and static/API routing.

## Browser and real-model smoke

Used the Codex in-app Chromium browser and a development-only labeled synthetic canvas MediaStream. This exercises HTMLVideoElement playback, browser JPEG encoding and the actual upload path, without camera hardware.

- Explicit start produced local playback; front preview mirrored and rear preview unmirrored in the earlier synthetic camera review.
- Camera demo produced no ingest publication; both demos retained visible DEMO identification.
- Live synthetic upload authenticated to localhost, traversed the real `PoseTracker` with local `yolo26n-pose.pt`, explicit ByteTrack and shared M1–M4 processing, and delivered schema 4.0 snapshots through M5.
- Observed service ready, camera ok, advancing frame/publication IDs, visible_count 0, empty occupancy and no verifier. At a diagnostic sample, 421 frames had been sent, 0 dropped and animation measured approximately 60 FPS. This is a synthetic desktop observation, not a phone performance claim.
- Stop camera disconnected upload and scene stream; backend changed to unavailable with zero scene clients.
- Inspected 320×568, 360×800, 390×844, 412×915, 844×390 and 1366×768. Controls and compact overlay stayed visible; page dimensions matched viewport without page scrolling. Bottom sheets were inspected at small portrait and landscape. Diagnostic sheets may scroll internally.
- Inspected original thumb/Verified pill at phone scale, including the static reduced-motion alternative. No cream eyeballs, brows, starfield or permanent greeting remains in the active surface.
- Production frontend reviewed from FastAPI same-origin static serving on localhost. API routing is separately covered by tests.

No real-person detection quality, face occlusion, mobile thermal/frame performance, camera permissions on hardware, GPU operation or trusted LAN TLS was verified. No real verification provider exists; success is demo-only. No frames were recorded or persisted. Manual checks are in README.md.

## Actual environment versions

No new dependency was needed for M9. These are installed versions actually used in these checks, not a claim about every version allowed by pyproject.toml:

| Component | Version |
|---|---|
| Python | 3.11.16 |
| OpenCV Python | 5.0.0.93 |
| Ultralytics | 8.4.156 |
| Torch | 2.14.0 |
| lap | 0.5.13 |
| Pydantic | 2.13.5 |
| FastAPI | 0.141.1 |
| Uvicorn | 0.53.0 |
| websockets | 16.1.1 |
| pytest | 9.1.1 |
| Node.js | 24.13.0 |
| npm | 11.6.2 |
| React / React DOM | 19.3.0 |
| Vite | 8.3.0 |
| TypeScript | 5.9.3 |
| Vitest | 5.0.1 |

Model: local pretrained `yolo26n-pose.pt`; no training or model download during this smoke test. CPU inference completed successfully. Frontend exact dependency graph remains in package-lock.json.

## Files in M9

Backend: new `src/reactive_vision/browser_camera.py`; updated `server.py`, `main.py`, `pipeline.py`; new `tests/test_browser_camera.py`.

Frontend: new `camera.ts`, `ingest.ts`, `placement.ts`, `Character.tsx`; updated `App.tsx`, `renderer.ts`, `reaction.ts`, `contracts.ts`, `config.ts`, `styles.css`, `index.html`, `.env.example`; new `src/test/camera.test.ts`; updated `App.test.tsx`, `demo.test.ts`, `verification.test.ts`; new development-only `test-support/camera.html`.

Documentation: updated root README.md, display/README.md and .gitignore; new display/M9_VALIDATION.md.
