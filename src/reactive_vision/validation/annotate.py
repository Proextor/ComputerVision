"""Local OpenCV annotations; never writes video frames or modifies the video."""

import os
from pathlib import Path
import tempfile

import cv2

from .schemas import Annotations, ExpectedEvent, StateInterval

EVENTS = ["person_appeared", "person_disappeared", "zone_entered", "zone_exited",
          "approach_started", "departure_started", "became_stationary",
          "hand_raise_started", "hand_raise_ended", "both_hands_raised_started",
          "both_hands_raised_ended", "wave_detected", "room_became_occupied",
          "room_became_empty", "focus_candidate_changed"]
STATE_VALUES = {
    "visible_count": [0, 1, 2, 3],
    "occupancy_state": ["empty", "single", "small_group", "crowd", "unknown"],
    "interaction_state": ["none", "presence", "approaching", "signal_detected", "unknown"],
    "motion_state": ["stationary", "approaching", "departing", "unknown"],
    "activity_level": ["still", "low", "moderate", "high", "unknown"],
    "zone_membership": [[], ["interaction"]],
    "brief_occlusion": [True],
}
HELP = ["SPACE play/pause    J/L seek -/+ 0.25s    SHIFT+J/L seek -/+ 2s",
        ",/. choose event    E mark point event    B/N event window start/end",
        "T choose interval state    V choose value    [/] interval start/end",
        "A cycle anonymous actor    H cycle hand side    U undo last entry",
        "S save (Y confirms existing file)    ? help    Q/ESC quit"]


def template(scenario_id: str, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"annotation file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(Annotations(scenario_id=scenario_id).model_dump_json(indent=2) + "\n", encoding="utf-8")


def save_sidecar(output: Path, annotations: Annotations) -> None:
    """Called only after interactive overwrite confirmation (or on new file)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         suffix=".annotations.tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(annotations.model_dump_json(indent=2) + "\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def annotate(video: Path, output: Path, *, scenario_id: str | None = None,
             zone_id: str | None = None) -> None:
    if not video.is_file():
        raise ValueError(f"video not found: {video}")
    scenario_id = scenario_id or video.stem
    annotations = (Annotations.model_validate_json(output.read_text(encoding="utf-8"))
                   if output.is_file() else Annotations(scenario_id=scenario_id))
    if annotations.scenario_id != scenario_id:
        raise ValueError("existing annotation scenario ID does not match")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError("cannot open local video")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError("video FPS unavailable; timestamped annotation requires a valid FPS")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    event_index = 0
    state_index = 0
    value_index = 0
    actor_index = 0
    side_index = 0
    actors = [None, "person_a", "person_b", "person_c"]
    sides = [None, "left", "right"]
    event_start: float | None = None
    interval_start: float | None = None
    playing = False
    help_open = True
    confirm = False
    dirty = False
    frame = None
    index = 0
    window = "Reactive Vision validation annotations"
    try:
        while True:
            if frame is None or playing:
                ok, image = capture.read()
                if not ok:
                    playing = False
                    if frame is None:
                        break
                else:
                    frame = image
                    index = max(0, int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            assert frame is not None
            at = index / fps
            view = frame.copy()  # display copy only; no frame is written to disk.
            info = [f"{scenario_id}  {at:.3f}s  frame {index}/{max(0, frames - 1)}  "
                    f"{'PLAY' if playing else 'PAUSE'}",
                    f"Event: {EVENTS[event_index]}  actor: {actors[actor_index] or 'unmapped'}  "
                    f"hand: {sides[side_index] or 'unspecified'}  zone: {zone_id or 'unspecified'}",
                    f"Interval: {list(STATE_VALUES)[state_index]}="
                    f"{STATE_VALUES[list(STATE_VALUES)[state_index]][value_index]}  "
                    f"events: {len(annotations.events)}  states: {len(annotations.states)}",
                    f"Event start: {event_start}  interval start: {interval_start}  "
                    f"{'CONFIRM OVERWRITE: press Y' if confirm else 'unsaved edits' if dirty else 'saved'}"]
            if help_open:
                info += HELP
            overlay = view.copy()
            cv2.rectangle(overlay, (0, 0), (view.shape[1], min(view.shape[0], 30 + len(info) * 27)), (8, 15, 24), -1)
            cv2.addWeighted(overlay, 0.83, view, 0.17, 0, view)
            for line, text in enumerate(info):
                cv2.putText(view, text, (12, 24 + line * 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, (230, 244, 238), 1, cv2.LINE_AA)
            cv2.imshow(window, view)
            key = cv2.waitKey(max(1, round(1000 / fps)) if playing else 0) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("?"):
                help_open = not help_open
            elif key == ord(" "):
                playing = not playing
            elif key in (ord("j"), ord("J"), ord("l"), ord("L")):
                delta = (-1 if key in (ord("j"), ord("J")) else 1) * (2 if key in (ord("J"), ord("L")) else 0.25)
                index = max(0, min(max(0, frames - 1), round((at + delta) * fps)))
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                frame = None
            elif key == ord(","):
                event_index = (event_index - 1) % len(EVENTS)
            elif key == ord("."):
                event_index = (event_index + 1) % len(EVENTS)
            elif key == ord("a"):
                actor_index = (actor_index + 1) % len(actors)
            elif key == ord("h"):
                side_index = (side_index + 1) % len(sides)
            elif key == ord("t"):
                state_index = (state_index + 1) % len(STATE_VALUES)
                value_index = 0
            elif key == ord("v"):
                value_index = (value_index + 1) % len(STATE_VALUES[list(STATE_VALUES)[state_index]])
            elif key == ord("e"):
                annotations.events.append(ExpectedEvent(type=EVENTS[event_index], time_seconds=at,
                    actor=actors[actor_index], hand_side=sides[side_index] if EVENTS[event_index] in ("hand_raise_started", "hand_raise_ended", "wave_detected") else None,
                    zone_id=zone_id if EVENTS[event_index] in ("zone_entered", "zone_exited") else None))
                dirty = True
            elif key == ord("b"):
                event_start = at
            elif key == ord("n") and event_start is not None and at >= event_start:
                annotations.events.append(ExpectedEvent(type=EVENTS[event_index], start_seconds=event_start,
                    end_seconds=at, actor=actors[actor_index], hand_side=sides[side_index] if EVENTS[event_index] in ("hand_raise_started", "hand_raise_ended", "wave_detected") else None,
                    zone_id=zone_id if EVENTS[event_index] in ("zone_entered", "zone_exited") else None))
                event_start = None
                dirty = True
            elif key == ord("["):
                interval_start = at
            elif key == ord("]") and interval_start is not None and at > interval_start:
                field = list(STATE_VALUES)[state_index]
                value = STATE_VALUES[field][value_index]
                if field == "zone_membership" and value:
                    value = [zone_id or "interaction"]
                annotations.states.append(StateInterval(
                    start_seconds=interval_start, end_seconds=at,
                    actor=actors[actor_index] if field in ("motion_state", "zone_membership", "brief_occlusion") else None,
                    **({"brief_occlusion": True} if field == "brief_occlusion" else {field: value}),
                ))
                interval_start = None
                dirty = True
            elif key == ord("u"):
                if annotations.states:
                    annotations.states.pop()
                elif annotations.events:
                    annotations.events.pop()
                dirty = True
            elif key == ord("s"):
                if output.exists():
                    confirm = True
                else:
                    save_sidecar(output, annotations)
                    dirty = False
            elif key == ord("y") and confirm:
                save_sidecar(output, annotations)
                dirty = confirm = False
            else:
                confirm = False
    finally:
        capture.release()
        cv2.destroyWindow(window)
