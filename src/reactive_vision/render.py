"""Debug overlay only; inference and serialization do not depend on it."""

from collections.abc import Sequence

import cv2
import numpy as np

from .schemas import GestureStatus, SceneSnapshot
from .zones import Zone


SKELETON = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
)

STATUS_COLORS = {
    GestureStatus.ACTIVE: (40, 255, 40),
    GestureStatus.INACTIVE: (190, 190, 190),
    GestureStatus.UNKNOWN: (0, 180, 255),
}


def draw_preview(
    frame: np.ndarray, snapshot: SceneSnapshot,
    joint_threshold: float = 0.25, zones: Sequence[Zone] = (),
) -> np.ndarray:
    image = frame.copy()
    height, width = image.shape[:2]
    for zone in zones:
        polygon = np.asarray([(round(x * width), round(y * height)) for x, y in zone.polygon], dtype=np.int32)
        cv2.polylines(image, [polygon], True, (255, 120, 0), 2)
        label_at = tuple(int(value) for value in polygon[0])
        cv2.putText(image, zone.label, (label_at[0], max(48, label_at[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 0), 2)
    for person in snapshot.people:
        box = person.bbox
        p1 = (round(box.x1 * width), round(box.y1 * height))
        p2 = (round(box.x2 * width), round(box.y2 * height))
        focus = snapshot.room_state.focus_candidate if snapshot.room_state else None
        focused = focus is not None and focus.is_visible and person.track_id == focus.track_id
        cv2.rectangle(image, p1, p2, (255, 0, 255) if focused else (0, 200, 255), 4 if focused else 2)
        label = f"ID {person.track_id}" if person.track_id is not None else "ID pending"
        if focused:
            label += " FOCUS"
        current_zones = ",".join(person.zones) if person.zones else "none"
        label += f"  {current_zones}  {person.motion_state.value}"
        if person.dwell_seconds:
            label += "  " + ",".join(f"{name}:{seconds:.1f}s" for name, seconds in person.dwell_seconds.items())
        cv2.putText(image, label, (p1[0], max(46, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 2)
        if len(person.trajectory) >= 2:
            points = np.asarray(
                [(round(point.x * width), round(point.y * height)) for point in person.trajectory],
                dtype=np.int32,
            )
            cv2.polylines(image, [points], False, (255, 0, 255), 2)
        if person.anchor_point is not None:
            anchor = (round(person.anchor_point.x * width), round(person.anchor_point.y * height))
            cv2.circle(image, anchor, 5, (255, 0, 255), -1)
        if person.pose_analysis is not None:
            pose = person.pose_analysis
            score = f"{pose.activity.activity_score:.2f}" if pose.activity.activity_score is not None else "--"
            quality = pose.quality.pose_quality.value
            quality_color = ((40, 255, 40) if quality == "good" else
                             (0, 220, 255) if quality == "partial" else (0, 120, 255))
            line_y = min(height - 40, max(20, p1[1] + 18))
            cv2.putText(image, f"Pose {quality} {pose.quality.usable_keypoint_count}/17",
                        (p1[0], line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, quality_color, 2)
            cv2.putText(image, f"Activity {pose.activity.activity_level.value} {score}",
                        (p1[0], line_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 2)
            gestures = pose.gestures
            labels = (
                ("L", gestures.left_hand_raised), ("R", gestures.right_hand_raised),
                ("Both", gestures.both_hands_raised),
                ("WaveL", gestures.left_wave), ("WaveR", gestures.right_wave),
            )
            x = p1[0]
            for title, status in labels:
                token = f"{title}:{status.value[0].upper()} "
                cv2.putText(image, token, (x, line_y + 34), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, STATUS_COLORS[status], 2)
                x += cv2.getTextSize(token, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)[0][0]
        visible = {}
        for joint in person.keypoints or []:
            if joint.x is None or joint.y is None or joint.confidence is None or joint.confidence < joint_threshold:
                continue
            point = (round(joint.x * width), round(joint.y * height))
            visible[joint.index] = point
            cv2.circle(image, point, 3, (40, 255, 40), -1)
        for start, end in SKELETON:
            if start in visible and end in visible:
                cv2.line(image, visible[start], visible[end], (40, 255, 40), 2)
    fps = f"{snapshot.processing_fps:.1f}" if snapshot.processing_fps is not None else "--"
    cv2.putText(image, f"Visible: {snapshot.visible_count}  Processing FPS: {fps}  {snapshot.camera_status.value}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(image, f"Visible: {snapshot.visible_count}  Processing FPS: {fps}  {snapshot.camera_status.value}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    room = snapshot.room_state
    if room is not None:
        activity = room.activity
        score = f"{activity.aggregate_activity_score:.2f}" if activity.aggregate_activity_score is not None else "--"
        focus = room.focus_candidate
        focus_text = (f"{focus.track_id} {focus.score:.2f}" + (" visible" if focus.is_visible else " held")) if focus else "none"
        centroid = room.spatial.centroid
        centroid_text = f"({centroid.x:.2f},{centroid.y:.2f})" if centroid else "--"
        gesture = room.gesture_summary
        lines = [
            f"Room: {room.occupancy.state.value} ({room.occupancy.stabilized_count})  now {snapshot.visible_count}",
            f"Activity: {activity.aggregate_activity_level.value} {score}",
            f"Interaction: {room.interaction.state.value}",
            f"Flow: toward {room.motion_summary.approaching} away {room.motion_summary.departing}",
            f"Gestures: L{gesture.left_hand_raised} R{gesture.right_hand_raised} B{gesture.both_hands_raised} W{gesture.recently_waving}",
            f"Focus: {focus_text}  Centre: {centroid_text}",
        ]
        panel_x = max(10, width - 405)
        for row, line in enumerate(lines):
            y = 55 + row * 20
            cv2.putText(image, line, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.47, (0, 0, 0), 3)
            cv2.putText(image, line, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.47, (255, 255, 255), 1)
    return image
