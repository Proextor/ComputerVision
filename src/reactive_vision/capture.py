"""Single-frame, synchronous OpenCV capture; no frame queue is created."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .schemas import CameraStatus, finite_or_none


@dataclass(frozen=True)
class CaptureResult:
    status: CameraStatus
    frame: np.ndarray | None = None
    source_frame_index: int | None = None
    source_time_seconds: float | None = None


class FrameSource:
    def __init__(self, webcam_index: int, video: Path | None, width: int, height: int):
        self.video = video
        self.label = str(video) if video is not None else f"webcam:{webcam_index}"
        self.capture = cv2.VideoCapture(str(video) if video is not None else webcam_index)
        self._index = 0
        self._opened = self.capture.isOpened()
        reported_count = finite_or_none(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) if self._opened and video is not None else None
        self._frame_count = int(reported_count) if reported_count is not None and reported_count > 0 else None
        if self._opened and video is None:
            # Best effort: some camera backends ignore this, but we never queue frames ourselves.
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> CaptureResult:
        if not self._opened:
            return CaptureResult(CameraStatus.UNAVAILABLE)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            if self.video is None:
                status = CameraStatus.READ_ERROR
            elif self._frame_count is not None and self._index < self._frame_count:
                status = CameraStatus.READ_ERROR
            else:
                status = CameraStatus.EOF
            return CaptureResult(status)
        self._index += 1
        if self.video is None:
            return CaptureResult(CameraStatus.OK, frame)
        # OpenCV reports the position after read; fall back when a backend omits it.
        position = finite_or_none(self.capture.get(cv2.CAP_PROP_POS_MSEC))
        fps = finite_or_none(self.capture.get(cv2.CAP_PROP_FPS))
        seconds = position / 1000 if position is not None and position > 0 else None
        if seconds is None and fps is not None and fps > 0:
            seconds = (self._index - 1) / fps
        return CaptureResult(CameraStatus.OK, frame, self._index - 1, seconds)

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
