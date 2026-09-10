from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TEMP_DEFAULT_VIDEO_PATH = Path(
    r"../../720Example of Hi-Definition Video Surveillance of a Factory Floor - by CCTVDOC.COM.mp4"
)
VideoInput = str | Path | int


class VideoSourceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FramePacket:
    camera_id: str
    frame: np.ndarray
    frame_index: int
    timestamp: float
    fps: float
    width: int
    height: int


class VideoSource:
    def __init__(
        self,
        source: VideoInput = TEMP_DEFAULT_VIDEO_PATH,
        camera_id: str = "factory-floor-demo",
        loop: bool = False,
    ) -> None:
        self.source = source
        self.camera_id = camera_id
        self.loop = loop
        self._capture: cv2.VideoCapture | None = None
        self._frame_index = 0
        self._fps = 0.0
        self._total_frames = 0

    @property
    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def open(self) -> None:
        if self.is_opened:
            return

        capture_source: str | int
        if isinstance(self.source, int):
            capture_source = self.source
        else:
            path = Path(self.source).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Video file not found: {path}")
            capture_source = str(path)

        capture = cv2.VideoCapture(capture_source)
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"Unable to open video source: {self.source}")

        self._capture = capture
        self._frame_index = 0
        self._fps = max(0.0, float(capture.get(cv2.CAP_PROP_FPS)))
        self._total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    def read(self) -> FramePacket | None:
        if not self.is_opened:
            self.open()

        capture = self._require_capture()
        success, frame = capture.read()

        if not success and self.loop:
            self.reset()
            success, frame = capture.read()

        if not success or frame is None:
            return None

        height, width = frame.shape[:2]
        timestamp = self._read_timestamp(capture)
        packet = FramePacket(
            camera_id=self.camera_id,
            frame=frame,
            frame_index=self._frame_index,
            timestamp=timestamp,
            fps=self._fps,
            width=width,
            height=height,
        )
        self._frame_index += 1
        return packet

    def reset(self) -> None:
        capture = self._require_capture()
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
            raise VideoSourceError(f"Unable to reset video source: {self.source}")
        self._frame_index = 0

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __iter__(self) -> Iterator[FramePacket]:
        while (packet := self.read()) is not None:
            yield packet

    def __enter__(self) -> VideoSource:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _read_timestamp(self, capture: cv2.VideoCapture) -> float:
        position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        if position_ms > 0.0:
            return position_ms / 1000.0
        if self._fps > 0.0:
            return self._frame_index / self._fps
        return 0.0

    def _require_capture(self) -> cv2.VideoCapture:
        if self._capture is None or not self._capture.isOpened():
            raise VideoSourceError("Video source is not open")
        return self._capture
