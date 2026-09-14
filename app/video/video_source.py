"""영상 입력을 공통 FramePacket 형식으로 변환하는 모듈."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TEMP_DEFAULT_VIDEO_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "samples" / "factory_floor_demo.mp4"
)
# OpenCV는 파일 경로와 카메라 번호를 같은 VideoCapture 인터페이스로 처리한다.
VideoInput = str | Path | int


class VideoSourceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FramePacket:
    # 프레임과 함께 카메라 ID, 시간, 크기를 묶어 다음 처리 단계의 입력 계약을 고정한다.
    # frozen=True는 프레임 메타데이터가 처리 중 임의로 바뀌는 일을 막기 위한 선택이다.
    # 단, numpy 배열 자체는 mutable이므로 화면 표시용 렌더링은 원본을 copy해서 처리한다.
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
            # 정수 source는 OpenCV 카메라 인덱스다. 노트북 기본 카메라는 보통 0번이다.
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
        # 일부 카메라나 코덱은 FPS/프레임 수를 0 또는 잘못된 값으로 줄 수 있으므로 음수는 막는다.
        self._fps = max(0.0, float(capture.get(cv2.CAP_PROP_FPS)))
        self._total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    def read(self) -> FramePacket | None:
        if not self.is_opened:
            self.open()

        capture = self._require_capture()
        success, frame = capture.read()

        if not success and self.loop:
            # 데모 영상 반복 재생용 경로다. 실시간 카메라는 rewind 개념이 없으므로 별도 RTSP 소스에서 다룬다.
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
            # 일부 입력은 POS_MSEC를 제공하지 않는다. 이때 frame_index와 FPS로 재현 가능한 시간을 만든다.
            return self._frame_index / self._fps
        return 0.0

    def _require_capture(self) -> cv2.VideoCapture:
        if self._capture is None or not self._capture.isOpened():
            raise VideoSourceError("Video source is not open")
        return self._capture
