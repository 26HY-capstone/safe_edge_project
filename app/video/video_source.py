"""영상 입력을 공통 FramePacket 형식으로 변환하는 모듈."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMERA_CONFIG_PATH = PROJECT_ROOT / "config" / "cameras.yaml"
TEMP_DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data" / "samples" / "factory_floor_demo.mp4"
# OpenCV VideoCapture 입력: 파일 경로 또는 카메라 번호.
VideoInput = str | Path | int


class VideoSourceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FramePacket:
    # FramePacket: 프레임, 카메라 ID, 시간, 크기를 묶은 다음 처리 단계 입력 계약.
    # frozen=True: 프레임 메타데이터 변경 방지.
    # numpy 배열은 mutable이므로 화면 표시용 렌더링은 원본 copy 사용.
    camera_id: str
    frame: np.ndarray
    frame_index: int
    timestamp: float
    fps: float
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class CameraConfig:
    camera_id: str
    name: str
    camera_type: str
    source: VideoInput
    loop: bool = False
    width: int | None = None
    height: int | None = None
    fps: float | None = None


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
            # 정수 source: OpenCV 카메라 인덱스. 일반적인 내장 카메라 기본값은 0번.
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
        # FPS/프레임 수 메타데이터가 0 또는 잘못된 값으로 제공되는 입력이 있다.
        self._fps = max(0.0, float(capture.get(cv2.CAP_PROP_FPS)))
        self._total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    def read(self) -> FramePacket | None:
        if not self.is_opened:
            self.open()

        capture = self._require_capture()
        success, frame = capture.read()

        if not success and self.loop:
            # loop 재생은 파일 입력 전용 흐름이다. 실시간 카메라는 별도 RTSP 소스에서 처리한다.
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
            # POS_MSEC가 없는 입력의 timestamp fallback: frame_index / FPS.
            return self._frame_index / self._fps
        return 0.0

    def _require_capture(self) -> cv2.VideoCapture:
        if self._capture is None or not self._capture.isOpened():
            raise VideoSourceError("Video source is not open")
        return self._capture


def load_camera_configs(
    config_path: str | Path = DEFAULT_CAMERA_CONFIG_PATH,
) -> list[CameraConfig]:
    """cameras.yaml에서 카메라 입력 설정 목록을 읽는다."""
    resolved_config_path = _resolve_project_path(config_path)

    with resolved_config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, Mapping):
        raise ValueError("camera config must be a mapping")

    cameras = config.get("cameras", [])
    if not isinstance(cameras, list):
        raise ValueError("cameras must be a list")

    return [_camera_config_from_mapping(camera) for camera in cameras]


def get_camera_config(
    camera_id: str | None = None,
    config_path: str | Path = DEFAULT_CAMERA_CONFIG_PATH,
) -> CameraConfig:
    """camera_id가 지정되면 해당 카메라를, 없으면 첫 번째 카메라를 반환한다."""
    cameras = load_camera_configs(config_path)
    if not cameras:
        raise ValueError("cameras config must contain at least one camera")

    if camera_id is None:
        return cameras[0]

    for camera in cameras:
        if camera.camera_id == camera_id:
            return camera
    raise ValueError(f"Camera not found: {camera_id}")


def create_video_source_from_config(camera_config: CameraConfig) -> VideoSource:
    """CameraConfig를 OpenCV 기반 VideoSource로 변환한다."""
    if camera_config.camera_type not in {"video", "webcam"}:
        raise ValueError(
            f"Unsupported VideoSource camera type: {camera_config.camera_type}"
        )

    source = camera_config.source
    if isinstance(source, Path):
        source = _resolve_project_path(source)

    return VideoSource(
        source=source,
        camera_id=camera_config.camera_id,
        loop=camera_config.loop,
    )


def create_video_source_from_cameras_config(
    config_path: str | Path = DEFAULT_CAMERA_CONFIG_PATH,
    camera_id: str | None = None,
) -> VideoSource:
    """cameras.yaml의 선택된 카메라 설정으로 VideoSource를 생성한다."""
    camera_config = get_camera_config(camera_id=camera_id, config_path=config_path)
    return create_video_source_from_config(camera_config)


def _camera_config_from_mapping(camera: object) -> CameraConfig:
    """YAML의 단일 camera 항목을 CameraConfig 데이터 계약으로 변환한다."""
    if not isinstance(camera, Mapping):
        raise ValueError("camera entry must be a mapping")

    camera_id = _required_str(camera, "camera_id")
    camera_type = _required_str(camera, "type")
    source = _required_source(camera, camera_type)

    return CameraConfig(
        camera_id=camera_id,
        name=str(camera.get("name", camera_id)),
        camera_type=camera_type,
        source=source,
        loop=bool(camera.get("loop", False)),
        width=_optional_int(camera.get("width")),
        height=_optional_int(camera.get("height")),
        fps=_optional_float(camera.get("fps")),
    )


def _required_str(values: Mapping[str, Any], key: str) -> str:
    """필수 문자열 설정값을 공백 제거 후 반환한다."""
    value = str(values.get(key, "")).strip()
    if not value:
        raise ValueError(f"{key} must not be empty")
    return value


def _required_source(values: Mapping[str, Any], camera_type: str) -> VideoInput:
    """카메라 type에 맞는 OpenCV 입력값을 반환한다."""
    if "source" not in values:
        raise ValueError("source must be configured")

    source = values["source"]
    if camera_type == "webcam":
        if isinstance(source, bool):
            raise ValueError("webcam source must be an integer camera index")
        if isinstance(source, int):
            return source
        if isinstance(source, str):
            return int(source)
        raise ValueError("webcam source must be an integer camera index")
    if camera_type == "video":
        return Path(str(source))
    return str(source)


def _optional_int(value: object) -> int | None:
    """설정값이 있으면 int로 변환하고 없으면 None을 유지한다."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("integer config value must not be boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    raise ValueError("integer config value must be int, float, or str")


def _optional_float(value: object) -> float | None:
    """설정값이 있으면 float로 변환하고 없으면 None을 유지한다."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("float config value must not be boolean")
    if isinstance(value, float):
        return value
    if isinstance(value, int | str):
        return float(value)
    raise ValueError("float config value must be int, float, or str")


def _resolve_project_path(path: str | Path) -> Path:
    """상대경로를 현재 실행 위치가 아니라 프로젝트 루트 기준으로 해석한다."""
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute():
        return resolved_path
    if resolved_path.is_file():
        return resolved_path
    return PROJECT_ROOT / resolved_path
