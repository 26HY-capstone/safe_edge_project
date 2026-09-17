"""Vision Guard 실행 진입점."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import cv2
import numpy as np

from app.inference.detector import Detector
from app.inference.pytorch_backend import create_pytorch_detector_from_model_config
from app.tracking.tracker import Tracker, create_tracker_from_system_config
from app.tracking.trajectory import TrajectoryAnalyzer
from app.video.sample_dataset import create_random_ceiling_eye_video_sources
from app.video.video_source import (
    CameraConfig,
    PROJECT_ROOT,
    VideoSource,
    VideoSourceError,
    create_video_source_from_config,
    load_camera_configs,
)

WINDOW_NAME = "Vision Guard"
TILE_WIDTH = 640
TILE_HEIGHT = 360
MAX_VIEW_COUNT = 4
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "data" / "samples" / "forklift_human_nearmiss"
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"
SYSTEM_CONFIG_PATH = PROJECT_ROOT / "config" / "system.yaml"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessingComponents:
    detector: Detector | None
    trackers: dict[str, Tracker]
    trajectory_analyzers: dict[str, TrajectoryAnalyzer]


def main() -> None:
    """설정된 영상 입력을 최대 4개까지 한 창의 2x2 화면으로 표시한다."""
    camera_configs = load_camera_configs()
    video_sources = _create_display_video_sources(camera_configs)
    _processing_components = _create_processing_components(video_sources)

    try:
        while True:
            frames = [_read_display_frame(video_source) for video_source in video_sources]
            display_frame = _compose_2x2_grid(frames)

            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for video_source in video_sources:
            video_source.close()
        cv2.destroyAllWindows()


def _create_processing_components(
    video_sources: list[VideoSource],
) -> ProcessingComponents:
    """입력별 tracking 상태와 trajectory analyzer를 생성한다."""
    detector = _create_detector()
    trackers = {
        video_source.camera_id: _create_tracker()
        for video_source in video_sources
    }
    trajectory_analyzers = {
        video_source.camera_id: TrajectoryAnalyzer()
        for video_source in video_sources
    }
    return ProcessingComponents(
        detector=detector,
        trackers=trackers,
        trajectory_analyzers=trajectory_analyzers,
    )


def _create_detector() -> Detector | None:
    """model.yaml 기반 detector를 생성하고 실패 시 화면 출력만 계속 가능하게 한다."""
    try:
        return create_pytorch_detector_from_model_config(MODEL_CONFIG_PATH)
    except Exception as exc:
        logger.warning("Detector is disabled: %s", exc)
        return None


def _create_tracker() -> Tracker:
    """system.yaml 기반 tracker를 생성한다."""
    return create_tracker_from_system_config(SYSTEM_CONFIG_PATH)


def _create_display_video_sources(
    camera_configs: list[CameraConfig],
) -> list[VideoSource]:
    """샘플 ceiling/eye 영상과 설정 기반 입력을 4분할 화면 입력으로 구성한다."""
    sample_sources = _create_sample_video_sources()
    camera_source_count = MAX_VIEW_COUNT - len(sample_sources)
    camera_sources = _create_video_sources(camera_configs[:camera_source_count])
    return [*camera_sources, *sample_sources][:MAX_VIEW_COUNT]


def _create_sample_video_sources() -> list[VideoSource]:
    """샘플 metadata에서 ceiling/eye 영상 쌍을 생성한다."""
    if not DEFAULT_SAMPLE_DIR.is_dir():
        return []

    try:
        return create_random_ceiling_eye_video_sources(
            sample_dir=DEFAULT_SAMPLE_DIR,
            loop=True,
        )
    except (FileNotFoundError, ValueError):
        return []


def _create_video_sources(camera_configs: list[CameraConfig]) -> list[VideoSource]:
    """지원되는 카메라 설정만 VideoSource로 변환한다."""
    video_sources: list[VideoSource] = []

    for camera_config in camera_configs:
        try:
            video_sources.append(create_video_source_from_config(camera_config))
        except ValueError:
            # RTSP처럼 아직 별도 source 구현이 필요한 입력은 현재 4분할 화면에서 제외한다.
            continue

    return video_sources


def _read_display_frame(video_source: VideoSource) -> np.ndarray:
    """단일 입력에서 프레임을 읽고 4분할 타일 크기로 변환한다."""
    try:
        frame_packet = video_source.read()
    except (FileNotFoundError, VideoSourceError):
        return _make_blank_tile(f"{video_source.camera_id}: no input")

    if frame_packet is None:
        return _make_blank_tile(f"{video_source.camera_id}: no frame")

    frame = _resize_tile(frame_packet.frame)
    return _draw_camera_label(frame=frame, label=video_source.camera_id)


def _compose_2x2_grid(frames: list[np.ndarray]) -> np.ndarray:
    """최대 4개의 프레임을 2x2 격자 이미지로 합친다."""
    tiles = [_resize_tile(frame) for frame in frames[:MAX_VIEW_COUNT]]

    while len(tiles) < MAX_VIEW_COUNT:
        tiles.append(_make_blank_tile("empty"))

    top_row = np.hstack([tiles[0], tiles[1]])
    bottom_row = np.hstack([tiles[2], tiles[3]])
    return np.vstack([top_row, bottom_row])


def _resize_tile(frame: np.ndarray) -> np.ndarray:
    """입력 프레임을 고정 크기 타일로 맞춘다."""
    return cv2.resize(frame, (TILE_WIDTH, TILE_HEIGHT))


def _make_blank_tile(label: str) -> np.ndarray:
    """프레임이 없는 입력을 표시하기 위한 빈 타일을 만든다."""
    frame = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
    return _draw_camera_label(frame=frame, label=label)


def _draw_camera_label(frame: np.ndarray, label: str) -> np.ndarray:
    """타일 왼쪽 위에 카메라 식별자를 표시한다."""
    cv2.putText(
        frame,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return frame


if __name__ == "__main__":
    main()
