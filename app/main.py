"""Vision Guard 실행 진입점."""

from __future__ import annotations

import cv2
import numpy as np

from app.video.video_source import (
    CameraConfig,
    VideoSource,
    VideoSourceError,
    create_video_source_from_config,
    load_camera_configs,
)

WINDOW_NAME = "Vision Guard"
TILE_WIDTH = 640
TILE_HEIGHT = 360
MAX_VIEW_COUNT = 4


def main() -> None:
    """설정된 영상 입력을 최대 4개까지 한 창의 2x2 화면으로 표시한다."""
    camera_configs = load_camera_configs()
    video_sources = _create_video_sources(camera_configs[:MAX_VIEW_COUNT])

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
