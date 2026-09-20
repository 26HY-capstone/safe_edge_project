"""설정 및 샘플 영상을 최대 네 개까지 2x2로 표시하는 데모."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from app.video.sample_dataset import create_random_ceiling_eye_video_sources
from app.video.video_source import (
    CameraConfig,
    VideoSource,
    VideoSourceError,
    create_video_source_from_config,
    load_camera_configs,
)

WINDOW_NAME = "Vision Guard Multi-Camera Viewer"
TILE_WIDTH = 640
TILE_HEIGHT = 360
MAX_VIEW_COUNT = 4
LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--sample-dir", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    camera_configs = (
        load_camera_configs(args.camera_config)
        if args.camera_config is not None
        else load_camera_configs()
    )
    video_sources = create_display_video_sources(
        camera_configs=camera_configs,
        sample_dir=args.sample_dir,
        seed=args.seed,
    )

    try:
        while True:
            frames = [_read_display_frame(source) for source in video_sources]
            cv2.imshow(WINDOW_NAME, compose_2x2_grid(frames))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for source in video_sources:
            source.close()
        cv2.destroyAllWindows()


def create_display_video_sources(
    camera_configs: list[CameraConfig],
    sample_dir: Path | None = None,
    seed: int | None = None,
) -> list[VideoSource]:
    sample_sources = _create_sample_video_sources(sample_dir, seed)
    remaining = MAX_VIEW_COUNT - len(sample_sources)
    camera_sources = _create_video_sources(camera_configs[:remaining])
    return [*camera_sources, *sample_sources][:MAX_VIEW_COUNT]


def compose_2x2_grid(frames: list[np.ndarray]) -> np.ndarray:
    tiles = [_resize_tile(frame) for frame in frames[:MAX_VIEW_COUNT]]
    while len(tiles) < MAX_VIEW_COUNT:
        tiles.append(_make_blank_tile("empty"))
    return np.vstack(
        [
            np.hstack([tiles[0], tiles[1]]),
            np.hstack([tiles[2], tiles[3]]),
        ]
    )


def _create_sample_video_sources(
    sample_dir: Path | None,
    seed: int | None,
) -> list[VideoSource]:
    if sample_dir is None:
        return []
    try:
        return create_random_ceiling_eye_video_sources(
            sample_dir=sample_dir,
            seed=seed,
            loop=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.warning("Unable to load sample dataset: %s", exc)
        return []


def _create_video_sources(
    camera_configs: list[CameraConfig],
) -> list[VideoSource]:
    sources: list[VideoSource] = []
    for camera_config in camera_configs:
        try:
            sources.append(create_video_source_from_config(camera_config))
        except ValueError as exc:
            LOGGER.warning(
                "Skipping camera %s: %s",
                camera_config.camera_id,
                exc,
            )
    return sources


def _read_display_frame(video_source: VideoSource) -> np.ndarray:
    try:
        frame_packet = video_source.read()
    except (FileNotFoundError, VideoSourceError) as exc:
        LOGGER.warning("Camera %s unavailable: %s", video_source.camera_id, exc)
        return _make_blank_tile(f"{video_source.camera_id}: no input")
    if frame_packet is None:
        return _make_blank_tile(f"{video_source.camera_id}: no frame")
    return _draw_camera_label(
        frame=_resize_tile(frame_packet.frame),
        label=video_source.camera_id,
    )


def _resize_tile(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (TILE_WIDTH, TILE_HEIGHT))


def _make_blank_tile(label: str) -> np.ndarray:
    frame = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
    return _draw_camera_label(frame=frame, label=label)


def _draw_camera_label(frame: np.ndarray, label: str) -> np.ndarray:
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
    logging.basicConfig(level=logging.INFO)
    main()
