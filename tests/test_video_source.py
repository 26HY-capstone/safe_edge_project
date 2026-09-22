"""카메라 설정 파일을 VideoSource 입력으로 변환하는 테스트."""

from pathlib import Path

import pytest

from app.video.video_source import (
    CameraConfig,
    PROJECT_ROOT,
    VideoSource,
    create_video_source_from_cameras_config,
    create_video_source_from_config,
    get_camera_config,
    load_camera_configs,
)


def _write_cameras_config(path: Path) -> None:
    path.write_text(
        """
cameras:
  - camera_id: demo-video
    name: Demo Video
    type: video
    source: data/samples/demo.mp4
    loop: true
  - camera_id: laptop-camera
    name: Laptop Camera
    type: webcam
    source: 0
    width: 1280
    height: 720
    fps: 30
    loop: false
""",
        encoding="utf-8",
    )


def test_load_camera_configs_reads_video_and_webcam_sources(tmp_path) -> None:
    config_path = tmp_path / "cameras.yaml"
    _write_cameras_config(config_path)

    cameras = load_camera_configs(config_path)

    assert cameras[0] == CameraConfig(
        camera_id="demo-video",
        name="Demo Video",
        camera_type="video",
        source=Path("data/samples/demo.mp4"),
        loop=True,
    )
    assert cameras[1].camera_id == "laptop-camera"
    assert cameras[1].source == 0
    assert cameras[1].width == 1280
    assert cameras[1].height == 720
    assert cameras[1].fps == 30.0


def test_get_camera_config_returns_requested_camera(tmp_path) -> None:
    config_path = tmp_path / "cameras.yaml"
    _write_cameras_config(config_path)

    camera = get_camera_config(
        camera_id="laptop-camera",
        config_path=config_path,
    )

    assert camera.camera_id == "laptop-camera"
    assert camera.source == 0


def test_create_video_source_from_cameras_config_uses_first_camera(tmp_path) -> None:
    config_path = tmp_path / "cameras.yaml"
    _write_cameras_config(config_path)

    video_source = create_video_source_from_cameras_config(config_path)

    assert isinstance(video_source, VideoSource)
    assert video_source.camera_id == "demo-video"
    assert video_source.source == PROJECT_ROOT / "data/samples/demo.mp4"
    assert video_source.loop is True


def test_create_video_source_rejects_rtsp_until_rtsp_source_is_implemented() -> None:
    camera_config = CameraConfig(
        camera_id="rtsp-camera",
        name="RTSP Camera",
        camera_type="rtsp",
        source="rtsp://example/stream",
    )

    with pytest.raises(ValueError, match="Unsupported VideoSource camera type"):
        create_video_source_from_config(camera_config)
