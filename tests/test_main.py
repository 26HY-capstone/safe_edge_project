"""main 실행 입력 구성 로직을 검증하는 테스트."""

from app import main
from app.video.video_source import CameraConfig, VideoSource


def test_create_display_video_sources_puts_sample_sources_on_bottom(
    monkeypatch,
) -> None:
    sample_sources = [
        VideoSource(source=0, camera_id="sample-ceiling"),
        VideoSource(source=1, camera_id="sample-eye"),
    ]
    monkeypatch.setattr(main, "_create_sample_video_sources", lambda: sample_sources)

    video_sources = main._create_display_video_sources(
        [
            CameraConfig(
                camera_id="laptop-camera",
                name="Laptop Camera",
                camera_type="webcam",
                source=2,
            )
        ]
    )

    assert [source.camera_id for source in video_sources] == [
        "laptop-camera",
        "sample-ceiling",
        "sample-eye",
    ]


def test_create_display_video_sources_limits_to_four(monkeypatch) -> None:
    sample_sources = [
        VideoSource(source=0, camera_id="sample-ceiling"),
        VideoSource(source=1, camera_id="sample-eye"),
    ]
    monkeypatch.setattr(main, "_create_sample_video_sources", lambda: sample_sources)

    video_sources = main._create_display_video_sources(
        [
            CameraConfig(
                camera_id=f"camera-{index}",
                name=f"Camera {index}",
                camera_type="webcam",
                source=index + 2,
            )
            for index in range(4)
        ]
    )

    assert len(video_sources) == 4
    assert [source.camera_id for source in video_sources] == [
        "camera-0",
        "camera-1",
        "sample-ceiling",
        "sample-eye",
    ]


def test_create_processing_components_uses_per_source_state(monkeypatch) -> None:
    fake_detector = object()
    created_trackers = []

    def create_tracker() -> object:
        tracker = object()
        created_trackers.append(tracker)
        return tracker

    monkeypatch.setattr(main, "_create_detector", lambda: fake_detector)
    monkeypatch.setattr(main, "_create_tracker", create_tracker)

    components = main._create_processing_components(
        [
            VideoSource(source=0, camera_id="camera-0"),
            VideoSource(source=1, camera_id="camera-1"),
        ]
    )

    assert components.detector is fake_detector
    assert set(components.trackers) == {"camera-0", "camera-1"}
    assert list(components.trackers.values()) == created_trackers
    assert set(components.trajectory_analyzers) == {"camera-0", "camera-1"}
    assert (
        components.trajectory_analyzers["camera-0"]
        is not components.trajectory_analyzers["camera-1"]
    )
