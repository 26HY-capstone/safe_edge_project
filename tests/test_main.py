"""main 실행 입력 구성 로직을 검증하는 테스트."""

from dataclasses import dataclass

import numpy as np

from app import main
from app.inference.detector import Detector
from app.tracking.tracker import SimpleTracker
from app.tracking.trajectory import TrajectoryAnalyzer
from app.video.video_source import CameraConfig, FramePacket, VideoSource


class _FakeDetector(Detector):
    def detect(self, frame):
        return []


@dataclass(slots=True)
class _FakeProcessedFrame:
    frame_packet: FramePacket
    rendered_frame: np.ndarray | None


class _FakeProcessor:
    def __init__(self, processed_frame: _FakeProcessedFrame | None) -> None:
        self.processed_frame = processed_frame

    def process_next(self) -> _FakeProcessedFrame | None:
        return self.processed_frame


def _frame_packet() -> FramePacket:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    return FramePacket(
        camera_id="camera-0",
        frame=frame,
        frame_index=0,
        timestamp=0.0,
        fps=30.0,
        width=100,
        height=100,
    )


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


def test_create_frame_processor_returns_none_without_detector() -> None:
    video_source = VideoSource(source=0, camera_id="camera-0")
    components = main.ProcessingComponents(
        detector=None,
        trackers={},
        trajectory_analyzers={},
    )

    processor = main._create_frame_processor(
        video_source=video_source,
        processing_components=components,
    )

    assert processor is None


def test_create_frame_processor_uses_matching_source_state() -> None:
    video_source = VideoSource(source=0, camera_id="camera-0")
    tracker = SimpleTracker()
    trajectory_analyzer = TrajectoryAnalyzer()
    components = main.ProcessingComponents(
        detector=_FakeDetector(),
        trackers={"camera-0": tracker},
        trajectory_analyzers={"camera-0": trajectory_analyzer},
    )

    processor = main._create_frame_processor(
        video_source=video_source,
        processing_components=components,
    )

    assert processor is not None
    assert processor.video_source is video_source
    assert processor.detector is components.detector
    assert processor.tracker is tracker
    assert processor.trajectory_analyzer is trajectory_analyzer


def test_toggle_camera_view_closes_source_when_disabled(monkeypatch) -> None:
    video_source = VideoSource(source=0, camera_id="camera-0")
    camera_view = main.CameraViewState(video_source=video_source)
    closed = []

    monkeypatch.setattr(video_source, "close", lambda: closed.append(True))

    main._toggle_camera_view([camera_view], "camera-0")

    assert camera_view.enabled is False
    assert closed == [True]

    main._toggle_camera_view([camera_view], "camera-0")

    assert camera_view.enabled is True
    assert closed == [True]


def test_compose_2x2_grid_returns_power_button_bounds() -> None:
    frame = main._make_blank_tile("camera-0")
    display_frame, button_bounds = main._compose_2x2_grid(
        frames=[frame],
        camera_views=[
            main.CameraViewState(
                video_source=VideoSource(source=0, camera_id="camera-0")
            )
        ],
    )

    assert display_frame.shape == (
        main.TILE_HEIGHT * 2,
        main.TILE_WIDTH * 2,
        3,
    )
    assert len(button_bounds) == 1
    assert button_bounds[0].camera_id == "camera-0"
    assert button_bounds[0].contains(main.TILE_WIDTH - 60, 30)


def test_read_display_frame_uses_processor_rendered_frame() -> None:
    rendered_frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    camera_view = main.CameraViewState(
        video_source=VideoSource(source=0, camera_id="camera-0"),
        processor=_FakeProcessor(
            _FakeProcessedFrame(
                frame_packet=_frame_packet(),
                rendered_frame=rendered_frame,
            )
        ),
    )

    display_frame = main._read_display_frame(camera_view)

    assert display_frame.shape == (main.TILE_HEIGHT, main.TILE_WIDTH, 3)
    assert display_frame.mean() > 0
