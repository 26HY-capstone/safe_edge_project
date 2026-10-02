"""main 실행 입력 구성 로직을 검증하는 테스트."""

from dataclasses import dataclass, field

import numpy as np

from app import main
from app.alerts.event_manager import EventManager
from app.inference.detector import Detector
from app.risk.risk_engine import RiskEngine
from app.risk.risk_logger import RiskLogger
from app.tracking.tracker import SimpleTracker
from app.tracking.trajectory import TrajectoryAnalyzer
from app.video.video_source import CameraConfig, FramePacket, VideoSource
from app.zones.models import ZoneFrameResult
from app.zones.zone_manager import ZoneManager


class _FakeDetector(Detector):
    def detect(self, frame):
        return []


@dataclass(slots=True)
class _FakeProcessedFrame:
    frame_packet: FramePacket
    rendered_frame: np.ndarray | None
    # Risk Log 연결 이후 _read_display_frame()이 항상 읽는 필드.
    # 실제 ProcessedFrame.zone_result는 Optional이 아니므로 빈 ZoneFrameResult를
    # 기본값으로 둔다. 이 테스트는 렌더링 결과만 검증하므로 내용은 비워둔다.
    zone_result: ZoneFrameResult = field(
        default_factory=lambda: ZoneFrameResult(
            timestamp=0.0,
            frame_index=0,
            camera_id="camera-0",
            workers=[],
            equipments=[],
        )
    )
    risk_assessments: list = field(default_factory=list)


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


def test_assign_display_camera_ids_uses_slot_order() -> None:
    """video_sources 순서대로 cam_01부터 camera_id를 부여해야 한다."""
    video_sources = [
        VideoSource(source=0, camera_id="laptop-camera"),
        VideoSource(source=0, camera_id="factory-floor-demo"),
        VideoSource(source=0, camera_id="run-xyz-ceiling"),
        VideoSource(source=0, camera_id="run-xyz-eye"),
    ]

    main._assign_display_camera_ids(video_sources)

    assert [source.camera_id for source in video_sources] == [
        "cam_01",
        "cam_02",
        "cam_03",
        "cam_04",
    ]


def test_prepare_risk_logger_always_creates_four_log_files(tmp_path) -> None:
    """실제 video_sources가 4개보다 적어도 로그 파일 네 개가 모두 준비돼야 한다."""
    risk_logger = RiskLogger(log_dir=tmp_path)

    # _prepare_risk_logger()는 DISPLAY_CAMERA_IDS(고정 4개)만 사용하므로
    # 실제로 연결된 입력 개수와는 무관하다.
    main._prepare_risk_logger(risk_logger)

    created_files = sorted(path.name for path in tmp_path.iterdir())
    assert created_files == [
        "risk_log_cam_01.jsonl",
        "risk_log_cam_02.jsonl",
        "risk_log_cam_03.jsonl",
        "risk_log_cam_04.jsonl",
    ]
    assert all(path.read_text(encoding="utf-8") == "" for path in tmp_path.iterdir())


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
    assert isinstance(components.zone_manager, ZoneManager)
    assert isinstance(components.risk_engine, RiskEngine)


def test_create_frame_processor_returns_none_without_detector() -> None:
    video_source = VideoSource(source=0, camera_id="camera-0")
    components = main.ProcessingComponents(
        detector=None,
        trackers={},
        trajectory_analyzers={},
        zone_manager=ZoneManager(),
        risk_engine=RiskEngine(),
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
    zone_manager = ZoneManager()
    risk_engine = RiskEngine()
    components = main.ProcessingComponents(
        detector=_FakeDetector(),
        trackers={"camera-0": tracker},
        trajectory_analyzers={"camera-0": trajectory_analyzer},
        zone_manager=zone_manager,
        risk_engine=risk_engine,
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
    assert processor.zone_manager is zone_manager
    assert processor.risk_engine is risk_engine


def test_toggle_camera_view_closes_source_when_disabled(monkeypatch) -> None:
    video_source = VideoSource(source=0, camera_id="camera-0")
    camera_view = main.CameraViewState(video_source=video_source, enabled=True)
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
        enabled=True,
    )

    # risk_assessments가 비어 있어 RiskLogger.log()는 아무 파일도 쓰지 않는다.
    display_frame = main._read_display_frame(
        camera_view, RiskLogger(), EventManager(), now=0.0
    )

    assert display_frame.shape == (main.TILE_HEIGHT, main.TILE_WIDTH, 3)
    assert display_frame.mean() > 0
