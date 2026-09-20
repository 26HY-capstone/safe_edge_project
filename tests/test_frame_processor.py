"""프레임 처리 단계에서 detection과 tracking 연결을 검증하는 테스트."""

import numpy as np

from app.inference.detector import Detection, Detector
from app.alerts.event_manager import EventManager, RiskEventTransition
from app.risk.models import RiskLevel
from app.risk.risk_engine import RiskEngine
from app.tracking.tracker import SimpleTracker, Tracker, TrackedObject
from app.tracking.trajectory import TrajectoryAnalyzer
from app.video.frame_processor import FrameProcessor
from app.video.video_source import FramePacket
from app.zones.models import Zone
from app.zones.zone_manager import ZoneManager


class _StaticDetector(Detector):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.validate_frame(frame)
        return [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=(10.0, 10.0, 40.0, 60.0),
            )
        ]


class _SequenceDetector(Detector):
    def __init__(self) -> None:
        self._next_x1 = 10.0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.validate_frame(frame)
        x1 = self._next_x1
        self._next_x1 += 3.0
        return [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=(x1, 10.0, x1 + 30.0, 60.0),
            )
        ]


class _FrameSource:
    def __init__(self) -> None:
        self._frame_index = 0

    def read(self) -> FramePacket | None:
        if self._frame_index >= 2:
            return None

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        packet = FramePacket(
            camera_id="test-camera",
            frame=frame,
            frame_index=self._frame_index,
            timestamp=float(self._frame_index),
            fps=30.0,
            width=100,
            height=100,
        )
        self._frame_index += 1
        return packet


class _RecordingAlertManager:
    def __init__(self) -> None:
        self.events = []

    def handle(self, events) -> None:
        self.events.extend(events)


class _EpochSource:
    def __init__(self) -> None:
        self._packets = [
            FramePacket(
                camera_id="test-camera",
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                frame_index=10,
                timestamp=10.0,
                fps=30.0,
                width=100,
                height=100,
                stream_epoch=0,
            ),
            FramePacket(
                camera_id="test-camera",
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                frame_index=0,
                timestamp=0.0,
                fps=30.0,
                width=100,
                height=100,
                stream_epoch=1,
            ),
        ]

    def read(self) -> FramePacket | None:
        return self._packets.pop(0) if self._packets else None


class _ResetSpyTracker(Tracker):
    def __init__(self) -> None:
        self.reset_count = 0

    def update(
        self,
        detections: list[Detection],
        frame_packet: FramePacket,
    ) -> list[TrackedObject]:
        return []

    def reset(self) -> None:
        self.reset_count += 1


def test_frame_processor_returns_tracked_objects() -> None:
    processor = FrameProcessor(
        video_source=_FrameSource(),
        detector=_StaticDetector(),
        tracker=SimpleTracker(),
        draw_bbox=False,
        draw_metrics=False,
    )

    processed_frame = processor.process_next()

    assert processed_frame is not None
    assert len(processed_frame.detections) == 1
    assert len(processed_frame.tracked_objects) == 1
    assert processed_frame.tracked_objects[0].track_id == 1
    assert processed_frame.tracked_objects[0].class_name == "person"
    assert processed_frame.rendered_frame is None


def test_frame_processor_returns_motion_summaries() -> None:
    processor = FrameProcessor(
        video_source=_FrameSource(),
        detector=_SequenceDetector(),
        tracker=SimpleTracker(),
        trajectory_analyzer=TrajectoryAnalyzer(),
        draw_bbox=False,
        draw_metrics=False,
    )

    first_frame = processor.process_next()
    second_frame = processor.process_next()

    assert first_frame is not None
    assert second_frame is not None
    assert second_frame.motion_summaries[1].distance_px == 3.0
    assert second_frame.motion_summaries[1].direction == (1.0, 0.0)
    assert second_frame.motion_summaries[1].speed_px_per_sec == 3.0


def test_frame_processor_runs_zone_risk_event_and_alert_pipeline() -> None:
    alert_manager = _RecordingAlertManager()
    processor = FrameProcessor(
        video_source=_FrameSource(),
        detector=_StaticDetector(),
        tracker=SimpleTracker(),
        zone_manager=ZoneManager(
            zones=[
                Zone(
                    zone_id="danger",
                    camera_id="test-camera",
                    zone_type="static_danger",
                    polygon=(
                        (0.0, 0.0),
                        (80.0, 0.0),
                        (80.0, 80.0),
                        (0.0, 80.0),
                    ),
                    risk_level="warning",
                )
            ]
        ),
        risk_engine=RiskEngine(),
        event_manager=EventManager(),
        alert_manager=alert_manager,
        draw_bbox=False,
        draw_metrics=False,
        draw_zones=False,
        draw_risks=False,
    )

    result = processor.process_next()

    assert result is not None
    assert result.risk_assessments[0].risk_level == RiskLevel.WARNING
    assert result.risk_events[0].transition == RiskEventTransition.CREATED
    assert alert_manager.events == result.risk_events


def test_frame_processor_resets_temporal_state_on_stream_epoch_change() -> None:
    tracker = _ResetSpyTracker()
    processor = FrameProcessor(
        video_source=_EpochSource(),
        detector=_StaticDetector(),
        tracker=tracker,
        draw_bbox=False,
        draw_metrics=False,
        draw_zones=False,
        draw_risks=False,
    )

    assert processor.process_next() is not None
    assert processor.process_next() is not None
    assert tracker.reset_count == 1
