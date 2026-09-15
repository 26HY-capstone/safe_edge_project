"""프레임 처리 단계에서 detection과 tracking 연결을 검증하는 테스트."""

import numpy as np

from app.inference.detector import Detection, Detector
from app.tracking.tracker import SimpleTracker
from app.tracking.trajectory import TrajectoryAnalyzer
from app.video.frame_processor import FrameProcessor
from app.video.video_source import FramePacket


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
