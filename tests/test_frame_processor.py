"""프레임 처리 단계에서 detection과 tracking 연결을 검증하는 테스트."""

import numpy as np

from app.inference.detector import Detection, Detector
from app.tracking.tracker import SimpleTracker
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


class _SingleFrameSource:
    def __init__(self) -> None:
        self._has_frame = True

    def read(self) -> FramePacket | None:
        if not self._has_frame:
            return None

        self._has_frame = False
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        return FramePacket(
            camera_id="test-camera",
            frame=frame,
            frame_index=0,
            timestamp=0.0,
            fps=30.0,
            width=100,
            height=100,
        )


def test_frame_processor_returns_tracked_objects() -> None:
    processor = FrameProcessor(
        video_source=_SingleFrameSource(),
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
