"""객체 추적과 track ID 유지를 검증하는 테스트."""

import numpy as np

from app.inference.detector import Detection
from app.tracking.tracker import SimpleTracker, TrackerConfig
from app.video.video_source import FramePacket


def _frame_packet(frame_index: int, timestamp: float) -> FramePacket:
    return FramePacket(
        camera_id="test-camera",
        frame=np.zeros((100, 100, 3), dtype=np.uint8),
        frame_index=frame_index,
        timestamp=timestamp,
        fps=30.0,
        width=100,
        height=100,
    )


def test_simple_tracker_keeps_track_id_for_overlapping_detection() -> None:
    tracker = SimpleTracker(
        TrackerConfig(track_threshold=0.5, match_threshold=0.3)
    )

    first_tracks = tracker.update(
        detections=[
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=(10.0, 10.0, 40.0, 60.0),
            )
        ],
        frame_packet=_frame_packet(frame_index=0, timestamp=0.0),
    )
    second_tracks = tracker.update(
        detections=[
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.88,
                bbox=(12.0, 12.0, 42.0, 62.0),
            )
        ],
        frame_packet=_frame_packet(frame_index=1, timestamp=1.0),
    )

    assert first_tracks[0].track_id == second_tracks[0].track_id
    assert second_tracks[0].center == (27.0, 37.0)
    assert second_tracks[0].bottom_center == (27.0, 62.0)
    assert second_tracks[0].timestamp == 1.0


def test_simple_tracker_filters_low_confidence_detection() -> None:
    tracker = SimpleTracker(TrackerConfig(track_threshold=0.5))

    tracks = tracker.update(
        detections=[
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.49,
                bbox=(10.0, 10.0, 40.0, 60.0),
            )
        ],
        frame_packet=_frame_packet(frame_index=0, timestamp=0.0),
    )

    assert tracks == []


def test_simple_tracker_reset_restarts_track_ids() -> None:
    tracker = SimpleTracker()
    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=(10.0, 10.0, 40.0, 60.0),
    )

    first_tracks = tracker.update(
        detections=[detection],
        frame_packet=_frame_packet(frame_index=0, timestamp=0.0),
    )
    tracker.reset()
    second_tracks = tracker.update(
        detections=[detection],
        frame_packet=_frame_packet(frame_index=1, timestamp=1.0),
    )

    assert first_tracks[0].track_id == 1
    assert second_tracks[0].track_id == 1
