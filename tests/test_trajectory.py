"""Track별 trajectory와 이동 요약을 검증하는 테스트."""

import numpy as np

from app.tracking.tracker import TrackedObject
from app.tracking.trajectory import TrajectoryAnalyzer, TrajectoryConfig
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


def _tracked_object(
    track_id: int,
    bottom_center: tuple[float, float],
    timestamp: float,
) -> TrackedObject:
    x, y = bottom_center
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name="person",
        bbox=(x - 5.0, y - 10.0, x + 5.0, y),
        center=(x, y - 5.0),
        bottom_center=bottom_center,
        confidence=0.9,
        timestamp=timestamp,
    )


def test_trajectory_analyzer_calculates_motion_summary() -> None:
    analyzer = TrajectoryAnalyzer()

    analyzer.update(
        tracked_objects=[
            _tracked_object(track_id=1, bottom_center=(0.0, 0.0), timestamp=0.0)
        ],
        frame_packet=_frame_packet(frame_index=0, timestamp=0.0),
    )
    summaries = analyzer.update(
        tracked_objects=[
            _tracked_object(track_id=1, bottom_center=(3.0, 4.0), timestamp=1.0)
        ],
        frame_packet=_frame_packet(frame_index=1, timestamp=1.0),
    )

    summary = summaries[1]
    assert summary.distance_px == 5.0
    assert summary.displacement_px == 5.0
    assert summary.speed_px_per_sec == 5.0
    assert summary.direction == (0.6, 0.8)
    assert summary.latest_position == (3.0, 4.0)


def test_trajectory_analyzer_calculates_stationary_seconds() -> None:
    analyzer = TrajectoryAnalyzer(
        TrajectoryConfig(stationary_distance_threshold_px=2.0)
    )

    analyzer.update(
        tracked_objects=[
            _tracked_object(track_id=1, bottom_center=(10.0, 10.0), timestamp=0.0)
        ],
        frame_packet=_frame_packet(frame_index=0, timestamp=0.0),
    )
    analyzer.update(
        tracked_objects=[
            _tracked_object(track_id=1, bottom_center=(11.0, 10.0), timestamp=1.0)
        ],
        frame_packet=_frame_packet(frame_index=1, timestamp=1.0),
    )
    summaries = analyzer.update(
        tracked_objects=[
            _tracked_object(track_id=1, bottom_center=(12.0, 10.0), timestamp=2.0)
        ],
        frame_packet=_frame_packet(frame_index=2, timestamp=2.0),
    )

    assert summaries[1].stationary_seconds == 2.0


def test_trajectory_analyzer_prunes_old_points() -> None:
    analyzer = TrajectoryAnalyzer(TrajectoryConfig(history_seconds=1.0))

    analyzer.update(
        tracked_objects=[
            _tracked_object(track_id=1, bottom_center=(0.0, 0.0), timestamp=0.0)
        ],
        frame_packet=_frame_packet(frame_index=0, timestamp=0.0),
    )
    analyzer.update(
        tracked_objects=[
            _tracked_object(track_id=1, bottom_center=(10.0, 0.0), timestamp=2.0)
        ],
        frame_packet=_frame_packet(frame_index=1, timestamp=2.0),
    )

    history = analyzer.get_history(track_id=1)

    assert history is not None
    assert len(history.points) == 1
    assert history.points[0].position == (10.0, 0.0)
