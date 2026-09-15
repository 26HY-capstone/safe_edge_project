"""Track별 위치 이력과 이동 정보를 계산하는 모듈."""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot

from app.inference.detector import Point
from app.tracking.tracker import TrackedObject
from app.video.video_source import FramePacket


@dataclass(frozen=True, slots=True)
class TrajectoryConfig:
    history_seconds: float = 5.0
    max_points_per_track: int = 300
    stationary_distance_threshold_px: float = 3.0

    def __post_init__(self) -> None:
        """이력 유지 시간과 정지 판정 threshold의 유효 범위를 검증한다."""
        if self.history_seconds <= 0.0:
            raise ValueError("history_seconds must be positive")
        if self.max_points_per_track <= 0:
            raise ValueError("max_points_per_track must be positive")
        if self.stationary_distance_threshold_px < 0.0:
            raise ValueError(
                "stationary_distance_threshold_px must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    track_id: int
    position: Point
    timestamp: float
    frame_index: int


@dataclass(frozen=True, slots=True)
class TrackHistory:
    track_id: int
    points: tuple[TrajectoryPoint, ...]


@dataclass(frozen=True, slots=True)
class MotionSummary:
    track_id: int
    class_id: int
    class_name: str
    latest_position: Point
    distance_px: float
    displacement_px: float
    speed_px_per_sec: float
    direction: Point | None
    stationary_seconds: float
    timestamp: float


class TrajectoryAnalyzer:
    def __init__(self, config: TrajectoryConfig | None = None) -> None:
        """Track별 위치 이력을 저장하는 analyzer 상태를 생성한다."""
        self.config = config or TrajectoryConfig()
        self._histories: dict[int, deque[TrajectoryPoint]] = {}

    def update(
        self,
        tracked_objects: Sequence[TrackedObject],
        frame_packet: FramePacket,
    ) -> dict[int, MotionSummary]:
        """현재 프레임의 tracked object를 반영하고 이동 요약을 반환한다."""
        self._prune_all(current_timestamp=frame_packet.timestamp)

        summaries: dict[int, MotionSummary] = {}
        for tracked_object in tracked_objects:
            point = TrajectoryPoint(
                track_id=tracked_object.track_id,
                position=tracked_object.bottom_center,
                timestamp=tracked_object.timestamp,
                frame_index=frame_packet.frame_index,
            )
            history = self._history_for(tracked_object.track_id)
            history.append(point)
            self._prune_history(
                history=history,
                current_timestamp=frame_packet.timestamp,
            )
            summaries[tracked_object.track_id] = self._summarize(
                tracked_object=tracked_object,
                history=history,
            )

        return summaries

    def get_history(self, track_id: int) -> TrackHistory | None:
        """track ID에 해당하는 위치 이력을 조회한다."""
        history = self._histories.get(track_id)
        if history is None:
            return None
        return TrackHistory(track_id=track_id, points=tuple(history))

    def reset(self) -> None:
        """저장된 모든 trajectory 이력을 초기화한다."""
        self._histories.clear()

    def _history_for(self, track_id: int) -> deque[TrajectoryPoint]:
        """track ID별 위치 이력 버퍼를 반환한다."""
        if track_id not in self._histories:
            self._histories[track_id] = deque()
        return self._histories[track_id]

    def _prune_all(self, current_timestamp: float) -> None:
        """전체 track 이력에서 시간 범위를 벗어난 point를 제거한다."""
        expired_track_ids: list[int] = []

        for track_id, history in self._histories.items():
            self._prune_history(
                history=history,
                current_timestamp=current_timestamp,
            )
            if not history:
                expired_track_ids.append(track_id)

        for track_id in expired_track_ids:
            del self._histories[track_id]

    def _prune_history(
        self,
        history: deque[TrajectoryPoint],
        current_timestamp: float,
    ) -> None:
        """단일 track 이력의 시간 길이와 최대 point 수를 제한한다."""
        oldest_allowed_timestamp = (
            current_timestamp - self.config.history_seconds
        )

        while history and history[0].timestamp < oldest_allowed_timestamp:
            history.popleft()

        while len(history) > self.config.max_points_per_track:
            history.popleft()

    def _summarize(
        self,
        tracked_object: TrackedObject,
        history: deque[TrajectoryPoint],
    ) -> MotionSummary:
        """위치 이력을 거리, 속도, 방향, 정지시간으로 변환한다."""
        points = tuple(history)
        distance_px = _path_distance(points)
        displacement_px = _point_distance(
            first=points[0].position,
            second=points[-1].position,
        )
        elapsed_seconds = max(0.0, points[-1].timestamp - points[0].timestamp)
        speed_px_per_sec = (
            distance_px / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        )

        return MotionSummary(
            track_id=tracked_object.track_id,
            class_id=tracked_object.class_id,
            class_name=tracked_object.class_name,
            latest_position=points[-1].position,
            distance_px=distance_px,
            displacement_px=displacement_px,
            speed_px_per_sec=speed_px_per_sec,
            direction=_direction(points[0].position, points[-1].position),
            stationary_seconds=_stationary_seconds(
                points=points,
                distance_threshold_px=(
                    self.config.stationary_distance_threshold_px
                ),
            ),
            timestamp=points[-1].timestamp,
        )


def _path_distance(points: Sequence[TrajectoryPoint]) -> float:
    """연속된 trajectory point 사이의 누적 이동거리를 계산한다."""
    distance = 0.0
    for previous, current in zip(points, points[1:]):
        distance += _point_distance(previous.position, current.position)
    return distance


def _point_distance(first: Point, second: Point) -> float:
    """두 점 사이의 pixel 거리를 계산한다."""
    return hypot(second[0] - first[0], second[1] - first[1])


def _direction(first: Point, second: Point) -> Point | None:
    """시작점에서 끝점으로 향하는 단위 방향 벡터를 계산한다."""
    distance = _point_distance(first, second)
    if distance == 0.0:
        return None
    return ((second[0] - first[0]) / distance, (second[1] - first[1]) / distance)


def _stationary_seconds(
    points: Sequence[TrajectoryPoint],
    distance_threshold_px: float,
) -> float:
    """마지막 point부터 역순으로 정지 상태가 이어진 시간을 계산한다."""
    stationary_seconds = 0.0

    for previous, current in zip(reversed(points[:-1]), reversed(points[1:])):
        distance = _point_distance(previous.position, current.position)
        if distance > distance_threshold_px:
            break
        stationary_seconds += max(0.0, current.timestamp - previous.timestamp)

    return stationary_seconds
