"""Detection 결과에 track ID를 부여하는 객체 추적 모듈."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.inference.detector import BBox, Detection, Point
from app.video.video_source import FramePacket


@dataclass(frozen=True, slots=True)
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    bbox: BBox
    center: Point
    bottom_center: Point
    confidence: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    track_threshold: float = 0.5
    match_threshold: float = 0.3
    track_buffer: int = 30
    frame_rate: int = 30

    def __post_init__(self) -> None:
        if not 0.0 <= self.track_threshold <= 1.0:
            raise ValueError("track_threshold must be between 0 and 1")
        if not 0.0 <= self.match_threshold <= 1.0:
            raise ValueError("match_threshold must be between 0 and 1")
        if self.track_buffer < 0:
            raise ValueError("track_buffer must be non-negative")
        if self.frame_rate <= 0:
            raise ValueError("frame_rate must be positive")


@dataclass(slots=True)
class _TrackState:
    track_id: int
    class_id: int
    class_name: str
    bbox: BBox
    confidence: float
    timestamp: float
    last_frame_index: int
    lost_frames: int = 0


class Tracker(ABC):
    @abstractmethod
    def update(
        self,
        detections: list[Detection],
        frame_packet: FramePacket,
    ) -> list[TrackedObject]:
        raise NotImplementedError

    def reset(self) -> None:
        return None


class SimpleTracker(Tracker):
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._next_track_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def update(
        self,
        detections: list[Detection],
        frame_packet: FramePacket,
    ) -> list[TrackedObject]:
        # 낮은 confidence detection은 track 생성과 갱신 대상에서 제외된다.
        # 일시적인 오탐이 위험 판단까지 전파되는 상황을 줄이기 위한 필터다.
        candidates = [
            detection
            for detection in detections
            if detection.confidence >= self.config.track_threshold
        ]

        matched_track_ids: set[int] = set()
        tracked_objects: list[TrackedObject] = []

        for detection in candidates:
            track_id = self._find_best_track_id(
                detection=detection,
                excluded_track_ids=matched_track_ids,
            )

            if track_id is None:
                track_id = self._allocate_track_id()

            state = _TrackState(
                track_id=track_id,
                class_id=detection.class_id,
                class_name=detection.class_name,
                bbox=detection.bbox,
                confidence=detection.confidence,
                timestamp=frame_packet.timestamp,
                last_frame_index=frame_packet.frame_index,
            )
            self._tracks[track_id] = state
            matched_track_ids.add(track_id)
            tracked_objects.append(self._to_tracked_object(state))

        self._age_unmatched_tracks(
            matched_track_ids=matched_track_ids,
            frame_index=frame_packet.frame_index,
        )
        return tracked_objects

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()

    def _allocate_track_id(self) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def _find_best_track_id(
        self,
        detection: Detection,
        excluded_track_ids: set[int],
    ) -> int | None:
        best_track_id = None
        best_iou = 0.0

        for track_id, state in self._tracks.items():
            if track_id in excluded_track_ids:
                continue
            if state.class_id != detection.class_id:
                continue

            iou = _bbox_iou(state.bbox, detection.bbox)
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id

        if best_iou < self.config.match_threshold:
            return None
        return best_track_id

    def _age_unmatched_tracks(
        self,
        matched_track_ids: set[int],
        frame_index: int,
    ) -> None:
        expired_track_ids: list[int] = []

        for track_id, state in self._tracks.items():
            if track_id in matched_track_ids:
                continue

            lost_frames = max(
                state.lost_frames + 1,
                frame_index - state.last_frame_index,
            )
            self._tracks[track_id] = _TrackState(
                track_id=state.track_id,
                class_id=state.class_id,
                class_name=state.class_name,
                bbox=state.bbox,
                confidence=state.confidence,
                timestamp=state.timestamp,
                last_frame_index=state.last_frame_index,
                lost_frames=lost_frames,
            )

            if lost_frames > self.config.track_buffer:
                expired_track_ids.append(track_id)

        for track_id in expired_track_ids:
            del self._tracks[track_id]

    @staticmethod
    def _to_tracked_object(state: _TrackState) -> TrackedObject:
        return TrackedObject(
            track_id=state.track_id,
            class_id=state.class_id,
            class_name=state.class_name,
            bbox=state.bbox,
            center=_bbox_center(state.bbox),
            bottom_center=_bbox_bottom_center(state.bbox),
            confidence=state.confidence,
            timestamp=state.timestamp,
        )


def _bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_bottom_center(bbox: BBox) -> Point:
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def _bbox_iou(first: BBox, second: BBox) -> float:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second

    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    first_area = (first_x2 - first_x1) * (first_y2 - first_y1)
    second_area = (second_x2 - second_x1) * (second_y2 - second_y1)
    union_area = first_area + second_area - intersection_area

    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area
