"""추가 추론 없이 프레임 단위 PPE 소유자와 착용 후보를 계산한다."""

from dataclasses import dataclass
from math import isfinite

from app.inference.detector import Detection, Point
from app.tracking.tracker import TrackedObject
from app.zones.geometry import is_point_in_bbox

PPE_CLASSES = ("helmet", "gloves", "safety_vest", "harness_body")


@dataclass(frozen=True, slots=True)
class PPEStatus:
    """한 종류의 매칭 근거. 미탐지는 실제 미착용 확정이 아니다."""

    class_name: str
    detections: tuple[Detection, ...]

    @property
    def is_worn(self) -> bool:
        """현재 프레임에서 해당 PPE가 하나 이상 연결됐는지 반환한다."""
        return bool(self.detections)


@dataclass(frozen=True, slots=True)
class WorkerPPEStatus:
    """현재 프레임의 person track에 연결된 PPE 종류별 결과."""

    track_id: int
    items: tuple[PPEStatus, ...]


class PPEMatcher:
    def match(
        self,
        detections: list[Detection],
        tracked_objects: list[TrackedObject],
    ) -> dict[int, WorkerPPEStatus]:
        """PPE마다 소유자 한 명을 선택하고 모든 person의 상태를 반환한다."""
        workers = [obj for obj in tracked_objects if obj.class_name == "person"]
        if len({worker.track_id for worker in workers}) != len(workers):
            raise ValueError("person track_id must be unique within a frame")
        for worker in workers:
            _validate_worker(worker)

        assignments: dict[int, dict[str, list[Detection]]] = {
            worker.track_id: {name: [] for name in PPE_CLASSES}
            for worker in workers
        }
        for detection in detections:
            if detection.class_name not in PPE_CLASSES:
                continue
            center = detection.center
            candidates = [
                worker for worker in workers
                if is_point_in_bbox(center, worker.bbox)
            ]
            if not candidates:
                continue
            # 각 detection을 한 번만 배정한다. 동률에서도 입력 순서에 의존하지 않는다.
            owner = min(
                candidates,
                key=lambda worker: (_normalized_distance(center, worker), worker.track_id),
            )
            assignments[owner.track_id][detection.class_name].append(detection)

        return {
            track_id: WorkerPPEStatus(
                track_id=track_id,
                items=tuple(
                    PPEStatus(class_name=name, detections=tuple(items[name]))
                    for name in PPE_CLASSES
                ),
            )
            for track_id, items in assignments.items()
        }


def _normalized_distance(center: Point, worker: TrackedObject) -> float:
    """person 크기로 정규화한 중심 거리의 제곱을 반환한다."""
    x1, y1, x2, y2 = worker.bbox
    cx, cy = worker.center
    return ((center[0] - cx) / (x2 - x1)) ** 2 + ((center[1] - cy) / (y2 - y1)) ** 2


def _validate_worker(worker: TrackedObject) -> None:
    """검증 없는 TrackedObject의 잘못된 geometry를 명확히 거부한다."""
    if len(worker.bbox) != 4 or len(worker.center) != 2:
        raise ValueError("person bbox and center must have four and two coordinates")
    if not all(isfinite(value) for value in (*worker.bbox, *worker.center)):
        raise ValueError("person geometry must contain finite coordinates")
    x1, y1, x2, y2 = worker.bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError("person bbox must have positive width and height")
