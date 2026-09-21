"""Tracking된 작업자 정보를 WorkerZoneInfo로 변환하는 모듈."""

from app.tracking.tracker import TrackedObject
from app.zones.models import WorkerZoneInfo


def create_worker_zone_info(
    tracked_object: TrackedObject,
) -> WorkerZoneInfo:
    return WorkerZoneInfo(
        person_id=tracked_object.track_id,
        person_bbox=tracked_object.bbox,
        bottom_center=tracked_object.bottom_center,
    )


def create_worker_zone_infos(
    tracked_objects: list[TrackedObject],
) -> list[WorkerZoneInfo]:
    workers: list[WorkerZoneInfo] = []

    for tracked_object in tracked_objects:
        if tracked_object.class_name != "person":
            continue

        workers.append(
            create_worker_zone_info(tracked_object)
        )

    return workers