"""Tracked worker를 위험구역 판정용 데이터로 변환하는 모듈."""

from collections.abc import Sequence

from app.tracking.tracker import TrackedObject
from app.zones.models import WorkerZoneInfo


def create_worker_zone_info(tracked_object: TrackedObject) -> WorkerZoneInfo:
    return WorkerZoneInfo(
        person_id=tracked_object.track_id,
        person_bbox=tracked_object.bbox,
        bottom_center=tracked_object.bottom_center,
    )


def create_worker_zone_infos(
    tracked_objects: Sequence[TrackedObject],
) -> list[WorkerZoneInfo]:
    return [
        create_worker_zone_info(tracked_object)
        for tracked_object in tracked_objects
        if tracked_object.class_name == "person"
    ]
