"""Tracking된 컨베이어를 기준으로 Warning/Critical Zone을 생성하는 모듈."""

from app.tracking.tracker import TrackedObject
from app.zones.geometry import scale_bbox
from app.zones.models import EquipmentType, EquipmentZoneInfo


DEFAULT_WARNING_SCALE = 1.2


def create_conveyor_zone_info(
    tracked_object: TrackedObject,
    warning_scale: float = DEFAULT_WARNING_SCALE,
) -> EquipmentZoneInfo:
    equipment_bbox = tracked_object.bbox

    critical_zone = equipment_bbox
    warning_zone = scale_bbox(
        equipment_bbox,
        warning_scale,
    )

    return EquipmentZoneInfo(
        equipment_id=tracked_object.track_id,
        equipment_type=EquipmentType.CONVEYOR,
        equipment_bbox=equipment_bbox,
        warning_zone=warning_zone,
        critical_zone=critical_zone,
    )


def create_conveyor_zone_infos(
    tracked_objects: list[TrackedObject],
    warning_scale: float = DEFAULT_WARNING_SCALE,
) -> list[EquipmentZoneInfo]:
    equipments: list[EquipmentZoneInfo] = []

    for tracked_object in tracked_objects:
        if tracked_object.class_name != EquipmentType.CONVEYOR.value:
            continue

        equipments.append(
            create_conveyor_zone_info(
                tracked_object,
                warning_scale,
            )
        )

    return equipments