"""Tracking된 지게차를 기준으로 Warning/Critical Zone을 생성하는 모듈."""

from app.tracking.tracker import TrackedObject
from app.zones.geometry import scale_bbox
from app.zones.models import EquipmentType, EquipmentZoneInfo


DEFAULT_WARNING_SCALE = 1.2


# 설비 작동 여부를 실제로 판단하는 app/equipment/forklift_state.py가 아직 구현되지 않았으므로
# 임시로 항상 작동 중(True)으로 간주한다. 해당 모듈이 추가되면 그 판단 결과를 전달하도록 교체한다.
DEFAULT_IS_ACTIVE = True


def create_forklift_zone_info(
    tracked_object: TrackedObject,
    warning_scale: float = DEFAULT_WARNING_SCALE,
    is_active: bool = DEFAULT_IS_ACTIVE,
) -> EquipmentZoneInfo:
    equipment_bbox = tracked_object.bbox

    critical_zone = equipment_bbox
    warning_zone = scale_bbox(
        equipment_bbox,
        warning_scale,
    )

    return EquipmentZoneInfo(
        equipment_id=tracked_object.track_id,
        equipment_type=EquipmentType.FORKLIFT,
        equipment_bbox=equipment_bbox,
        warning_zone=warning_zone,
        critical_zone=critical_zone,
        is_active=is_active,
    )


def create_forklift_zone_infos(
    tracked_objects: list[TrackedObject],
    warning_scale: float = DEFAULT_WARNING_SCALE,
) -> list[EquipmentZoneInfo]:
    equipments: list[EquipmentZoneInfo] = []

    for tracked_object in tracked_objects:
        if tracked_object.class_name != EquipmentType.FORKLIFT.value:
            continue

        equipments.append(
            create_forklift_zone_info(
                tracked_object,
                warning_scale,
            )
        )

    return equipments