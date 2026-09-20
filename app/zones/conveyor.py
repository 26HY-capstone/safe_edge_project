"""컨베이어 Detection Zone과 설정 기반 Static Zone을 지원하는 모듈."""

from collections.abc import Mapping, Sequence

from app.tracking.tracker import TrackedObject
from app.zones.geometry import bbox_to_polygon, expand_bbox
from app.zones.models import EquipmentState, EquipmentType, EquipmentZoneInfo

DEFAULT_WARNING_MARGIN_PX = 0.0


def create_conveyor_zone_info(
    tracked_object: TrackedObject,
    warning_margin_px: float = DEFAULT_WARNING_MARGIN_PX,
    state: EquipmentState = EquipmentState.UNKNOWN,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> EquipmentZoneInfo:
    warning_bbox = expand_bbox(
        tracked_object.bbox,
        warning_margin_px,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    return EquipmentZoneInfo(
        equipment_id=tracked_object.track_id,
        equipment_type=EquipmentType.CONVEYOR,
        equipment_bbox=tracked_object.bbox,
        warning_zone=bbox_to_polygon(warning_bbox),
        critical_zone=bbox_to_polygon(tracked_object.bbox),
        state=state,
    )


def create_conveyor_zone_infos(
    tracked_objects: Sequence[TrackedObject],
    warning_margin_px: float = DEFAULT_WARNING_MARGIN_PX,
    states: Mapping[int, EquipmentState] | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> list[EquipmentZoneInfo]:
    equipment_states = states or {}
    return [
        create_conveyor_zone_info(
            tracked_object,
            warning_margin_px=warning_margin_px,
            state=equipment_states.get(
                tracked_object.track_id,
                EquipmentState.UNKNOWN,
            ),
            frame_width=frame_width,
            frame_height=frame_height,
        )
        for tracked_object in tracked_objects
        if tracked_object.class_name == EquipmentType.CONVEYOR.value
    ]
