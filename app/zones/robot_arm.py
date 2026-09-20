"""로봇팔 bbox에 margin과 temporal smoothing을 적용하는 모듈."""

from collections import deque
from collections.abc import Mapping, Sequence

from app.inference.detector import BBox
from app.tracking.tracker import TrackedObject
from app.zones.geometry import bbox_to_polygon, expand_bbox
from app.zones.models import EquipmentState, EquipmentType, EquipmentZoneInfo

DEFAULT_BBOX_MARGIN_PX = 40.0
DEFAULT_SMOOTHING_FRAMES = 5


def create_robot_arm_zone_info(
    tracked_object: TrackedObject,
    bbox: BBox | None = None,
    warning_margin_px: float = DEFAULT_BBOX_MARGIN_PX,
    state: EquipmentState = EquipmentState.UNKNOWN,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> EquipmentZoneInfo:
    equipment_bbox = bbox or tracked_object.bbox
    warning_bbox = expand_bbox(
        equipment_bbox,
        warning_margin_px,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    return EquipmentZoneInfo(
        equipment_id=tracked_object.track_id,
        equipment_type=EquipmentType.ROBOT_ARM,
        equipment_bbox=equipment_bbox,
        warning_zone=bbox_to_polygon(warning_bbox),
        critical_zone=bbox_to_polygon(equipment_bbox),
        state=state,
    )


class RobotArmZoneBuilder:
    def __init__(
        self,
        warning_margin_px: float = DEFAULT_BBOX_MARGIN_PX,
        smoothing_frames: int = DEFAULT_SMOOTHING_FRAMES,
    ) -> None:
        if smoothing_frames <= 0:
            raise ValueError("smoothing_frames must be positive")
        self.warning_margin_px = warning_margin_px
        self.smoothing_frames = smoothing_frames
        self._histories: dict[int, deque[BBox]] = {}

    def update(
        self,
        tracked_objects: Sequence[TrackedObject],
        states: Mapping[int, EquipmentState] | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> list[EquipmentZoneInfo]:
        equipment_states = states or {}
        robot_arms = [
            tracked_object
            for tracked_object in tracked_objects
            if tracked_object.class_name == EquipmentType.ROBOT_ARM.value
        ]
        active_ids = {tracked_object.track_id for tracked_object in robot_arms}
        for track_id in tuple(self._histories):
            if track_id not in active_ids:
                del self._histories[track_id]

        zones: list[EquipmentZoneInfo] = []
        for tracked_object in robot_arms:
            history = self._histories.setdefault(
                tracked_object.track_id,
                deque(maxlen=self.smoothing_frames),
            )
            history.append(tracked_object.bbox)
            zones.append(
                create_robot_arm_zone_info(
                    tracked_object,
                    bbox=_average_bbox(history),
                    warning_margin_px=self.warning_margin_px,
                    state=equipment_states.get(
                        tracked_object.track_id,
                        EquipmentState.UNKNOWN,
                    ),
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )
        return zones

    def reset(self) -> None:
        self._histories.clear()


def _average_bbox(history: Sequence[BBox]) -> BBox:
    count = float(len(history))
    sums = [sum(bbox[index] for bbox in history) for index in range(4)]
    return (
        sums[0] / count,
        sums[1] / count,
        sums[2] / count,
        sums[3] / count,
    )
