"""Zone과 Risk 계층 사이에서 사용하는 표준 데이터 계약."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.inference.detector import BBox, Point

Polygon = tuple[Point, ...]
_SUPPORTED_RISK_LEVELS = {"caution", "warning", "critical"}


class EquipmentType(Enum):
    FORKLIFT = "forklift"
    CONVEYOR = "conveyor"
    ROBOT_ARM = "robot_arm"


class EquipmentState(Enum):
    MOVING = "moving"
    RUNNING = "running"
    STOPPED = "stopped"
    UNKNOWN = "unknown"

    @property
    def is_operating(self) -> bool:
        return self in {EquipmentState.MOVING, EquipmentState.RUNNING}


@dataclass(frozen=True, slots=True)
class Zone:
    zone_id: str
    camera_id: str
    zone_type: str
    polygon: Polygon
    risk_level: str = "warning"
    equipment_track_id: int | None = None
    timestamp: float | None = None

    def __post_init__(self) -> None:
        if not self.zone_id.strip():
            raise ValueError("zone_id must not be empty")
        if not self.camera_id.strip():
            raise ValueError("camera_id must not be empty")
        if len(self.polygon) < 3:
            raise ValueError("zone polygon must contain at least three points")
        if self.risk_level not in _SUPPORTED_RISK_LEVELS:
            raise ValueError(
                f"Unsupported zone risk level: {self.risk_level}"
            )


@dataclass(frozen=True, slots=True)
class WorkerZoneInfo:
    person_id: int
    person_bbox: BBox
    bottom_center: Point


@dataclass(frozen=True, slots=True)
class EquipmentZoneInfo:
    equipment_id: int
    equipment_type: EquipmentType
    equipment_bbox: BBox
    warning_zone: Polygon
    critical_zone: Polygon
    state: EquipmentState = EquipmentState.UNKNOWN


@dataclass(frozen=True, slots=True)
class ZoneFrameResult:
    timestamp: float
    frame_index: int
    camera_id: str
    workers: list[WorkerZoneInfo]
    equipments: list[EquipmentZoneInfo]
    zones: list[Zone] = field(default_factory=list)
