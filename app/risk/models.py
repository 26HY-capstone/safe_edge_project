"""Risk 계층의 표준 위험등급과 판정 결과 데이터 모델."""

from dataclasses import dataclass
from enum import Enum

from app.zones.models import EquipmentState, EquipmentType


class RiskLevel(Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    risk_level: RiskLevel
    risk_type: str
    camera_id: str
    worker_track_id: int
    equipment_track_id: int | None
    equipment_type: EquipmentType | None
    equipment_state: EquipmentState | None
    distance: float | None
    zone_id: str | None
    ppe_status: str | None
    reason: str
    timestamp: float
    frame_index: int

    @property
    def person_id(self) -> int:
        return self.worker_track_id

    @property
    def equipment_id(self) -> int | None:
        return self.equipment_track_id
