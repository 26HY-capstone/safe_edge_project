"""Alert 계층에서 사용하는 이벤트와 camera 단위 상태 데이터 계약."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from app.inference.detector import Point
from app.risk.models import RiskLevel
from app.zones.models import EquipmentType


@dataclass(frozen=True, slots=True)
class EventKey:
    """실행 중 동일한 작업자-설비 위험 조합을 식별한다."""

    camera_id: str
    person_track_id: int
    equipment_track_id: int


@dataclass(frozen=True, slots=True)
class RiskEvent:
    """DB 저장 계층과 공유하는 위험 이벤트 스냅샷."""

    camera_id: str
    started_utc: datetime
    last_utc: datetime
    source_timestamp_sec: float
    last_source_timestamp_sec: float
    person_track_id: int
    person_bottom_center: Point
    equipment_track_id: int
    equipment_type: EquipmentType
    risk_level: RiskLevel

    @property
    def key(self) -> EventKey:
        return EventKey(
            camera_id=self.camera_id,
            person_track_id=self.person_track_id,
            equipment_track_id=self.equipment_track_id,
        )


class EventTransitionType(Enum):
    """DB 쓰기나 사용자 로그가 필요한 이벤트 상태 변화."""

    CREATED = "created"
    ESCALATED = "escalated"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class EventTransition:
    """위험 이벤트 생명주기에서 외부 계층에 전달하는 상태 변화."""

    transition_type: EventTransitionType
    event: RiskEvent


@dataclass(slots=True)
class CameraAlertState:
    """카메라별 화면/소리 Alert 상태."""

    camera_id: str
    risk_level: RiskLevel
    state_started_at: float


@dataclass(frozen=True, slots=True)
class CameraAlertUpdate:
    """카메라별 Alert 상태 갱신 결과."""

    camera_id: str
    previous_level: RiskLevel
    current_level: RiskLevel
    changed: bool
    state_started_at: float
