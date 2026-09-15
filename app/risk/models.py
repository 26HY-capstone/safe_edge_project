"""Risk 모듈에서 사용하는 위험도 및 위험 판정 결과 데이터 모델."""

from dataclasses import dataclass
from enum import Enum

from app.zones.models import EquipmentType


class RiskLevel(Enum):
    """작업자와 설비 사이의 위험 수준을 정의한다."""

    # 작업자가 위험구역에 포함되지 않은 정상 상태
    NORMAL = "normal"

    # 작업자가 설비의 Warning Zone에 진입한 상태
    WARNING = "warning"

    # 작업자가 설비의 Critical Zone에 진입한 상태
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """
    특정 작업자와 설비 사이의 위험 판정 결과를 저장한다.

    RiskEngine은 ZoneFrameResult의 작업자와 설비 정보를 비교한 뒤
    각 작업자-설비 조합에 대해 RiskAssessment 객체를 생성한다.
    """

    # 위험 판정의 기준이 된 프레임의 시간 정보
    # ZoneFrameResult에서 전달받은 timestamp를 그대로 사용한다.
    timestamp: float

    # 위험 판정의 기준이 된 프레임의 순번
    frame_index: int

    # 위험 상황이 발생한 카메라의 식별자
    camera_id: str

    # 위험 판정 대상 작업자의 Tracking ID
    person_id: int

    # 위험 판정 대상 설비의 Tracking ID
    equipment_id: int

    # 위험 판정 대상 설비의 종류
    equipment_type: EquipmentType

    # Risk 모듈이 최종적으로 판단한 위험 수준
    risk_level: RiskLevel