"""설비별 위험 조건과 위험등급 규칙을 정의하는 모듈."""

from __future__ import annotations

from app.risk.models import RiskLevel
from app.zones.geometry import is_point_in_bbox
from app.zones.models import EquipmentZoneInfo, WorkerZoneInfo


def determine_risk_level(worker: WorkerZoneInfo, equipment: EquipmentZoneInfo) -> RiskLevel:
    """
    작업자 한 명과 설비 하나 사이의 위험등급을 판정한다.

    판정 기준점은 작업자 bbox의 bottom_center이며,
    Critical Zone이 Warning Zone 내부에 있으므로 먼저 검사한다.

    설비 작동 여부에 따라 같은 Zone이라도 위험등급이 한 단계씩 달라진다.

                    Zone 외부   Warning Zone   Critical Zone
    설비 정지         NORMAL      NORMAL         WARNING
    설비 작동         NORMAL      WARNING        CRITICAL
    """

    # Critical Zone 내부인지 먼저 확인한다.
    if is_point_in_bbox(worker.bottom_center, equipment.critical_zone):
        # 설비가 작동 중이면 CRITICAL, 정지 상태면 한 단계 낮춰 WARNING으로 판정한다.
        return RiskLevel.CRITICAL if equipment.is_active else RiskLevel.WARNING

    # Critical Zone에는 없다면 Warning Zone 내부인지 확인한다.
    if is_point_in_bbox(worker.bottom_center, equipment.warning_zone):
        # 설비가 작동 중이면 WARNING, 정지 상태면 한 단계 낮춰 NORMAL로 판정한다.
        return RiskLevel.WARNING if equipment.is_active else RiskLevel.NORMAL

    # 어느 Zone에도 속하지 않으면 설비 작동 여부와 관계없이 정상 상태로 판정한다.
    return RiskLevel.NORMAL
