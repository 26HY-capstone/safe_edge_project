"""설비별 위험 조건과 위험등급 규칙을 정의하는 모듈."""

from __future__ import annotations

from app.risk.models import RiskLevel
from app.zones.geometry import is_point_in_bbox
from app.zones.models import EquipmentZoneInfo, WorkerZoneInfo


def determine_risk_level(worker: WorkerZoneInfo, equipment: EquipmentZoneInfo) -> RiskLevel:
    """
    작업자 한 명과 설비 하나 사이의 기본 위험등급을 판정한다.

    판정 기준점은 작업자 bbox의 bottom_center이며,
    Critical Zone이 Waring Zone 내부에 있으므로 먼저 검사한다.
    """

    # Critical Zone 내부인지 먼저 확인한다.
    if is_point_in_bbox(worker.bottom_center, equipment.critical_zone):
        return RiskLevel.CRITICAL

    # Critical Zone에는 없다면 Warning Zone 내부인지 확인한다.
    if is_point_in_bbox(worker.bottom_center, equipment.warning_zone):
        return RiskLevel.WARNING

    # 어느 Zone에도 속하지 않으면 정상 상태로 판정한다.
    return RiskLevel.NORMAL
