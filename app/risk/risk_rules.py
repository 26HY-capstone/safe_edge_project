"""설비별 위험 조건과 위험등급 규칙을 정의하는 모듈."""

from __future__ import annotations

from app.risk.models import RiskLevel
from app.zones.geometry import is_point_in_bbox
from app.zones.models import EquipmentZoneInfo, WorkerZoneInfo


def determine_risk_level(worker: WorkerZoneInfo, equipment: EquipmentZoneInfo) -> RiskLevel:
    """
    작업자 한 명과 설비 하나 사이의 기본 위험등급을 판정한다.

    판정 기준점은 작업자 bbox의 bottom_center이며,
    Critical Zone을 Warning Zone보다 항상 먼저 검사한다.
    설비 작동 여부, 이동 방향, 거리, 체류 시간 등은 이 단계에서 다루지 않고
    이후 risk_engine.py에서 다른 분석 결과와 함께 종합한다.
    """

    # 1) Critical Zone 내부인지 먼저 확인한다.
    if is_point_in_bbox(worker.bottom_center, equipment.critical_zone):
        return RiskLevel.CRITICAL

    # 2) Critical Zone에는 없지만 Warning Zone 내부인지 확인한다.
    if is_point_in_bbox(worker.bottom_center, equipment.warning_zone):
        return RiskLevel.WARNING

    # 3) 어느 Zone에도 속하지 않으면 정상 상태로 판정한다.
    return RiskLevel.NORMAL
