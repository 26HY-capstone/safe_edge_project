"""설비 상태와 Polygon Zone을 사용하는 순수 위험 판정 규칙."""

from __future__ import annotations

from app.risk.models import RiskLevel
from app.zones.geometry import is_point_in_polygon
from app.zones.models import EquipmentState, EquipmentZoneInfo, WorkerZoneInfo, Zone


def determine_risk_level(
    worker: WorkerZoneInfo,
    equipment: EquipmentZoneInfo,
) -> RiskLevel:
    if is_point_in_polygon(worker.bottom_center, equipment.critical_zone):
        if equipment.state.is_operating:
            return RiskLevel.CRITICAL
        if equipment.state == EquipmentState.UNKNOWN:
            return RiskLevel.WARNING
        return RiskLevel.WARNING

    if is_point_in_polygon(worker.bottom_center, equipment.warning_zone):
        if equipment.state.is_operating:
            return RiskLevel.WARNING
        if equipment.state == EquipmentState.UNKNOWN:
            return RiskLevel.CAUTION
        return RiskLevel.NORMAL

    return RiskLevel.NORMAL


def determine_zone_risk_level(
    worker: WorkerZoneInfo,
    zone: Zone,
) -> RiskLevel:
    if not is_point_in_polygon(worker.bottom_center, zone.polygon):
        return RiskLevel.NORMAL
    return RiskLevel(zone.risk_level)
