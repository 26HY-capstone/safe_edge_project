"""EquipmentState와 Static Polygon을 사용하는 Risk 규칙을 검증한다."""

from app.risk.models import RiskLevel
from app.risk.risk_engine import RiskEngine
from app.risk.risk_rules import determine_risk_level, determine_zone_risk_level
from app.zones.geometry import bbox_to_polygon
from app.zones.models import (
    EquipmentState,
    EquipmentType,
    EquipmentZoneInfo,
    WorkerZoneInfo,
    Zone,
    ZoneFrameResult,
)


def _worker(point: tuple[float, float]) -> WorkerZoneInfo:
    x, y = point
    return WorkerZoneInfo(
        person_id=1,
        person_bbox=(x - 5.0, y - 10.0, x + 5.0, y),
        bottom_center=point,
    )


def _equipment(state: EquipmentState) -> EquipmentZoneInfo:
    return EquipmentZoneInfo(
        equipment_id=2,
        equipment_type=EquipmentType.FORKLIFT,
        equipment_bbox=(40.0, 40.0, 60.0, 60.0),
        warning_zone=bbox_to_polygon((0.0, 0.0, 100.0, 100.0)),
        critical_zone=bbox_to_polygon((40.0, 40.0, 60.0, 60.0)),
        state=state,
    )


def test_operating_equipment_escalates_warning_and_critical_zones() -> None:
    equipment = _equipment(EquipmentState.MOVING)
    assert determine_risk_level(_worker((10.0, 10.0)), equipment) == RiskLevel.WARNING
    assert determine_risk_level(_worker((50.0, 50.0)), equipment) == RiskLevel.CRITICAL


def test_stopped_equipment_downgrades_risk() -> None:
    equipment = _equipment(EquipmentState.STOPPED)
    assert determine_risk_level(_worker((10.0, 10.0)), equipment) == RiskLevel.NORMAL
    assert determine_risk_level(_worker((50.0, 50.0)), equipment) == RiskLevel.WARNING


def test_unknown_equipment_uses_caution_in_warning_zone() -> None:
    equipment = _equipment(EquipmentState.UNKNOWN)
    assert determine_risk_level(_worker((10.0, 10.0)), equipment) == RiskLevel.CAUTION


def test_static_polygon_returns_configured_risk_level() -> None:
    zone = Zone(
        zone_id="danger",
        camera_id="cam-1",
        zone_type="static_danger",
        polygon=((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)),
        risk_level="critical",
    )
    assert determine_zone_risk_level(_worker((10.0, 10.0)), zone) == RiskLevel.CRITICAL
    assert determine_zone_risk_level(_worker((30.0, 30.0)), zone) == RiskLevel.NORMAL


def test_risk_engine_returns_equipment_and_static_zone_assessments() -> None:
    zone_result = ZoneFrameResult(
        timestamp=3.0,
        frame_index=4,
        camera_id="cam-1",
        workers=[_worker((50.0, 50.0))],
        equipments=[_equipment(EquipmentState.MOVING)],
        zones=[
            Zone(
                zone_id="danger",
                camera_id="cam-1",
                zone_type="static_danger",
                polygon=((0.0, 0.0), (80.0, 0.0), (80.0, 80.0)),
            )
        ],
    )
    assessments = RiskEngine().evaluate(zone_result)

    assert len(assessments) == 2
    assert assessments[0].risk_level == RiskLevel.CRITICAL
    assert assessments[0].equipment_state == EquipmentState.MOVING
    assert assessments[1].risk_type == "zone_intrusion"
    assert assessments[1].zone_id == "danger"
