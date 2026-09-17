"""위험 규칙과 위험등급 계산을 검증하는 테스트."""

from app.risk.models import RiskAssessment, RiskLevel
from app.risk.risk_engine import RiskEngine
from app.risk.risk_rules import determine_risk_level
from app.zones.geometry import is_point_in_bbox
from app.zones.models import EquipmentType, EquipmentZoneInfo, WorkerZoneInfo, ZoneFrameResult


def _worker(person_id: int, bottom_center: tuple[float, float]) -> WorkerZoneInfo:
    x, y = bottom_center
    return WorkerZoneInfo(
        person_id=person_id,
        person_bbox=(x - 5.0, y - 10.0, x + 5.0, y),
        bottom_center=bottom_center,
    )


def _equipment(
    equipment_id: int,
    warning_zone: tuple[float, float, float, float],
    critical_zone: tuple[float, float, float, float],
    equipment_type: EquipmentType = EquipmentType.FORKLIFT,
) -> EquipmentZoneInfo:
    return EquipmentZoneInfo(
        equipment_id=equipment_id,
        equipment_type=equipment_type,
        equipment_bbox=critical_zone,
        warning_zone=warning_zone,
        critical_zone=critical_zone,
    )


# 테스트들이 공통으로 사용하는 warning/critical zone.
_WARNING_ZONE = (0.0, 0.0, 100.0, 100.0)
_CRITICAL_ZONE = (40.0, 40.0, 60.0, 60.0)


# ---------------------------------------------------------------------------
# 기본 RiskLevel 판정
# ---------------------------------------------------------------------------


def test_determine_risk_level_returns_normal_when_outside_both_zones() -> None:
    """warning_zone과 critical_zone 밖에 있으면 NORMAL을 반환해야 한다."""

    worker = _worker(person_id=1, bottom_center=(200.0, 200.0))
    equipment = _equipment(equipment_id=1, warning_zone=_WARNING_ZONE, critical_zone=_CRITICAL_ZONE)

    assert determine_risk_level(worker, equipment) == RiskLevel.NORMAL


def test_determine_risk_level_returns_warning_when_inside_warning_zone_only() -> None:
    """warning_zone 내부이지만 critical_zone 밖이면 WARNING을 반환해야 한다."""

    worker = _worker(person_id=1, bottom_center=(10.0, 10.0))
    equipment = _equipment(equipment_id=1, warning_zone=_WARNING_ZONE, critical_zone=_CRITICAL_ZONE)

    assert determine_risk_level(worker, equipment) == RiskLevel.WARNING


def test_determine_risk_level_returns_critical_when_inside_critical_zone() -> None:
    """critical_zone 내부에 있으면 CRITICAL을 반환해야 한다."""

    worker = _worker(person_id=1, bottom_center=(50.0, 50.0))
    equipment = _equipment(equipment_id=1, warning_zone=_WARNING_ZONE, critical_zone=_CRITICAL_ZONE)

    assert determine_risk_level(worker, equipment) == RiskLevel.CRITICAL


def test_is_point_in_bbox_treats_boundary_as_inside() -> None:
    """point가 bbox 경계 위에 있으면 is_point_in_bbox의 현재 정의상 내부(True)로 판정돼야 한다."""

    bbox = (0.0, 0.0, 100.0, 100.0)

    # 좌상단, 우하단 경계 좌표 모두 내부로 판정되는지 확인한다.
    assert is_point_in_bbox((0.0, 0.0), bbox) is True
    assert is_point_in_bbox((100.0, 100.0), bbox) is True


# ---------------------------------------------------------------------------
# RiskEngine
# ---------------------------------------------------------------------------


def test_risk_engine_returns_empty_list_when_no_workers_and_no_equipments() -> None:
    """workers와 equipments가 모두 비어 있으면 빈 리스트를 반환해야 한다."""

    zone_result = ZoneFrameResult(
        timestamp=0.0, frame_index=0, camera_id="cam_01", workers=[], equipments=[]
    )

    assert RiskEngine().evaluate(zone_result) == []


def test_risk_engine_returns_empty_list_when_only_workers_exist() -> None:
    """equipments가 비어 있으면 worker가 있어도 빈 리스트를 반환해야 한다."""

    zone_result = ZoneFrameResult(
        timestamp=0.0,
        frame_index=0,
        camera_id="cam_01",
        workers=[_worker(person_id=1, bottom_center=(10.0, 10.0))],
        equipments=[],
    )

    assert RiskEngine().evaluate(zone_result) == []


def test_risk_engine_returns_empty_list_when_only_equipments_exist() -> None:
    """workers가 비어 있으면 equipment가 있어도 빈 리스트를 반환해야 한다."""

    zone_result = ZoneFrameResult(
        timestamp=0.0,
        frame_index=0,
        camera_id="cam_01",
        workers=[],
        equipments=[_equipment(equipment_id=1, warning_zone=_WARNING_ZONE, critical_zone=_CRITICAL_ZONE)],
    )

    assert RiskEngine().evaluate(zone_result) == []


def test_risk_engine_returns_one_assessment_for_one_worker_and_one_equipment() -> None:
    """worker 1명 x equipment 1개 조합은 RiskAssessment 1개를 생성해야 한다."""

    zone_result = ZoneFrameResult(
        timestamp=0.0,
        frame_index=0,
        camera_id="cam_01",
        workers=[_worker(person_id=1, bottom_center=(50.0, 50.0))],
        equipments=[_equipment(equipment_id=1, warning_zone=_WARNING_ZONE, critical_zone=_CRITICAL_ZONE)],
    )

    assessments = RiskEngine().evaluate(zone_result)

    assert len(assessments) == 1
    assert assessments[0].risk_level == RiskLevel.CRITICAL


def test_risk_engine_returns_all_combinations_for_multiple_workers_and_equipments() -> None:
    """worker 2명 x equipment 2개 조합은 RiskAssessment 4개(전체 조합)를 생성해야 한다."""

    zone_result = ZoneFrameResult(
        timestamp=0.0,
        frame_index=0,
        camera_id="cam_01",
        workers=[
            _worker(person_id=1, bottom_center=(50.0, 50.0)),
            _worker(person_id=2, bottom_center=(200.0, 200.0)),
        ],
        equipments=[
            _equipment(equipment_id=1, warning_zone=_WARNING_ZONE, critical_zone=_CRITICAL_ZONE),
            _equipment(equipment_id=2, warning_zone=_WARNING_ZONE, critical_zone=_CRITICAL_ZONE),
        ],
    )

    assessments = RiskEngine().evaluate(zone_result)

    assert len(assessments) == 4


def test_risk_engine_does_not_filter_by_risk_level() -> None:
    """같은 프레임에서 NORMAL, WARNING, CRITICAL이 함께 발생해도 모두 반환돼야 한다."""

    equipment = _equipment(equipment_id=1, warning_zone=_WARNING_ZONE, critical_zone=_CRITICAL_ZONE)
    zone_result = ZoneFrameResult(
        timestamp=0.0,
        frame_index=0,
        camera_id="cam_01",
        workers=[
            _worker(person_id=1, bottom_center=(200.0, 200.0)),  # NORMAL
            _worker(person_id=2, bottom_center=(10.0, 10.0)),  # WARNING
            _worker(person_id=3, bottom_center=(50.0, 50.0)),  # CRITICAL
        ],
        equipments=[equipment],
    )

    assessments = RiskEngine().evaluate(zone_result)
    risk_levels = {assessment.risk_level for assessment in assessments}

    assert len(assessments) == 3
    assert risk_levels == {RiskLevel.NORMAL, RiskLevel.WARNING, RiskLevel.CRITICAL}


def test_risk_engine_copies_frame_context_from_zone_result() -> None:
    """RiskAssessment의 timestamp, frame_index, camera_id는 입력 ZoneFrameResult와 동일해야 한다."""

    zone_result = ZoneFrameResult(
        timestamp=12.5,
        frame_index=7,
        camera_id="cam_02",
        workers=[_worker(person_id=1, bottom_center=(50.0, 50.0))],
        equipments=[_equipment(equipment_id=1, warning_zone=_WARNING_ZONE, critical_zone=_CRITICAL_ZONE)],
    )

    assessment = RiskEngine().evaluate(zone_result)[0]

    assert assessment.timestamp == 12.5
    assert assessment.frame_index == 7
    assert assessment.camera_id == "cam_02"


def test_risk_engine_maps_person_and_equipment_identity_per_combination() -> None:
    """person_id, equipment_id, equipment_type이 각 worker-equipment 조합에 맞게 전달돼야 한다."""

    worker_1 = _worker(person_id=1, bottom_center=(50.0, 50.0))  # CRITICAL 대상
    worker_2 = _worker(person_id=2, bottom_center=(200.0, 200.0))  # NORMAL 대상
    equipment_1 = _equipment(
        equipment_id=10,
        warning_zone=_WARNING_ZONE,
        critical_zone=_CRITICAL_ZONE,
        equipment_type=EquipmentType.FORKLIFT,
    )
    equipment_2 = _equipment(
        equipment_id=20,
        warning_zone=_WARNING_ZONE,
        critical_zone=_CRITICAL_ZONE,
        equipment_type=EquipmentType.ROBOT_ARM,
    )

    zone_result = ZoneFrameResult(
        timestamp=0.0,
        frame_index=0,
        camera_id="cam_01",
        workers=[worker_1, worker_2],
        equipments=[equipment_1, equipment_2],
    )

    assessments = RiskEngine().evaluate(zone_result)

    # (person_id, equipment_id) 조합을 key로 하여 각 조합의 결과를 검증한다.
    by_pair: dict[tuple[int, int], RiskAssessment] = {
        (a.person_id, a.equipment_id): a for a in assessments
    }

    assert len(by_pair) == 4

    assert by_pair[(1, 10)].risk_level == RiskLevel.CRITICAL
    assert by_pair[(1, 10)].equipment_type == EquipmentType.FORKLIFT

    assert by_pair[(1, 20)].risk_level == RiskLevel.CRITICAL
    assert by_pair[(1, 20)].equipment_type == EquipmentType.ROBOT_ARM

    assert by_pair[(2, 10)].risk_level == RiskLevel.NORMAL
    assert by_pair[(2, 10)].equipment_type == EquipmentType.FORKLIFT

    assert by_pair[(2, 20)].risk_level == RiskLevel.NORMAL
    assert by_pair[(2, 20)].equipment_type == EquipmentType.ROBOT_ARM
