"""위험 판정을 하나의 이벤트 생명주기로 묶는 규칙을 검증한다."""

from datetime import datetime, timedelta, timezone

from app.alerts.event_manager import EventManager, EventPolicy
from app.alerts.models import EventTransitionType
from app.risk.models import RiskAssessment, RiskLevel
from app.zones.models import (
    EquipmentType,
    EquipmentZoneInfo,
    WorkerZoneInfo,
    ZoneFrameResult,
)


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 3, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _frame(
    risk_level: RiskLevel,
    source_timestamp: float = 10.0,
    camera_id: str = "cam_01",
    bottom_center: tuple[float, float] = (120.0, 300.0),
) -> tuple[ZoneFrameResult, list[RiskAssessment]]:
    zone_result = ZoneFrameResult(
        timestamp=source_timestamp,
        frame_index=int(source_timestamp * 10),
        camera_id=camera_id,
        workers=[
            WorkerZoneInfo(
                person_id=11,
                person_bbox=(100.0, 100.0, 140.0, 300.0),
                bottom_center=bottom_center,
            )
        ],
        equipments=[
            EquipmentZoneInfo(
                equipment_id=22,
                equipment_type=EquipmentType.FORKLIFT,
                equipment_bbox=(200.0, 180.0, 400.0, 360.0),
                warning_zone=(160.0, 140.0, 440.0, 400.0),
                critical_zone=(200.0, 180.0, 400.0, 360.0),
                is_active=True,
            )
        ],
    )
    assessment = RiskAssessment(
        timestamp=source_timestamp,
        frame_index=zone_result.frame_index,
        camera_id=camera_id,
        person_id=11,
        equipment_id=22,
        equipment_type=EquipmentType.FORKLIFT,
        risk_level=risk_level,
    )
    return zone_result, [assessment]


def test_same_risk_is_not_emitted_again_and_escalation_is_emitted() -> None:
    clock = _Clock()
    manager = EventManager(clock=clock)

    created = manager.process(*_frame(RiskLevel.WARNING))
    clock.advance(0.1)
    repeated = manager.process(
        *_frame(RiskLevel.WARNING, 10.1, bottom_center=(121.0, 301.0))
    )
    clock.advance(0.1)
    escalated = manager.process(
        *_frame(RiskLevel.CRITICAL, 10.2, bottom_center=(122.0, 302.0))
    )

    assert [item.transition_type for item in created] == [
        EventTransitionType.CREATED
    ]
    assert repeated == []
    assert [item.transition_type for item in escalated] == [
        EventTransitionType.ESCALATED
    ]
    assert escalated[0].event.started_utc == created[0].event.started_utc
    assert escalated[0].event.risk_level is RiskLevel.CRITICAL
    assert escalated[0].event.person_bottom_center == (122.0, 302.0)


def test_deescalation_keeps_peak_risk_without_transition() -> None:
    clock = _Clock()
    manager = EventManager(clock=clock)
    manager.process(*_frame(RiskLevel.CRITICAL))

    clock.advance(0.1)
    transitions = manager.process(*_frame(RiskLevel.WARNING, 10.1))
    resolved = manager.resolve_all()

    assert transitions == []
    assert resolved[0].event.risk_level is RiskLevel.CRITICAL


def test_event_resolves_after_grace_and_reopens_during_cooldown() -> None:
    clock = _Clock()
    manager = EventManager(
        policy=EventPolicy(resolve_grace_seconds=1.0, cooldown_seconds=5.0),
        clock=clock,
    )
    created = manager.process(*_frame(RiskLevel.WARNING))

    clock.advance(0.9)
    assert manager.process(*_frame(RiskLevel.NORMAL, 10.9)) == []

    clock.advance(0.2)
    resolved = manager.process(*_frame(RiskLevel.NORMAL, 11.1))
    assert [item.transition_type for item in resolved] == [
        EventTransitionType.RESOLVED
    ]
    assert resolved[0].event.last_utc == created[0].event.last_utc

    clock.advance(4.0)
    assert manager.process(*_frame(RiskLevel.WARNING, 15.1)) == []

    clock.advance(1.1)
    second_resolution = manager.process(*_frame(RiskLevel.NORMAL, 16.2))
    assert second_resolution[0].event.started_utc == created[0].event.started_utc


def test_reentry_after_cooldown_creates_new_event() -> None:
    clock = _Clock()
    manager = EventManager(
        policy=EventPolicy(resolve_grace_seconds=1.0, cooldown_seconds=5.0),
        clock=clock,
    )
    first = manager.process(*_frame(RiskLevel.WARNING))

    clock.advance(1.1)
    manager.process(*_frame(RiskLevel.NORMAL, 11.1))
    clock.advance(5.1)
    second = manager.process(*_frame(RiskLevel.WARNING, 16.2))

    assert second[0].transition_type is EventTransitionType.CREATED
    assert second[0].event.started_utc != first[0].event.started_utc


def test_other_camera_frame_does_not_resolve_active_event() -> None:
    clock = _Clock()
    manager = EventManager(clock=clock)
    manager.process(*_frame(RiskLevel.WARNING, camera_id="cam_01"))

    clock.advance(2.0)
    transitions = manager.process(
        *_frame(RiskLevel.NORMAL, camera_id="cam_02")
    )

    assert transitions == []
    assert len(manager.resolve_all()) == 1
