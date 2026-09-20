"""위험 이벤트 중복 제거, 등급 상승, 해제를 검증한다."""

from app.alerts.event_manager import (
    EventManager,
    RiskEventStatus,
    RiskEventTransition,
)
from app.risk.models import RiskAssessment, RiskLevel


def _assessment(level: RiskLevel, timestamp: float = 1.0) -> RiskAssessment:
    return RiskAssessment(
        risk_level=level,
        risk_type="zone_intrusion",
        camera_id="cam-1",
        worker_track_id=1,
        equipment_track_id=None,
        equipment_type=None,
        equipment_state=None,
        distance=None,
        zone_id="danger",
        ppe_status=None,
        reason="test",
        timestamp=timestamp,
        frame_index=int(timestamp),
    )


def test_event_manager_deduplicates_active_event() -> None:
    manager = EventManager()
    first = manager.update([_assessment(RiskLevel.WARNING)])
    repeated = manager.update([_assessment(RiskLevel.WARNING, timestamp=2.0)])
    assert first[0].transition == RiskEventTransition.CREATED
    assert repeated == []


def test_event_manager_emits_escalation_and_resolution() -> None:
    manager = EventManager()
    manager.update([_assessment(RiskLevel.WARNING)])
    escalated = manager.update([_assessment(RiskLevel.CRITICAL, timestamp=2.0)])
    resolved = manager.update([_assessment(RiskLevel.NORMAL, timestamp=3.0)])

    assert escalated[0].transition == RiskEventTransition.ESCALATED
    assert resolved[0].status == RiskEventStatus.RESOLVED


def test_event_manager_uses_current_frame_when_target_disappears() -> None:
    manager = EventManager()
    manager.update([_assessment(RiskLevel.WARNING)])

    resolved = manager.update([], timestamp=2.0, frame_index=2)

    assert resolved[0].transition == RiskEventTransition.RESOLVED
    assert resolved[0].assessment.timestamp == 2.0
    assert resolved[0].assessment.frame_index == 2
