"""위험 이벤트 생명주기와 camera Alert 상태를 검증한다."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from app.alerts.event_manager import (
    ALERT_HOLD_SECONDS,
    EventManager,
    EventPolicy,
    RiskEventManager,
    get_highest_risk_level,
)
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
    manager = RiskEventManager(clock=clock)

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
    manager = RiskEventManager(clock=clock)
    manager.process(*_frame(RiskLevel.CRITICAL))

    clock.advance(0.1)
    transitions = manager.process(*_frame(RiskLevel.WARNING, 10.1))
    resolved = manager.resolve_all()

    assert transitions == []
    assert resolved[0].event.risk_level is RiskLevel.CRITICAL


def test_event_resolves_after_grace_and_reopens_during_cooldown() -> None:
    clock = _Clock()
    manager = RiskEventManager(
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
    manager = RiskEventManager(
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
    manager = RiskEventManager(clock=clock)
    manager.process(*_frame(RiskLevel.WARNING, camera_id="cam_01"))

    clock.advance(2.0)
    transitions = manager.process(
        *_frame(RiskLevel.NORMAL, camera_id="cam_02")
    )

    assert transitions == []
    assert len(manager.resolve_all()) == 1
class _FakeClock:
    """테스트에서 실제 sleep 없이 임의의 시각으로 바로 이동할 수 있는 가짜 시계.

    EventManager(clock=...)에 인스턴스 자체를 콜러블로 주입하고, advance()로
    "N초가 지났다"를 즉시 흉내낸다.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _assessment(
    risk_level: RiskLevel,
    camera_id: str = "cam_01",
    person_id: int = 1,
    equipment_id: int = 1,
) -> RiskAssessment:
    """테스트용 RiskAssessment를 최소 필드만 채워 생성한다."""
    return RiskAssessment(
        timestamp=0.0,
        frame_index=0,
        camera_id=camera_id,
        person_id=person_id,
        equipment_id=equipment_id,
        equipment_type=EquipmentType.FORKLIFT,
        risk_level=risk_level,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_get_highest_risk_level_returns_normal_for_empty_list() -> None:
    """RiskAssessment가 하나도 없으면 NORMAL로 본다."""
    assert get_highest_risk_level([]) == RiskLevel.NORMAL


def test_get_highest_risk_level_returns_normal_when_only_normal_present() -> None:
    assessments = [_assessment(RiskLevel.NORMAL), _assessment(RiskLevel.NORMAL)]
    assert get_highest_risk_level(assessments) == RiskLevel.NORMAL


def test_get_highest_risk_level_returns_warning_for_normal_and_warning() -> None:
    assessments = [_assessment(RiskLevel.NORMAL), _assessment(RiskLevel.WARNING)]
    assert get_highest_risk_level(assessments) == RiskLevel.WARNING


def test_get_highest_risk_level_returns_critical_for_warning_and_critical() -> None:
    assessments = [_assessment(RiskLevel.WARNING), _assessment(RiskLevel.CRITICAL)]
    assert get_highest_risk_level(assessments) == RiskLevel.CRITICAL


def test_get_highest_risk_level_picks_highest_among_many_combinations() -> None:
    """여러 person/equipment 조합 중 가장 높은 RiskLevel을 선택해야 한다."""
    assessments = [
        _assessment(RiskLevel.NORMAL, person_id=1, equipment_id=3),
        _assessment(RiskLevel.WARNING, person_id=2, equipment_id=3),
        _assessment(RiskLevel.CRITICAL, person_id=5, equipment_id=8),
    ]
    assert get_highest_risk_level(assessments) == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# CameraAlertUpdate.state_started_at (UI blink 계산에 사용)
# ---------------------------------------------------------------------------


def test_camera_alert_update_reports_current_state_started_at() -> None:
    """CameraAlertUpdate에 state_started_at이 담기고, EventManager는 UI가
    내부 _states에 접근하지 않아도 현재 상태의 시작 시각을 알 수 있게 해야 한다."""
    clock = _FakeClock(start=10.0)
    manager = EventManager(clock=clock)

    update = manager.update("cam_01", [_assessment(RiskLevel.WARNING)])

    assert update.state_started_at == 10.0


def test_event_manager_escalation_updates_state_started_at() -> None:
    """escalation이 발생하면 state_started_at이 escalation 시점으로 갱신돼야 한다."""
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)
    manager.update("cam_01", [_assessment(RiskLevel.WARNING)])  # t=0

    clock.advance(1.2)
    update = manager.update("cam_01", [_assessment(RiskLevel.CRITICAL)])  # t=1.2

    assert update.state_started_at == 1.2


def test_event_manager_sustained_level_keeps_state_started_at() -> None:
    """동일 RiskLevel이 계속 관측되면 state_started_at이 바뀌지 않아야 한다."""
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)
    manager.update("cam_01", [_assessment(RiskLevel.WARNING)])  # t=0, 시작 시각=0

    clock.advance(1.5)
    update = manager.update("cam_01", [_assessment(RiskLevel.WARNING)])  # 지속

    assert update.state_started_at == 0.0


# ---------------------------------------------------------------------------
# EventManager 기본 동작
# ---------------------------------------------------------------------------


def test_event_manager_initial_state_is_normal() -> None:
    """처음 보는 camera_id의 최초 상태는 NORMAL이어야 한다."""
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)

    update = manager.update("cam_01", [])

    assert update.current_level == RiskLevel.NORMAL


def test_event_manager_escalates_immediately_from_normal_to_warning() -> None:
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)

    update = manager.update("cam_01", [_assessment(RiskLevel.WARNING)])

    assert update.previous_level == RiskLevel.NORMAL
    assert update.current_level == RiskLevel.WARNING
    assert update.changed is True


def test_event_manager_escalates_immediately_from_normal_to_critical() -> None:
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)

    update = manager.update("cam_01", [_assessment(RiskLevel.CRITICAL)])

    assert update.previous_level == RiskLevel.NORMAL
    assert update.current_level == RiskLevel.CRITICAL
    assert update.changed is True


def test_event_manager_sustained_warning_does_not_change_or_reset_timer() -> None:
    """WARNING이 계속 관측되면 상태 변화가 없고, hold 시작 시각도 리셋되지 않아야 한다."""
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)

    manager.update("cam_01", [_assessment(RiskLevel.WARNING)])  # t=0, WARNING 시작

    clock.advance(2.9)
    sustained_update = manager.update("cam_01", [_assessment(RiskLevel.WARNING)])
    assert sustained_update.changed is False
    assert sustained_update.current_level == RiskLevel.WARNING

    # 만약 위의 "지속" 호출이 타이머를 t=2.9로 리셋했다면, 아래 시점(t=2.9+0.2=3.1)은
    # 원래 시작 시각(t=0) 기준 hold(3.0s)는 지났지만 리셋된 기준(2.9)으로는 아직이다.
    # 실제로는 리셋되지 않아야 하므로, t=0 기준으로 hold가 끝난 이 시점에 NORMAL로
    # 바로 내려가야 한다.
    clock.advance(0.2)
    final_update = manager.update("cam_01", [_assessment(RiskLevel.NORMAL)])
    assert final_update.current_level == RiskLevel.NORMAL
    assert final_update.changed is True


def test_event_manager_sustained_critical_does_not_change() -> None:
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)

    manager.update("cam_01", [_assessment(RiskLevel.CRITICAL)])

    clock.advance(1.0)
    update = manager.update("cam_01", [_assessment(RiskLevel.CRITICAL)])

    assert update.changed is False
    assert update.current_level == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# Escalation (상승은 항상 즉시)
# ---------------------------------------------------------------------------


def test_event_manager_escalates_warning_to_critical_before_hold_expires() -> None:
    """WARNING 시작 1.2초 후 CRITICAL이 감지되면, 3초 hold가 끝나기 전이어도
    즉시 CRITICAL로 격상돼야 한다."""
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)

    manager.update("cam_01", [_assessment(RiskLevel.WARNING)])  # t=0

    clock.advance(1.2)
    update = manager.update("cam_01", [_assessment(RiskLevel.CRITICAL)])  # t=1.2

    assert update.previous_level == RiskLevel.WARNING
    assert update.current_level == RiskLevel.CRITICAL
    assert update.changed is True


def test_event_manager_escalation_restarts_hold_from_escalation_moment() -> None:
    """escalation이 일어난 시점부터 새로운 3초 hold가 시작돼야 한다."""
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)

    manager.update("cam_01", [_assessment(RiskLevel.WARNING)])  # t=0, WARNING 시작

    clock.advance(1.2)
    manager.update("cam_01", [_assessment(RiskLevel.CRITICAL)])  # t=1.2, CRITICAL로 격상

    # 원래 WARNING 시작(t=0) 기준으로는 3초가 지난 시점(t=3.1)이지만,
    # CRITICAL의 hold는 격상 시점(t=1.2)부터 새로 시작했으므로 아직 2초밖에
    # 지나지 않아 하락이 반영되면 안 된다.
    clock.advance(1.9)  # now = 3.1
    update = manager.update("cam_01", [_assessment(RiskLevel.WARNING)])

    assert update.current_level == RiskLevel.CRITICAL
    assert update.changed is False


# ---------------------------------------------------------------------------
# De-escalation (하락은 hold가 끝난 뒤에만)
# ---------------------------------------------------------------------------


def test_event_manager_keeps_warning_before_hold_expires_when_observed_drops_to_normal() -> None:
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)
    manager.update("cam_01", [_assessment(RiskLevel.WARNING)])  # t=0

    clock.advance(ALERT_HOLD_SECONDS - 0.1)
    update = manager.update("cam_01", [])  # observed=NORMAL

    assert update.current_level == RiskLevel.WARNING
    assert update.changed is False


def test_event_manager_drops_warning_to_normal_after_hold_expires() -> None:
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)
    manager.update("cam_01", [_assessment(RiskLevel.WARNING)])  # t=0

    clock.advance(ALERT_HOLD_SECONDS + 0.1)
    update = manager.update("cam_01", [])  # observed=NORMAL

    assert update.current_level == RiskLevel.NORMAL
    assert update.changed is True


def test_event_manager_keeps_critical_before_hold_expires_when_observed_drops_to_warning() -> None:
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)
    manager.update("cam_01", [_assessment(RiskLevel.CRITICAL)])  # t=0

    clock.advance(ALERT_HOLD_SECONDS - 0.1)
    update = manager.update("cam_01", [_assessment(RiskLevel.WARNING)])

    assert update.current_level == RiskLevel.CRITICAL
    assert update.changed is False


def test_event_manager_drops_critical_to_warning_after_hold_expires() -> None:
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)
    manager.update("cam_01", [_assessment(RiskLevel.CRITICAL)])  # t=0

    clock.advance(ALERT_HOLD_SECONDS + 0.1)
    update = manager.update("cam_01", [_assessment(RiskLevel.WARNING)])

    assert update.current_level == RiskLevel.WARNING
    assert update.changed is True


def test_event_manager_keeps_critical_before_hold_expires_when_observed_drops_to_normal() -> None:
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)
    manager.update("cam_01", [_assessment(RiskLevel.CRITICAL)])  # t=0

    clock.advance(ALERT_HOLD_SECONDS - 0.1)
    update = manager.update("cam_01", [])  # observed=NORMAL

    assert update.current_level == RiskLevel.CRITICAL
    assert update.changed is False


def test_event_manager_drops_critical_directly_to_normal_after_hold_expires() -> None:
    """CRITICAL hold가 끝난 뒤 실제 위험도가 NORMAL이면 WARNING을 거치지 않고
    바로 NORMAL로 내려가야 한다."""
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)
    manager.update("cam_01", [_assessment(RiskLevel.CRITICAL)])  # t=0

    clock.advance(ALERT_HOLD_SECONDS + 0.1)
    update = manager.update("cam_01", [])  # observed=NORMAL

    assert update.current_level == RiskLevel.NORMAL
    assert update.changed is True


# ---------------------------------------------------------------------------
# Multi-camera: camera별 상태와 timer는 서로 영향을 주지 않아야 한다
# ---------------------------------------------------------------------------


def test_event_manager_tracks_multiple_cameras_independently() -> None:
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)

    cam_01_update = manager.update("cam_01", [_assessment(RiskLevel.WARNING, camera_id="cam_01")])
    cam_02_update = manager.update("cam_02", [])

    assert cam_01_update.current_level == RiskLevel.WARNING
    assert cam_02_update.current_level == RiskLevel.NORMAL


def test_event_manager_camera_timer_changes_do_not_affect_other_cameras() -> None:
    """한 카메라의 escalation/타이머 변화가 다른 카메라에 영향을 주면 안 된다."""
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)

    # cam_01: t=0에 WARNING 시작.
    manager.update("cam_01", [_assessment(RiskLevel.WARNING, camera_id="cam_01")])

    clock.advance(1.0)
    # cam_03: t=1.0에 CRITICAL 시작(독립적으로 새 타이머).
    manager.update("cam_03", [_assessment(RiskLevel.CRITICAL, camera_id="cam_03")])

    # cam_01의 hold는 t=0 기준이므로 t=2.95에는 아직 hold가 끝나지 않았다.
    clock.advance(1.95)  # now = 2.95
    cam_01_update = manager.update("cam_01", [])  # observed=NORMAL
    assert cam_01_update.current_level == RiskLevel.WARNING
    assert cam_01_update.changed is False

    # cam_03의 hold는 t=1.0 기준이므로 now=2.95에서는 1.95초밖에 지나지 않아
    # 역시 아직 끝나지 않았어야 한다(cam_01 호출이 cam_03에 영향을 주지 않는지 확인).
    cam_03_update = manager.update("cam_03", [_assessment(RiskLevel.WARNING, camera_id="cam_03")])
    assert cam_03_update.current_level == RiskLevel.CRITICAL
    assert cam_03_update.changed is False

    # now=3.05: cam_01은 t=0 기준 3.05초 지나 hold 종료 → NORMAL 반영.
    # cam_03은 t=1.0 기준 2.05초만 지나 아직 CRITICAL 유지.
    clock.advance(0.1)  # now = 3.05
    cam_01_final = manager.update("cam_01", [])
    cam_03_final = manager.update("cam_03", [_assessment(RiskLevel.WARNING, camera_id="cam_03")])

    assert cam_01_final.current_level == RiskLevel.NORMAL
    assert cam_01_final.changed is True
    assert cam_03_final.current_level == RiskLevel.CRITICAL
    assert cam_03_final.changed is False


# ---------------------------------------------------------------------------
# Clock: fake clock만으로 시간 테스트를 수행하며 실제 sleep이 없어야 한다
# ---------------------------------------------------------------------------


def test_event_manager_hold_expiry_test_uses_no_real_sleep() -> None:
    """fake clock으로 3초 이상을 "흘려보내도" 실제 wall-clock 시간은 거의 들지 않아야
    한다(= time.sleep 등 실제 대기가 전혀 없다는 뜻)."""
    clock = _FakeClock(start=0.0)
    manager = EventManager(clock=clock)
    manager.update("cam_01", [_assessment(RiskLevel.CRITICAL)])

    wall_clock_start = time.perf_counter()
    clock.advance(ALERT_HOLD_SECONDS + 0.1)
    update = manager.update("cam_01", [])
    wall_clock_elapsed = time.perf_counter() - wall_clock_start

    assert update.current_level == RiskLevel.NORMAL
    # 3초 이상 "경과"시켰는데도 실제로는 0.5초 안에 끝나야 real sleep이 없다고 볼 수 있다.
    assert wall_clock_elapsed < 0.5
