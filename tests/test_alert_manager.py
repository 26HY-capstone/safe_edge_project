"""app/alerts/alert_manager.py의 orchestration(책임 전달)만 검증하는 테스트.

EventManager/LocalSoundPlayer의 판정 로직 자체는 각각 test_event_manager.py,
test_sound.py에서 검증하므로 여기서는 fake로 대체해 "AlertManager가 올바른
인자로 올바르게 호출/전달하는가"만 본다. 실제 afplay는 실행하지 않는다.
"""

from __future__ import annotations

from app.alerts.alert_manager import AlertManager
from app.alerts.models import CameraAlertUpdate
from app.risk.models import RiskAssessment, RiskLevel
from app.zones.models import EquipmentType


class _FakeEventManager:
    def __init__(self) -> None:
        self.update_calls: list[tuple[str, list[RiskAssessment]]] = []

    def update(
        self, camera_id: str, risk_assessments: list[RiskAssessment]
    ) -> CameraAlertUpdate:
        self.update_calls.append((camera_id, risk_assessments))
        return CameraAlertUpdate(
            camera_id=camera_id,
            previous_level=RiskLevel.NORMAL,
            current_level=RiskLevel.WARNING,
            changed=True,
            state_started_at=0.0,
        )


class _FakeSoundPlayer:
    def __init__(self) -> None:
        self.update_calls: list[tuple[list[CameraAlertUpdate], float]] = []
        self.close_calls = 0

    def update(self, alert_updates: list[CameraAlertUpdate], now: float) -> None:
        self.update_calls.append((alert_updates, now))

    def close(self) -> None:
        self.close_calls += 1


def _risk_assessment(camera_id: str = "cam_01") -> RiskAssessment:
    return RiskAssessment(
        timestamp=0.0,
        frame_index=0,
        camera_id=camera_id,
        person_id=1,
        equipment_id=1,
        equipment_type=EquipmentType.FORKLIFT,
        risk_level=RiskLevel.WARNING,
    )


def test_update_camera_forwards_camera_id_and_risk_assessments() -> None:
    fake_event_manager = _FakeEventManager()
    manager = AlertManager(event_manager=fake_event_manager, sound_player=_FakeSoundPlayer())

    risk_assessments = [_risk_assessment("cam_01")]
    manager.update_camera(camera_id="cam_01", risk_assessments=risk_assessments)

    assert fake_event_manager.update_calls == [("cam_01", risk_assessments)]


def test_update_camera_returns_event_manager_result_unchanged() -> None:
    fake_event_manager = _FakeEventManager()
    manager = AlertManager(event_manager=fake_event_manager, sound_player=_FakeSoundPlayer())

    result = manager.update_camera(camera_id="cam_01", risk_assessments=[])

    assert result.camera_id == "cam_01"
    assert result.current_level == RiskLevel.WARNING
    assert result.changed is True


def test_update_sound_calls_sound_player_update_exactly_once() -> None:
    fake_sound_player = _FakeSoundPlayer()
    manager = AlertManager(event_manager=_FakeEventManager(), sound_player=fake_sound_player)

    alert_updates = [
        CameraAlertUpdate(
            camera_id="cam_01",
            previous_level=RiskLevel.NORMAL,
            current_level=RiskLevel.WARNING,
            changed=True,
            state_started_at=0.0,
        )
    ]
    manager.update_sound(alert_updates=alert_updates, now=7.0)

    assert fake_sound_player.update_calls == [(alert_updates, 7.0)]


def test_close_calls_sound_player_close() -> None:
    fake_sound_player = _FakeSoundPlayer()
    manager = AlertManager(event_manager=_FakeEventManager(), sound_player=fake_sound_player)

    manager.close()

    assert fake_sound_player.close_calls == 1


def test_multiple_camera_updates_use_the_same_event_manager_instance() -> None:
    fake_event_manager = _FakeEventManager()
    manager = AlertManager(event_manager=fake_event_manager, sound_player=_FakeSoundPlayer())

    manager.update_camera(camera_id="cam_01", risk_assessments=[])
    manager.update_camera(camera_id="cam_02", risk_assessments=[])
    manager.update_camera(camera_id="cam_03", risk_assessments=[])

    assert [camera_id for camera_id, _ in fake_event_manager.update_calls] == [
        "cam_01",
        "cam_02",
        "cam_03",
    ]


def test_multiple_sound_updates_use_the_same_sound_player_instance() -> None:
    fake_sound_player = _FakeSoundPlayer()
    manager = AlertManager(event_manager=_FakeEventManager(), sound_player=fake_sound_player)

    manager.update_sound(alert_updates=[], now=1.0)
    manager.update_sound(alert_updates=[], now=2.0)

    assert [now for _, now in fake_sound_player.update_calls] == [1.0, 2.0]


def test_default_construction_uses_real_event_manager_and_sound_player() -> None:
    """dependency를 주입하지 않으면 실제 EventManager/LocalSoundPlayer를 사용한다."""
    from app.alerts.event_manager import EventManager
    from app.alerts.sound import LocalSoundPlayer

    manager = AlertManager()

    assert isinstance(manager._event_manager, EventManager)
    assert isinstance(manager._sound_player, LocalSoundPlayer)
