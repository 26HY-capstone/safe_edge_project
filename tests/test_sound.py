"""app/alerts/sound.py의 Sound 이벤트 판단과 LocalSoundPlayer 동작을 검증하는 테스트.

실제 afplay를 실행하지 않도록 subprocess.Popen을 fake process로 교체해 검증한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.alerts import sound
from app.alerts.models import CameraAlertUpdate
from app.alerts.sound import LocalSoundPlayer, get_requested_sound_level
from app.risk.models import RiskLevel


def _update(
    previous_level: RiskLevel,
    current_level: RiskLevel,
    changed: bool,
    camera_id: str = "cam_01",
) -> CameraAlertUpdate:
    return CameraAlertUpdate(
        camera_id=camera_id,
        previous_level=previous_level,
        current_level=current_level,
        changed=changed,
        state_started_at=0.0,
    )


# ---------------------------------------------------------------------------
# get_requested_sound_level
# ---------------------------------------------------------------------------


def test_normal_to_warning_requests_warning() -> None:
    update = _update(RiskLevel.NORMAL, RiskLevel.WARNING, changed=True)
    assert get_requested_sound_level([update]) == RiskLevel.WARNING


def test_normal_to_critical_requests_critical() -> None:
    update = _update(RiskLevel.NORMAL, RiskLevel.CRITICAL, changed=True)
    assert get_requested_sound_level([update]) == RiskLevel.CRITICAL


def test_warning_to_critical_requests_critical() -> None:
    update = _update(RiskLevel.WARNING, RiskLevel.CRITICAL, changed=True)
    assert get_requested_sound_level([update]) == RiskLevel.CRITICAL


def test_warning_to_warning_requests_nothing() -> None:
    update = _update(RiskLevel.WARNING, RiskLevel.WARNING, changed=False)
    assert get_requested_sound_level([update]) is None


def test_critical_to_critical_requests_nothing() -> None:
    update = _update(RiskLevel.CRITICAL, RiskLevel.CRITICAL, changed=False)
    assert get_requested_sound_level([update]) is None


def test_critical_to_warning_requests_nothing() -> None:
    update = _update(RiskLevel.CRITICAL, RiskLevel.WARNING, changed=True)
    assert get_requested_sound_level([update]) is None


def test_warning_to_normal_requests_nothing() -> None:
    update = _update(RiskLevel.WARNING, RiskLevel.NORMAL, changed=True)
    assert get_requested_sound_level([update]) is None


def test_multiple_new_warning_events_request_warning_once() -> None:
    updates = [
        _update(RiskLevel.NORMAL, RiskLevel.WARNING, changed=True, camera_id="cam_01"),
        _update(RiskLevel.NORMAL, RiskLevel.WARNING, changed=True, camera_id="cam_04"),
    ]
    assert get_requested_sound_level(updates) == RiskLevel.WARNING


def test_simultaneous_warning_and_critical_events_request_critical() -> None:
    updates = [
        _update(RiskLevel.NORMAL, RiskLevel.WARNING, changed=True, camera_id="cam_01"),
        _update(RiskLevel.WARNING, RiskLevel.CRITICAL, changed=True, camera_id="cam_03"),
        _update(RiskLevel.NORMAL, RiskLevel.WARNING, changed=True, camera_id="cam_04"),
    ]
    assert get_requested_sound_level(updates) == RiskLevel.CRITICAL


def test_new_warning_on_one_camera_is_selected_even_if_global_max_is_unchanged() -> None:
    """전체 최고 RiskLevel이 WARNING -> WARNING으로 그대로여도, 다른 camera에서
    새로 발생한 WARNING은 Sound 이벤트로 선택될 수 있어야 한다."""
    updates = [
        _update(RiskLevel.WARNING, RiskLevel.NORMAL, changed=True, camera_id="cam_01"),
        _update(RiskLevel.NORMAL, RiskLevel.WARNING, changed=True, camera_id="cam_02"),
    ]
    assert get_requested_sound_level(updates) == RiskLevel.WARNING


def test_no_updates_requests_nothing() -> None:
    assert get_requested_sound_level([]) is None


# ---------------------------------------------------------------------------
# LocalSoundPlayer
# ---------------------------------------------------------------------------


@dataclass
class _FakeProcess:
    """subprocess.Popen을 대신하는 deterministic fake.

    finished를 True로 바꾸면 poll()이 종료 코드를 반환해 자연 종료를 흉내낸다.
    """

    args: list
    finished: bool = False
    terminated: bool = False
    terminate_should_fail: bool = False

    def poll(self):
        return 0 if self.finished else None

    def terminate(self):
        if self.terminate_should_fail:
            raise OSError("terminate failed")
        self.terminated = True
        self.finished = True


def _player_with_fake_popen(monkeypatch, popen_side_effect=None):
    created_processes: list[_FakeProcess] = []

    def fake_popen(args, **kwargs):
        if popen_side_effect is not None:
            raise popen_side_effect
        process = _FakeProcess(args=args)
        created_processes.append(process)
        return process

    monkeypatch.setattr(sound.subprocess, "Popen", fake_popen)
    player = LocalSoundPlayer()
    return player, created_processes


def test_play_warning_when_nothing_playing_starts_process(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.play(RiskLevel.WARNING)

    assert len(created) == 1
    assert str(sound.WARNING_SOUND_PATH) in created[0].args


def test_play_critical_when_nothing_playing_starts_process(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.play(RiskLevel.CRITICAL)

    assert len(created) == 1
    assert str(sound.CRITICAL_SOUND_PATH) in created[0].args


def test_play_warning_while_warning_playing_does_not_start_new_process(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.play(RiskLevel.WARNING)
    player.play(RiskLevel.WARNING)

    assert len(created) == 1
    assert created[0].terminated is False


def test_play_critical_interrupts_playing_warning(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.play(RiskLevel.WARNING)
    player.play(RiskLevel.CRITICAL)

    assert len(created) == 2
    assert created[0].terminated is True
    assert str(sound.CRITICAL_SOUND_PATH) in created[1].args


def test_play_warning_while_critical_playing_is_ignored(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.play(RiskLevel.CRITICAL)
    player.play(RiskLevel.WARNING)

    assert len(created) == 1
    assert created[0].terminated is False


def test_play_critical_while_critical_playing_does_not_start_new_process(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.play(RiskLevel.CRITICAL)
    player.play(RiskLevel.CRITICAL)

    assert len(created) == 1
    assert created[0].terminated is False


def test_warning_can_play_again_after_previous_process_finished_naturally(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.play(RiskLevel.WARNING)
    created[0].finished = True  # 자연 종료를 흉내낸다.

    player.play(RiskLevel.WARNING)

    assert len(created) == 2


def test_critical_can_play_again_after_previous_process_finished_naturally(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.play(RiskLevel.CRITICAL)
    created[0].finished = True

    player.play(RiskLevel.CRITICAL)

    assert len(created) == 2


def test_popen_failure_does_not_raise(monkeypatch) -> None:
    player, _ = _player_with_fake_popen(monkeypatch, popen_side_effect=OSError("no afplay"))

    player.play(RiskLevel.WARNING)  # 예외가 전파되지 않아야 한다.


def test_missing_sound_file_does_not_raise(monkeypatch, tmp_path) -> None:
    """sound 파일이 없는 경로를 Popen에 넘겨도(=실행 자체가 실패) 예외가 전파되지 않는다."""

    def fake_popen(args, **kwargs):
        raise FileNotFoundError("afplay not found")

    monkeypatch.setattr(sound.subprocess, "Popen", fake_popen)
    player = LocalSoundPlayer(
        warning_sound_path=tmp_path / "missing_warning.wav",
        critical_sound_path=tmp_path / "missing_critical.wav",
    )

    player.play(RiskLevel.WARNING)  # 예외가 전파되지 않아야 한다.


def test_terminate_failure_does_not_raise(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.play(RiskLevel.WARNING)
    created[0].terminate_should_fail = True

    player.play(RiskLevel.CRITICAL)  # terminate()가 실패해도 예외가 전파되지 않아야 한다.


def test_close_terminates_running_process(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.play(RiskLevel.WARNING)
    player.close()

    assert created[0].terminated is True


def test_close_without_playing_process_does_nothing(monkeypatch) -> None:
    player, _ = _player_with_fake_popen(monkeypatch)

    player.close()  # 아무 process도 없을 때 호출해도 예외가 없어야 한다.


# ---------------------------------------------------------------------------
# Main integration: main._dispatch_sound_alert
# ---------------------------------------------------------------------------


def test_dispatch_sound_alert_collects_all_updates_before_selecting(monkeypatch) -> None:
    from app import main

    play_calls = []
    fake_player = type("_FakePlayer", (), {"play": lambda self, level: play_calls.append(level)})()

    updates = [
        _update(RiskLevel.WARNING, RiskLevel.WARNING, changed=False, camera_id="cam_01"),
        _update(RiskLevel.NORMAL, RiskLevel.WARNING, changed=True, camera_id="cam_02"),
    ]
    main._dispatch_sound_alert(updates, fake_player)

    assert play_calls == [RiskLevel.WARNING]


def test_dispatch_sound_alert_requests_critical_once_when_warning_and_critical_both_new(
    monkeypatch,
) -> None:
    from app import main

    play_calls = []
    fake_player = type("_FakePlayer", (), {"play": lambda self, level: play_calls.append(level)})()

    updates = [
        _update(RiskLevel.NORMAL, RiskLevel.WARNING, changed=True, camera_id="cam_01"),
        _update(RiskLevel.WARNING, RiskLevel.CRITICAL, changed=True, camera_id="cam_03"),
    ]
    main._dispatch_sound_alert(updates, fake_player)

    assert play_calls == [RiskLevel.CRITICAL]


def test_dispatch_sound_alert_does_not_call_play_without_new_event(monkeypatch) -> None:
    from app import main

    play_calls = []
    fake_player = type("_FakePlayer", (), {"play": lambda self, level: play_calls.append(level)})()

    updates = [
        _update(RiskLevel.WARNING, RiskLevel.WARNING, changed=False, camera_id="cam_01"),
        _update(RiskLevel.CRITICAL, RiskLevel.NORMAL, changed=True, camera_id="cam_02"),
    ]
    main._dispatch_sound_alert(updates, fake_player)

    assert play_calls == []
