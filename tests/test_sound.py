"""app/alerts/sound.py의 Sound Alert "3초 재평가 + 반복 알림" 정책을 검증하는 테스트.

실제 afplay를 실행하지 않도록 subprocess.Popen을 fake process로 교체하고,
실제 time.sleep 대신 호출마다 전달하는 now 값으로 시간을 흉내낸다.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.alerts import sound
from app.alerts.models import CameraAlertUpdate
from app.alerts.sound import (
    LocalSoundPlayer,
    get_current_highest_alert_level,
    get_new_sound_event_level,
)
from app.risk.models import RiskLevel


def _update(current_level: RiskLevel, camera_id: str = "cam_01") -> CameraAlertUpdate:
    """이 테스트에서는 changed/previous_level을 쓰지 않으므로(3초 재평가는
    current_level만 본다) 값은 임의로 채운다."""
    return CameraAlertUpdate(
        camera_id=camera_id,
        previous_level=RiskLevel.NORMAL,
        current_level=current_level,
        changed=False,
        state_started_at=0.0,
    )


# ---------------------------------------------------------------------------
# get_current_highest_alert_level
# ---------------------------------------------------------------------------


def test_current_highest_alert_level_empty_is_normal() -> None:
    assert get_current_highest_alert_level([]) == RiskLevel.NORMAL


def test_current_highest_alert_level_ignores_changed_flag() -> None:
    """changed=False인 CRITICAL도 현재 최고 Alert Level 계산에 포함돼야 한다."""
    updates = [_update(RiskLevel.CRITICAL, camera_id="cam_01")]
    assert get_current_highest_alert_level(updates) == RiskLevel.CRITICAL


def test_current_highest_alert_level_picks_max_across_cameras() -> None:
    updates = [
        _update(RiskLevel.WARNING, camera_id="cam_01"),
        _update(RiskLevel.CRITICAL, camera_id="cam_02"),
        _update(RiskLevel.NORMAL, camera_id="cam_03"),
    ]
    assert get_current_highest_alert_level(updates) == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# get_new_sound_event_level
# ---------------------------------------------------------------------------


def test_new_event_level_is_none_when_everything_normal() -> None:
    assert get_new_sound_event_level(RiskLevel.NORMAL, current_sound_level=None) is None


def test_new_event_level_first_entry_warning() -> None:
    assert (
        get_new_sound_event_level(RiskLevel.WARNING, current_sound_level=None)
        == RiskLevel.WARNING
    )


def test_new_event_level_first_entry_critical() -> None:
    assert (
        get_new_sound_event_level(RiskLevel.CRITICAL, current_sound_level=None)
        == RiskLevel.CRITICAL
    )


def test_new_event_level_escalation_from_warning_to_critical() -> None:
    assert (
        get_new_sound_event_level(RiskLevel.CRITICAL, current_sound_level=RiskLevel.WARNING)
        == RiskLevel.CRITICAL
    )


def test_new_event_level_same_level_is_not_new_event() -> None:
    assert (
        get_new_sound_event_level(RiskLevel.WARNING, current_sound_level=RiskLevel.WARNING)
        is None
    )


def test_new_event_level_lower_level_is_not_new_event() -> None:
    assert (
        get_new_sound_event_level(RiskLevel.WARNING, current_sound_level=RiskLevel.CRITICAL)
        is None
    )


# ---------------------------------------------------------------------------
# LocalSoundPlayer.update() — 3초 재평가 + 반복 알림 시나리오
# ---------------------------------------------------------------------------


@dataclass
class _FakeProcess:
    """subprocess.Popen을 대신하는 deterministic fake. finished를 True로 바꾸면
    자연 종료를 흉내낸다(poll()이 종료 코드를 반환)."""

    args: list
    finished: bool = False
    terminated: bool = False

    def poll(self):
        return 0 if self.finished else None

    def terminate(self):
        self.terminated = True
        self.finished = True


def _player_with_fake_popen(monkeypatch):
    created_processes: list[_FakeProcess] = []

    def fake_popen(args, **kwargs):
        process = _FakeProcess(args=args)
        created_processes.append(process)
        return process

    monkeypatch.setattr(sound.subprocess, "Popen", fake_popen)
    return LocalSoundPlayer(), created_processes


def _played_levels(created: list[_FakeProcess]) -> list[str]:
    """fake process들의 args에서 실제로 어떤 wav가 재생 요청됐는지 레벨 이름으로 뽑아낸다."""
    levels = []
    for process in created:
        if str(sound.WARNING_SOUND_PATH) in process.args:
            levels.append("WARNING")
        elif str(sound.CRITICAL_SOUND_PATH) in process.args:
            levels.append("CRITICAL")
    return levels


def test_case1_critical_persists_and_replays_after_3s(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.update([_update(RiskLevel.CRITICAL)], now=1.0)
    player.update([_update(RiskLevel.CRITICAL)], now=4.0)

    assert _played_levels(created) == ["CRITICAL", "CRITICAL"]


def test_case2_warning_persists_and_replays_after_3s(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.update([_update(RiskLevel.WARNING)], now=1.0)
    player.update([_update(RiskLevel.WARNING)], now=4.0)

    assert _played_levels(created) == ["WARNING", "WARNING"]


def test_case3_lower_level_new_event_does_not_replay_or_reset_timer(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.update([_update(RiskLevel.CRITICAL, camera_id="cam_01")], now=1.0)
    # 다른 camera의 신규 WARNING: CRITICAL보다 낮으므로 즉시 반응 없음.
    player.update(
        [
            _update(RiskLevel.CRITICAL, camera_id="cam_01"),
            _update(RiskLevel.WARNING, camera_id="cam_02"),
        ],
        now=2.0,
    )
    assert _played_levels(created) == ["CRITICAL"]

    # 3초 경과(1.0 -> 4.0): CRITICAL이 여전히 유지 중이므로 다시 재생.
    player.update(
        [
            _update(RiskLevel.CRITICAL, camera_id="cam_01"),
            _update(RiskLevel.WARNING, camera_id="cam_02"),
        ],
        now=4.0,
    )
    assert _played_levels(created) == ["CRITICAL", "CRITICAL"]


def test_case4_escalation_resets_cycle_timer(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.update([_update(RiskLevel.WARNING, camera_id="cam_01")], now=1.0)
    # 2.0s: 다른 camera CRITICAL escalation -> 즉시 반응, cycle timer = 2.0으로 reset.
    player.update(
        [
            _update(RiskLevel.WARNING, camera_id="cam_01"),
            _update(RiskLevel.CRITICAL, camera_id="cam_02"),
        ],
        now=2.0,
    )
    assert _played_levels(created) == ["WARNING", "CRITICAL"]
    assert created[0].terminated is True  # escalation 시 기존 warning은 중단된다.

    # 4.0s: reset된 cycle 기준으로 아직 2초 경과 -> 반복 재생 없음.
    player.update(
        [
            _update(RiskLevel.WARNING, camera_id="cam_01"),
            _update(RiskLevel.CRITICAL, camera_id="cam_02"),
        ],
        now=4.0,
    )
    assert _played_levels(created) == ["WARNING", "CRITICAL"]

    # 5.0s: reset 시각(2.0)으로부터 3초 경과 -> CRITICAL 다시 재생.
    player.update(
        [
            _update(RiskLevel.WARNING, camera_id="cam_01"),
            _update(RiskLevel.CRITICAL, camera_id="cam_02"),
        ],
        now=5.0,
    )
    assert _played_levels(created) == ["WARNING", "CRITICAL", "CRITICAL"]


def test_case5_reevaluation_can_lower_level_after_escalation(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.update([_update(RiskLevel.WARNING, camera_id="cam_01")], now=1.0)
    player.update(
        [
            _update(RiskLevel.WARNING, camera_id="cam_01"),
            _update(RiskLevel.CRITICAL, camera_id="cam_02"),
        ],
        now=2.0,
    )
    assert _played_levels(created) == ["WARNING", "CRITICAL"]

    # 5.0s: reset 시각(2.0)으로부터 3초 경과. CRITICAL은 끝났고 WARNING만 유지 중
    # -> 재평가 결과로 WARNING을 다시 재생한다(이전 cycle보다 낮아져도 그대로 반영).
    player.update([_update(RiskLevel.WARNING, camera_id="cam_01")], now=5.0)
    assert _played_levels(created) == ["WARNING", "CRITICAL", "WARNING"]


def test_case6_same_level_new_camera_event_does_not_replay_or_reset_timer(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.update([_update(RiskLevel.WARNING, camera_id="cam_01")], now=1.0)
    # 2.0s: 다른 camera의 신규 WARNING(동일 레벨) -> 즉시 반응/추가 재생 없음.
    player.update(
        [
            _update(RiskLevel.WARNING, camera_id="cam_01"),
            _update(RiskLevel.WARNING, camera_id="cam_02"),
        ],
        now=2.0,
    )
    assert _played_levels(created) == ["WARNING"]

    # 4.0s: 3초 경과, WARNING 하나 이상 유지 -> 다시 재생.
    player.update(
        [
            _update(RiskLevel.WARNING, camera_id="cam_01"),
            _update(RiskLevel.WARNING, camera_id="cam_02"),
        ],
        now=4.0,
    )
    assert _played_levels(created) == ["WARNING", "WARNING"]


def test_case7_critical_ends_warning_persists_reevaluation_plays_warning(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.update([_update(RiskLevel.CRITICAL, camera_id="cam_01")], now=1.0)
    player.update(
        [
            _update(RiskLevel.CRITICAL, camera_id="cam_01"),
            _update(RiskLevel.WARNING, camera_id="cam_02"),
        ],
        now=2.0,
    )
    assert _played_levels(created) == ["CRITICAL"]

    # 4.0s: CRITICAL 종료, WARNING만 유지 -> 재평가 결과 WARNING 재생.
    player.update([_update(RiskLevel.WARNING, camera_id="cam_02")], now=4.0)
    assert _played_levels(created) == ["CRITICAL", "WARNING"]


def test_case8_cycle_ends_when_all_normal_then_new_event_plays_immediately(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.update([_update(RiskLevel.WARNING, camera_id="cam_01")], now=1.0)
    assert _played_levels(created) == ["WARNING"]

    # 4.0s: 모든 camera NORMAL -> sound 없음, cycle 종료.
    player.update([_update(RiskLevel.NORMAL, camera_id="cam_01")], now=4.0)
    assert _played_levels(created) == ["WARNING"]

    # 6.0s: 새로운 WARNING -> 최초 진입으로 간주해 즉시 재생.
    # (cycle이 실제로 종료됐는지는 "새 WARNING이 escalation 판정 없이 즉시 재생되는가"로
    # 검증한다 — 내부 상태를 직접 들여다보지 않는다.)
    player.update([_update(RiskLevel.WARNING, camera_id="cam_01")], now=6.0)
    assert _played_levels(created) == ["WARNING", "WARNING"]


def test_case9_natural_playback_end_before_3s_does_not_trigger_replay(monkeypatch) -> None:
    """wav 재생이 3초보다 먼저 자연 종료돼도, cycle이 아직 3초 미만이면 같은
    레벨의 새 이벤트 때문에 다시 재생하지 않는다."""
    player, created = _player_with_fake_popen(monkeypatch)

    player.update([_update(RiskLevel.WARNING, camera_id="cam_01")], now=1.0)
    created[0].finished = True  # warning.wav가 짧아서 먼저 끝났다고 가정.

    # 2.0s: 아직 cycle 3초 미만, 같은 레벨의 새 camera 이벤트 -> 반복 재생 없음.
    player.update(
        [
            _update(RiskLevel.WARNING, camera_id="cam_01"),
            _update(RiskLevel.WARNING, camera_id="cam_02"),
        ],
        now=2.0,
    )
    assert _played_levels(created) == ["WARNING"]


def test_case10_replay_after_3s_even_if_playback_already_finished(monkeypatch) -> None:
    """playback이 자연 종료된 뒤에도 3초가 지나고 위험이 유지 중이면 정상적으로
    다시 재생돼야 한다."""
    player, created = _player_with_fake_popen(monkeypatch)

    player.update([_update(RiskLevel.WARNING, camera_id="cam_01")], now=1.0)
    created[0].finished = True

    player.update([_update(RiskLevel.WARNING, camera_id="cam_01")], now=4.0)
    assert _played_levels(created) == ["WARNING", "WARNING"]


def test_case11_simultaneous_warning_and_critical_new_events_play_critical_once(
    monkeypatch,
) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.update(
        [
            _update(RiskLevel.WARNING, camera_id="cam_01"),
            _update(RiskLevel.CRITICAL, camera_id="cam_02"),
        ],
        now=1.0,
    )

    assert _played_levels(created) == ["CRITICAL"]


def test_case12_changed_false_critical_counts_in_3s_reevaluation(monkeypatch) -> None:
    """CameraAlertUpdate.changed=False인 CRITICAL도 3초 재평가 시 현재 최고
    Alert Level 계산에 포함돼야 한다."""
    player, created = _player_with_fake_popen(monkeypatch)

    player.update([_update(RiskLevel.CRITICAL, camera_id="cam_01")], now=1.0)

    unchanged_critical = CameraAlertUpdate(
        camera_id="cam_01",
        previous_level=RiskLevel.CRITICAL,
        current_level=RiskLevel.CRITICAL,
        changed=False,
        state_started_at=0.0,
    )
    player.update([unchanged_critical], now=4.0)

    assert _played_levels(created) == ["CRITICAL", "CRITICAL"]


# ---------------------------------------------------------------------------
# Playback 실패/정리 관련 (기존 동작 유지 확인)
# ---------------------------------------------------------------------------


def test_popen_failure_does_not_raise(monkeypatch) -> None:
    def fake_popen(args, **kwargs):
        raise OSError("no afplay")

    monkeypatch.setattr(sound.subprocess, "Popen", fake_popen)
    player = LocalSoundPlayer()

    player.update([_update(RiskLevel.WARNING)], now=0.0)  # 예외가 전파되지 않아야 한다.


def test_terminate_failure_does_not_raise(monkeypatch) -> None:
    class _FailingTerminateProcess(_FakeProcess):
        def terminate(self):
            raise OSError("terminate failed")

    created = []

    def fake_popen(args, **kwargs):
        process = _FailingTerminateProcess(args=args)
        created.append(process)
        return process

    monkeypatch.setattr(sound.subprocess, "Popen", fake_popen)
    player = LocalSoundPlayer()

    player.update([_update(RiskLevel.WARNING)], now=0.0)
    # escalation으로 기존 process를 terminate하다 실패해도 예외가 전파되지 않아야 한다.
    player.update([_update(RiskLevel.CRITICAL)], now=0.5)


def test_close_terminates_running_process(monkeypatch) -> None:
    player, created = _player_with_fake_popen(monkeypatch)

    player.update([_update(RiskLevel.WARNING)], now=0.0)
    player.close()

    assert created[0].terminated is True


def test_close_without_playing_process_does_nothing(monkeypatch) -> None:
    player, _ = _player_with_fake_popen(monkeypatch)

    player.close()  # 아무 process도 없을 때 호출해도 예외가 없어야 한다.


# main.py의 Sound orchestration(_dispatch_sound_alert)은 app/alerts/alert_manager.py의
# AlertManager로 이동했다. "alert_updates/now가 그대로 전달되는가"는 이제
# tests/test_alert_manager.py::test_update_sound_calls_sound_player_update_exactly_once에서
# 검증한다.
