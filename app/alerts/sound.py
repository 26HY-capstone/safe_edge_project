"""Sound Alert 재생을 담당하는 모듈.

EventManager가 만든 CameraAlertUpdate들을 보고 Sound를 "언제" 재생할지 판단하는
순수 함수들과, 실제로 macOS afplay로 wav를 non-blocking 재생/중단하는
LocalSoundPlayer로 구성된다.

  1) EventManager의 ALERT_HOLD_SECONDS: camera Alert 상태(UI에 쓰이는 current_level)를
     최소 유지하는 시간. app/alerts/event_manager.py의 책임.
  2) SOUND_REEVALUATION_SECONDS(이 모듈): 위험이 계속 유지되고 있을 때 Sound를
     "반복 알림"하는 주기. 값은 같은 3.0초지만 책임은 분리되어 있다.
  3) 실제 wav 파일의 playback 재생 시간: warning.wav/critical.wav 자체의 길이.
     3초보다 짧게 끝나도(자연 종료) Sound Alert cycle 자체는 끝난 것으로 보지 않는다
     (LocalSoundPlayer._cycle_started_at/​_current_alert_level이 playback process와
     독립적으로 유지되는 이유).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from app.alerts.models import CameraAlertUpdate
from app.risk.models import RiskLevel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOUND_DIR = PROJECT_ROOT / "data" / "sounds"
WARNING_SOUND_PATH = SOUND_DIR / "warning.wav"
CRITICAL_SOUND_PATH = SOUND_DIR / "critical.wav"

# 위험이 계속 유지되는 동안 Sound를 반복 알림하는 주기.
# EventManager.ALERT_HOLD_SECONDS와 값은 같지만(3.0초) 의미는 다르다 — 그쪽은 UI
# 상태의 "최소 유지 시간"이고, 이 값은 Sound의 "재평가/반복 알림 주기"다.
SOUND_REEVALUATION_SECONDS = 3.0

# RiskLevel 간 우선순위. event_manager.py의 동일한 우선순위 정의와 별개로 둔다
# (모듈 간 private 상수를 직접 참조하지 않기 위함). 의미는 같다: NORMAL < WARNING < CRITICAL.
_SOUND_LEVEL_PRIORITY: dict[RiskLevel, int] = {
    RiskLevel.NORMAL: 0,
    RiskLevel.WARNING: 1,
    RiskLevel.CRITICAL: 2,
}

logger = logging.getLogger(__name__)


def get_current_highest_alert_level(
    alert_updates: list[CameraAlertUpdate],
) -> RiskLevel:
    """이번 display cycle에서 모은 camera들의 "현재" Alert Level 중 최고를 계산한다.

    changed 여부와 무관하게 current_level만 본다 — 3초 재평가는 "새 이벤트가
    있었는가"가 아니라 "지금 이 순간 위험이 여전히 존재하는가"를 물어야 하기 때문이다.
    camera OFF/분석 불가로 alert_update가 None인 camera는 호출 전에 이미 걸러져
    이 목록에 들어오지 않는다(main._read_display_frame -> DisplayFrameResult 참조).
    목록이 비어 있으면 위험이 전혀 없는 것과 같으므로 NORMAL을 반환한다.
    """
    if not alert_updates:
        return RiskLevel.NORMAL

    return max(
        (update.current_level for update in alert_updates),
        key=lambda level: _SOUND_LEVEL_PRIORITY[level],
    )


def get_new_sound_event_level(
    current_highest_level: RiskLevel,
    current_sound_level: RiskLevel | None,
) -> RiskLevel | None:
    """즉시 반응해야 하는 "신규" Sound 이벤트 레벨을 판단한다.

    즉시 반응 대상은 다음 두 가지뿐이다:
      - 현재 재생 중인 Sound Alert cycle이 없는데(current_sound_level=None) 위험이
        새로 생긴 경우(current_highest_level > NORMAL) -> 최초 진입
      - 이미 cycle이 있는데 지금 관측된 최고 레벨이 그보다 더 높은 경우 -> escalation

    그 외(동일 레벨의 새 camera 이벤트, 더 낮은 레벨, 이미 NORMAL)는 None이다 — 이
    경우들은 즉시 반응하지 않고 LocalSoundPlayer의 3초 주기 재평가에서만 반영된다.
    """
    if current_highest_level == RiskLevel.NORMAL:
        return None

    if current_sound_level is None:
        return current_highest_level

    if (
        _SOUND_LEVEL_PRIORITY[current_highest_level]
        > _SOUND_LEVEL_PRIORITY[current_sound_level]
    ):
        return current_highest_level

    return None


class LocalSoundPlayer:
    """macOS afplay로 WARNING/CRITICAL 경고음을 non-blocking으로 재생/반복한다.

    두 종류의 상태를 서로 독립적으로 관리한다:
      - Sound Alert cycle 상태(_current_alert_level, _cycle_started_at): "지금
        Sound로 알리고 있는 위험 레벨은 무엇이고 그 cycle이 언제 시작됐는가". 위험이
        유지되는 동안 SOUND_REEVALUATION_SECONDS마다 다시 재생하기 위해 쓴다.
      - playback process 상태(_process): 실제 afplay subprocess. wav 파일 자체가
        3초보다 먼저 끝나 자연 종료돼도 cycle 상태는 그대로 유지된다 — "소리가
        꺼졌다"와 "위험이 해소됐다"는 서로 다른 사건이기 때문이다.
    """

    def __init__(
        self,
        warning_sound_path: Path = WARNING_SOUND_PATH,
        critical_sound_path: Path = CRITICAL_SOUND_PATH,
    ) -> None:
        self._sound_paths: dict[RiskLevel, Path] = {
            RiskLevel.WARNING: warning_sound_path,
            RiskLevel.CRITICAL: critical_sound_path,
        }
        self._process: subprocess.Popen | None = None

        # 현재 Sound Alert cycle이 대표하는 위험 레벨과 그 cycle의 시작 시각.
        # 둘 다 None이면 "현재 Sound로 알리고 있는 위험이 없다"는 뜻이다.
        self._current_alert_level: RiskLevel | None = None
        self._cycle_started_at: float | None = None

    def update(self, alert_updates: list[CameraAlertUpdate], now: float) -> None:
        """한 display cycle의 camera 결과로 Sound 상태를 한 번 갱신한다.

        main loop가 camera 4개의 DisplayFrameResult를 모두 모은 뒤, 그중
        alert_update가 있는(= 분석이 성공한) camera들만 모아 여기로 넘긴다.

        SOUND_REEVALUATION_SECONDS가 지난 경우에는 "지금 위험이 여전히 존재하는가"를
        재평가하는 쪽을 우선 적용하고(_reevaluate_cycle), 그렇지 않으면 최초 진입/
        escalation만 즉시 반응한다(_handle_immediate_event). 재평가 자체가 이미 그
        순간의 최신 상태를 반영하므로 두 분기를 동시에 적용할 필요는 없다.
        """
        current_highest_level = get_current_highest_alert_level(alert_updates)

        if (
            self._cycle_started_at is not None
            and now - self._cycle_started_at >= SOUND_REEVALUATION_SECONDS
        ):
            self._reevaluate_cycle(current_highest_level=current_highest_level, now=now)
            return

        self._handle_immediate_event(current_highest_level=current_highest_level, now=now)

    def close(self) -> None:
        """프로그램 종료 시 재생 중인 process가 있으면 안전하게 정리한다."""
        self._stop_current_process()

    def _handle_immediate_event(
        self, current_highest_level: RiskLevel, now: float
    ) -> None:
        """최초 진입/escalation만 즉시 반응한다. 동일/낮은 레벨의 새 이벤트는
        cycle을 건드리지 않고 다음 3초 재평가로 넘긴다."""
        new_event_level = get_new_sound_event_level(
            current_highest_level=current_highest_level,
            current_sound_level=self._current_alert_level,
        )
        if new_event_level is None:
            return

        self._start_cycle(risk_level=new_event_level, now=now)

    def _reevaluate_cycle(self, current_highest_level: RiskLevel, now: float) -> None:
        """SOUND_REEVALUATION_SECONDS가 지난 시점에 "지금" 위험이 여전히 있는지를
        처음부터 다시 판단한다. changed 여부는 보지 않고 current_highest_level만
        기준으로 삼으므로, 이전 cycle보다 레벨이 낮아져도(예: CRITICAL -> WARNING)
        그 낮아진 레벨로 바로 반영한다."""
        if current_highest_level == RiskLevel.NORMAL:
            # 위험이 모두 해소됨: cycle을 종료한다. 재생 중인 소리를 강제로 끊지는
            # 않는다 — 자연 종료를 그대로 두고 cycle 상태만 비운다.
            self._current_alert_level = None
            self._cycle_started_at = None
            return

        self._start_cycle(risk_level=current_highest_level, now=now)

    def _start_cycle(self, risk_level: RiskLevel, now: float) -> None:
        """새 Sound Alert cycle을 시작(또는 반복 재생)한다.

        최초 진입, escalation, 3초 주기 반복 알림이 모두 이 동작(기존 playback
        중단 + 새 playback 시작 + cycle 상태 갱신)을 그대로 공유한다.
        """
        self._stop_current_process()
        self._start_process(risk_level)
        self._current_alert_level = risk_level
        self._cycle_started_at = now

    def _start_process(self, risk_level: RiskLevel) -> None:
        sound_path = self._sound_paths[risk_level]
        try:
            self._process = subprocess.Popen(
                ["afplay", str(sound_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            logger.exception(
                "Failed to start sound playback: level=%s path=%s",
                risk_level.name,
                sound_path,
            )
            self._process = None

    def _stop_current_process(self) -> None:
        if self._process is None:
            return

        try:
            if self._process.poll() is None:
                self._process.terminate()
        except OSError:
            logger.exception("Failed to terminate sound playback process")
        finally:
            self._process = None
