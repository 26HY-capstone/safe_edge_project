"""Sound Alert 재생을 담당하는 모듈.

EventManager가 만든 CameraAlertUpdate들을 보고 "이번 display cycle에 새로
재생해야 할 Sound 등급이 있는가"를 판단하는 순수 함수(get_requested_sound_level)와,
실제로 macOS afplay로 wav를 non-blocking 재생/중단하는 LocalSoundPlayer로 구성된다.

3초 hold(최초 진입/escalation/de-escalation) 판단은 app/alerts/event_manager.py의
책임이며 이 모듈에서 다시 구현하지 않는다. 여기서는 EventManager가 이미 판단해 넘겨준
CameraAlertUpdate.changed/previous_level/current_level만 보고 Sound 여부를 고른다.
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

# RiskLevel 간 우선순위. event_manager.py의 동일한 우선순위 정의와 별개로 두되
# (모듈 간 private 상수를 직접 참조하지 않기 위해) 의미는 같다: NORMAL < WARNING < CRITICAL.
_SOUND_LEVEL_PRIORITY: dict[RiskLevel, int] = {
    RiskLevel.NORMAL: 0,
    RiskLevel.WARNING: 1,
    RiskLevel.CRITICAL: 2,
}

logger = logging.getLogger(__name__)


def get_requested_sound_level(
    alert_updates: list[CameraAlertUpdate],
) -> RiskLevel | None:
    """한 display cycle에서 재생해야 할 Sound 등급을 고른다(순수 함수).

    "새로운 위험 이벤트"로만 보는 경우:
      - NORMAL -> WARNING, NORMAL -> CRITICAL, WARNING -> CRITICAL (escalation)
    아래는 Sound 이벤트로 보지 않는다:
      - 동일 레벨 지속(changed=False), de-escalation, 종료(-> NORMAL)
    여러 camera에서 동시에 신규 이벤트가 발생하면 그 중 가장 높은 RiskLevel 하나만 반환한다.
    신규 이벤트가 하나도 없으면 None을 반환한다.
    """
    candidate_levels = [
        update.current_level
        for update in alert_updates
        if update.changed
        and _SOUND_LEVEL_PRIORITY[update.current_level]
        > _SOUND_LEVEL_PRIORITY[update.previous_level]
    ]
    if not candidate_levels:
        return None

    return max(candidate_levels, key=lambda level: _SOUND_LEVEL_PRIORITY[level])


class LocalSoundPlayer:
    """macOS afplay로 WARNING/CRITICAL 경고음을 non-blocking으로 재생한다.

    현재 재생 중인 subprocess와 그 RiskLevel을 기억해서, 더 높은 위험도의 요청만
    기존 재생을 중단시키고 새로 재생한다. 같은 등급이거나 더 낮은 등급의 요청은
    기존 재생을 그대로 두고 무시한다. 재생 실패(파일 없음, afplay 실행 불가 등)는
    로그만 남기고 예외를 밖으로 전파하지 않는다 — Sound는 안전 기능의 출력
    채널 중 하나일 뿐, 이 때문에 영상 분석 루프 전체가 멈추면 안 되기 때문이다.
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
        self._current_level: RiskLevel | None = None

    def play(self, risk_level: RiskLevel) -> None:
        """risk_level(WARNING/CRITICAL)의 경고음 재생을 요청한다.

        먼저 기존 process가 자연 종료됐는지 확인해 상태를 정리한 뒤, 요청한
        등급이 현재 재생 중인 등급보다 높을 때만 기존 재생을 중단하고 새로
        재생한다. 같거나 더 낮은 등급이면 아무 동작도 하지 않는다.
        """
        self._reap_finished_process()

        if self._current_level is not None and (
            _SOUND_LEVEL_PRIORITY[risk_level]
            <= _SOUND_LEVEL_PRIORITY[self._current_level]
        ):
            return

        self._stop_current_process()
        self._start_process(risk_level)

    def close(self) -> None:
        """프로그램 종료 시 재생 중인 process가 있으면 안전하게 정리한다."""
        self._stop_current_process()

    def _start_process(self, risk_level: RiskLevel) -> None:
        sound_path = self._sound_paths[risk_level]
        try:
            self._process = subprocess.Popen(
                ["afplay", str(sound_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._current_level = risk_level
        except OSError:
            logger.exception(
                "Failed to start sound playback: level=%s path=%s",
                risk_level.name,
                sound_path,
            )
            self._process = None
            self._current_level = None

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
            self._current_level = None

    def _reap_finished_process(self) -> None:
        """재생 중이던 wav가 자연 종료됐다면 상태를 비워 다시 재생 가능하게 한다."""
        if self._process is None:
            return

        if self._process.poll() is not None:
            self._process = None
            self._current_level = None
