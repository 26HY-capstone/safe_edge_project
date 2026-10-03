"""Alert 기능(EventManager + LocalSoundPlayer)의 orchestration을 담당하는 모듈.

AlertManager는 alert와 sound 두 stateful 컴포넌트를 소유하고, display 계층이 쓰기 편한
수준의 좁은 API만 노출한다. Alert 판정 로직(hold/escalation/de-escalation)이나
Sound 재생 로직(afplay, 3초 재평가 cycle)은 이 클래스에 복제하지 않고 각 모듈
(event_manager.py, sound.py)에 그대로 위임한다.
"""

from __future__ import annotations

from app.alerts.event_manager import EventManager
from app.alerts.models import CameraAlertUpdate
from app.alerts.sound import LocalSoundPlayer
from app.risk.models import RiskAssessment


class AlertManager:
    """camera Alert 상태(EventManager)와 Sound 재생(LocalSoundPlayer)을 함께 관리한다.

    두 컴포넌트 모두 내부에 시간 기반 state(hold 시각, sound cycle 시각)를 들고
    있으므로, main.py가 실행 중 단 하나의 AlertManager만 만들어 계속 재사용 해야한다.
    """

    def __init__(
        self,
        event_manager: EventManager | None = None,
        sound_player: LocalSoundPlayer | None = None,
    ) -> None:
        self._event_manager = event_manager or EventManager()
        self._sound_player = sound_player or LocalSoundPlayer()

    def update_camera(
        self,
        camera_id: str,
        risk_assessments: list[RiskAssessment],
    ) -> CameraAlertUpdate:
        """한 camera의 이번 프레임 RiskAssessment로 Alert 상태를 갱신하고 반환한다.

        hold/escalation/de-escalation 판단은 EventManager.update()가 그대로
        수행하며, 여기서는 그 결과를 전달만 한다.
        """
        return self._event_manager.update(
            camera_id=camera_id,
            risk_assessments=risk_assessments,
        )

    def update_sound(
        self,
        alert_updates: list[CameraAlertUpdate],
        now: float,
    ) -> None:
        """한 display cycle에서 모은 CameraAlertUpdate 전체로 Sound 상태를 갱신한다.

        "최초 진입/escalation 즉시 반응"과 "위험 유지 시 3초마다 반복 알림" 판단은
        LocalSoundPlayer.update()가 그대로 수행한다.
        """
        self._sound_player.update(alert_updates, now)

    def close(self) -> None:
        """프로그램 종료 시 Alert 관련 리소스를 정리한다."""
        self._sound_player.close()
