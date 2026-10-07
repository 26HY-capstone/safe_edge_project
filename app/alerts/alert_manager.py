"""Alert 상태와 사용자 알림 동작을 조립하는 모듈."""

from __future__ import annotations

import logging
from typing import Protocol

from app.alerts.event_manager import EventManager
from app.alerts.models import (
    CameraAlertUpdate,
    EventTransition,
    EventTransitionType,
)
from app.alerts.sound import LocalSoundPlayer
from app.risk.ppe_risk import PPERiskAssessment
from app.risk.models import RiskAssessment

logger = logging.getLogger(__name__)


class _CameraAlertStateManager(Protocol):
    """AlertManager가 camera 상태 관리자에 기대하는 최소 인터페이스."""

    def update(
        self,
        camera_id: str,
        risk_assessments: list[RiskAssessment],
    ) -> CameraAlertUpdate:
        """camera_id의 Alert 상태를 갱신하고 결과를 반환한다."""
        ...


class _SoundPlayer(Protocol):
    """AlertManager가 sound player에 기대하는 최소 인터페이스."""

    def update(self, alert_updates: list[CameraAlertUpdate], now: float) -> None:
        """현재 Alert 상태 목록으로 sound 상태를 갱신한다."""
        ...

    def close(self) -> None:
        """sound player가 가진 외부 리소스를 정리한다."""
        ...


class AlertManager:
    """camera Alert 상태와 Sound 재생을 함께 관리한다."""

    def __init__(
        self,
        event_manager: _CameraAlertStateManager | None = None,
        sound_player: _SoundPlayer | None = None,
    ) -> None:
        self._event_manager = event_manager or EventManager()
        self._sound_player = sound_player or LocalSoundPlayer()

    def update_camera(
        self,
        camera_id: str,
        risk_assessments: list[RiskAssessment | PPERiskAssessment],
    ) -> CameraAlertUpdate:
        """한 camera의 이번 프레임 RiskAssessment로 Alert 상태를 갱신하고 반환한다."""
        return self._event_manager.update(
            camera_id=camera_id,
            risk_assessments=risk_assessments,
        )

    def update_sound(
        self,
        alert_updates: list[CameraAlertUpdate],
        now: float,
    ) -> None:
        """한 display cycle의 CameraAlertUpdate 전체로 Sound 상태를 갱신한다."""
        self._sound_player.update(alert_updates, now)

    def handle(self, transitions: list[EventTransition]) -> None:
        """위험 이벤트 생성과 격상을 사용자 로그로 남긴다."""
        for transition in transitions:
            if transition.transition_type is EventTransitionType.RESOLVED:
                continue

            event = transition.event
            logger.warning(
                "%s camera=%s person=%s equipment=%s:%s",
                event.risk_level.name,
                event.camera_id,
                event.person_track_id,
                event.equipment_type.value,
                event.equipment_track_id,
            )

    def close(self) -> None:
        """프로그램 종료 시 Alert 관련 리소스를 정리한다."""
        self._sound_player.close()
