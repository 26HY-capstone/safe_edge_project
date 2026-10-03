"""위험 이벤트 전환을 중복 없는 터미널 로그로 변환한다."""

from __future__ import annotations

import logging

from app.alerts.models import EventTransition, EventTransitionType

logger = logging.getLogger(__name__)


class AlertManager:
    """최초 위험과 위험등급 격상만 사용자 로그로 알린다."""

    def handle(self, transitions: list[EventTransition]) -> None:
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
