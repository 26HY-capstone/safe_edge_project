"""위험 이벤트를 터미널 logging 경고로 출력하는 모듈."""

import logging

from app.alerts.event_manager import (
    RiskEvent,
    RiskEventStatus,
    RiskEventTransition,
)


class AlertManager:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("vision_guard.alert")

    def handle(self, events: list[RiskEvent]) -> None:
        for event in events:
            assessment = event.assessment
            context = (
                f"camera={assessment.camera_id} "
                f"worker={assessment.worker_track_id} "
                f"zone={assessment.zone_id or '-'} "
                f"equipment={assessment.equipment_track_id or '-'}"
            )
            if event.status == RiskEventStatus.RESOLVED:
                self.logger.info("RESOLVED %s", context)
                continue

            transition = (
                "ESCALATED"
                if event.transition == RiskEventTransition.ESCALATED
                else "ENTER"
            )
            self.logger.warning(
                "%s level=%s %s reason=%s",
                transition,
                assessment.risk_level.name,
                context,
                assessment.reason,
            )
