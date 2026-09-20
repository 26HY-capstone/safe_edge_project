"""RiskAssessment를 중복 없는 ACTIVE/RESOLVED 이벤트로 변환한다."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.risk.models import RiskAssessment, RiskLevel

_RISK_PRIORITY = {
    RiskLevel.NORMAL: 0,
    RiskLevel.CAUTION: 1,
    RiskLevel.WARNING: 2,
    RiskLevel.CRITICAL: 3,
}


class RiskEventStatus(Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"


class RiskEventTransition(Enum):
    CREATED = "created"
    ESCALATED = "escalated"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class RiskEvent:
    event_key: str
    status: RiskEventStatus
    transition: RiskEventTransition
    assessment: RiskAssessment


class EventManager:
    def __init__(self) -> None:
        self._active: dict[str, RiskAssessment] = {}

    def update(
        self,
        assessments: list[RiskAssessment],
        *,
        timestamp: float | None = None,
        frame_index: int | None = None,
    ) -> list[RiskEvent]:
        events: list[RiskEvent] = []
        seen_keys: set[str] = set()

        for assessment in assessments:
            key = self._event_key(assessment)
            seen_keys.add(key)
            previous = self._active.get(key)

            if assessment.risk_level == RiskLevel.NORMAL:
                if previous is not None:
                    events.append(self._resolved_event(key, previous, assessment))
                    del self._active[key]
                continue

            self._active[key] = assessment
            if previous is None:
                events.append(
                    RiskEvent(
                        event_key=key,
                        status=RiskEventStatus.ACTIVE,
                        transition=RiskEventTransition.CREATED,
                        assessment=assessment,
                    )
                )
            elif _RISK_PRIORITY[assessment.risk_level] > _RISK_PRIORITY[
                previous.risk_level
            ]:
                events.append(
                    RiskEvent(
                        event_key=key,
                        status=RiskEventStatus.ACTIVE,
                        transition=RiskEventTransition.ESCALATED,
                        assessment=assessment,
                    )
                )

        for key, previous in tuple(self._active.items()):
            if key in seen_keys:
                continue
            events.append(
                self._resolved_event(
                    key,
                    previous,
                    previous,
                    timestamp=timestamp,
                    frame_index=frame_index,
                )
            )
            del self._active[key]

        return events

    def reset(self) -> None:
        self._active.clear()

    @staticmethod
    def _event_key(assessment: RiskAssessment) -> str:
        target = assessment.zone_id
        if target is None:
            target = (
                f"{assessment.equipment_type.value}:"
                f"{assessment.equipment_track_id}"
                if assessment.equipment_type is not None
                else "unknown"
            )
        return (
            f"{assessment.camera_id}:"
            f"{assessment.worker_track_id}:"
            f"{assessment.risk_type}:{target}"
        )

    @staticmethod
    def _resolved_event(
        key: str,
        previous: RiskAssessment,
        current: RiskAssessment,
        *,
        timestamp: float | None = None,
        frame_index: int | None = None,
    ) -> RiskEvent:
        resolved_assessment = RiskAssessment(
            risk_level=RiskLevel.NORMAL,
            risk_type=previous.risk_type,
            camera_id=current.camera_id,
            worker_track_id=previous.worker_track_id,
            equipment_track_id=previous.equipment_track_id,
            equipment_type=previous.equipment_type,
            equipment_state=current.equipment_state,
            distance=current.distance,
            zone_id=previous.zone_id,
            ppe_status=current.ppe_status,
            reason=f"resolved: {previous.reason}",
            timestamp=current.timestamp if timestamp is None else timestamp,
            frame_index=(
                current.frame_index if frame_index is None else frame_index
            ),
        )
        return RiskEvent(
            event_key=key,
            status=RiskEventStatus.RESOLVED,
            transition=RiskEventTransition.RESOLVED,
            assessment=resolved_assessment,
        )
