"""위험 판단 결과를 중복 없는 이벤트 생명주기로 변환한다."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.alerts.models import (
    EventKey,
    EventTransition,
    EventTransitionType,
    RiskEvent,
)
from app.risk.models import RiskAssessment, RiskLevel
from app.zones.models import ZoneFrameResult


@dataclass(frozen=True, slots=True)
class EventPolicy:
    """위험 이벤트 종료와 재진입 판단에 사용하는 시간 정책."""

    resolve_grace_seconds: float = 1.0
    cooldown_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.resolve_grace_seconds < 0:
            raise ValueError("resolve_grace_seconds must be non-negative")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")


@dataclass(slots=True)
class _ResolvedEvent:
    event: RiskEvent
    resolved_utc: datetime


class EventManager:
    """카메라별 작업자-설비 조합의 위험 이벤트 상태를 관리한다."""

    def __init__(
        self,
        policy: EventPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.policy = policy or EventPolicy()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._active_events: dict[EventKey, RiskEvent] = {}
        self._recently_resolved_events: dict[EventKey, _ResolvedEvent] = {}

    def process(
        self,
        zone_result: ZoneFrameResult,
        risk_assessments: list[RiskAssessment],
    ) -> list[EventTransition]:
        """한 프레임의 위험 판정을 이벤트 상태 변화로 변환한다."""
        now_utc = _normalize_utc(self._clock())
        workers_by_id = {
            worker.person_id: worker for worker in zone_result.workers
        }
        dangerous_keys: set[EventKey] = set()
        transitions: list[EventTransition] = []

        for assessment in risk_assessments:
            if assessment.risk_level is RiskLevel.NORMAL:
                continue

            worker = workers_by_id.get(assessment.person_id)
            if worker is None:
                continue

            key = EventKey(
                camera_id=assessment.camera_id,
                person_track_id=assessment.person_id,
                equipment_track_id=assessment.equipment_id,
            )
            dangerous_keys.add(key)
            transition = self._record_danger(
                key=key,
                assessment=assessment,
                person_bottom_center=worker.bottom_center,
                now_utc=now_utc,
            )
            if transition is not None:
                transitions.append(transition)

        transitions.extend(
            self._resolve_missing_events(
                camera_id=zone_result.camera_id,
                dangerous_keys=dangerous_keys,
                now_utc=now_utc,
            )
        )
        self._discard_expired_resolved_events(now_utc)
        return transitions

    def resolve_all(self) -> list[EventTransition]:
        """프로그램 종료 시 남아 있는 활성 이벤트를 모두 종료한다."""
        transitions = [
            EventTransition(EventTransitionType.RESOLVED, event)
            for event in self._active_events.values()
        ]
        self._active_events.clear()
        self._recently_resolved_events.clear()
        return transitions

    def _record_danger(
        self,
        key: EventKey,
        assessment: RiskAssessment,
        person_bottom_center: tuple[float, float],
        now_utc: datetime,
    ) -> EventTransition | None:
        current_event = self._active_events.get(key)
        if current_event is None:
            current_event = self._reopen_recent_event(key, now_utc)

        if current_event is None:
            event = RiskEvent(
                camera_id=assessment.camera_id,
                started_utc=now_utc,
                last_utc=now_utc,
                source_timestamp_sec=assessment.timestamp,
                last_source_timestamp_sec=assessment.timestamp,
                person_track_id=assessment.person_id,
                person_bottom_center=person_bottom_center,
                equipment_track_id=assessment.equipment_id,
                equipment_type=assessment.equipment_type,
                risk_level=assessment.risk_level,
            )
            self._active_events[key] = event
            return EventTransition(EventTransitionType.CREATED, event)

        escalated = _risk_rank(assessment.risk_level) > _risk_rank(
            current_event.risk_level
        )
        updated_event = replace(
            current_event,
            last_utc=now_utc,
            last_source_timestamp_sec=assessment.timestamp,
            person_bottom_center=person_bottom_center,
            risk_level=(
                assessment.risk_level if escalated else current_event.risk_level
            ),
        )
        self._active_events[key] = updated_event

        if escalated:
            return EventTransition(EventTransitionType.ESCALATED, updated_event)
        return None

    def _reopen_recent_event(
        self,
        key: EventKey,
        now_utc: datetime,
    ) -> RiskEvent | None:
        resolved = self._recently_resolved_events.get(key)
        if resolved is None:
            return None

        elapsed_seconds = (now_utc - resolved.resolved_utc).total_seconds()
        if elapsed_seconds > self.policy.cooldown_seconds:
            del self._recently_resolved_events[key]
            return None

        del self._recently_resolved_events[key]
        self._active_events[key] = resolved.event
        return resolved.event

    def _resolve_missing_events(
        self,
        camera_id: str,
        dangerous_keys: set[EventKey],
        now_utc: datetime,
    ) -> list[EventTransition]:
        transitions: list[EventTransition] = []

        for key, event in list(self._active_events.items()):
            if key.camera_id != camera_id or key in dangerous_keys:
                continue

            elapsed_seconds = (now_utc - event.last_utc).total_seconds()
            if elapsed_seconds < self.policy.resolve_grace_seconds:
                continue

            del self._active_events[key]
            self._recently_resolved_events[key] = _ResolvedEvent(
                event=event,
                resolved_utc=now_utc,
            )
            transitions.append(
                EventTransition(EventTransitionType.RESOLVED, event)
            )

        return transitions

    def _discard_expired_resolved_events(self, now_utc: datetime) -> None:
        for key, resolved in list(self._recently_resolved_events.items()):
            elapsed_seconds = (now_utc - resolved.resolved_utc).total_seconds()
            if elapsed_seconds > self.policy.cooldown_seconds:
                del self._recently_resolved_events[key]


def create_event_manager_from_system_config(config_path: Path) -> EventManager:
    """system.yaml의 risk 설정으로 EventManager를 생성한다."""
    with config_path.open("r", encoding="utf-8") as config_file:
        try:
            config = yaml.safe_load(config_file) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid system config: {config_path}") from exc

    risk_config = config.get("risk", {})
    policy = EventPolicy(
        resolve_grace_seconds=float(
            risk_config.get("event_resolve_grace_seconds", 1.0)
        ),
        cooldown_seconds=float(
            risk_config.get("event_cooldown_seconds", 5.0)
        ),
    )
    return EventManager(policy=policy)


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("event clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _risk_rank(risk_level: RiskLevel) -> int:
    return {
        RiskLevel.NORMAL: 0,
        RiskLevel.WARNING: 1,
        RiskLevel.CRITICAL: 2,
    }[risk_level]
