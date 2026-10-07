"""위험 이벤트의 생성·격상·종료를 SQLite에 반영한다."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.alerts.models import RiskEvent
from app.storage.database import Database
from app.storage.models import RiskEventRecord


class EventRepository:
    """이벤트 전환이 발생할 때만 risk_events 테이블을 변경한다."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def create_event(self, event: RiskEvent) -> int:
        """최초 위험 이벤트를 저장하고 자동 생성된 id를 반환한다."""
        with self._database.session() as session:
            existing = self._find_event(session, event)
            if existing is not None:
                return existing.id

            x, y = event.person_bottom_center
            record = RiskEventRecord(
                camera_id=event.camera_id,
                started_utc=event.started_utc,
                last_utc=event.last_utc,
                source_timestamp_sec=event.source_timestamp_sec,
                last_source_timestamp_sec=event.last_source_timestamp_sec,
                person_track_id=event.person_track_id,
                person_bottom_center_x=x,
                person_bottom_center_y=y,
                equipment_track_id=event.equipment_track_id,
                equipment_type=event.equipment_type.value,
                risk_level=event.risk_level.value,
            )
            session.add(record)
            session.flush()
            return record.id

    def escalate_event(self, event: RiskEvent) -> None:
        """기존 이벤트의 최고 위험등급과 최근 관찰 정보를 갱신한다."""
        with self._database.session() as session:
            record = self._require_event(session, event)
            self._update_last_observation(record, event)
            record.risk_level = event.risk_level.value

    def resolve_event(self, event: RiskEvent) -> None:
        """위험 종료 시 마지막으로 관찰된 시각과 위치를 저장한다."""
        with self._database.session() as session:
            record = self._require_event(session, event)
            self._update_last_observation(record, event)

    def list_events(self) -> list[RiskEventRecord]:
        """API와 테스트에서 사용할 수 있도록 발생 순서대로 조회한다."""
        with self._database.session() as session:
            statement = select(RiskEventRecord).order_by(RiskEventRecord.id)
            return list(session.scalars(statement))

    def _require_event(
        self,
        session: Session,
        event: RiskEvent,
    ) -> RiskEventRecord:
        record = self._find_event(session, event)
        if record is None:
            raise LookupError(
                "risk event was not created before update: "
                f"camera={event.camera_id}, person={event.person_track_id}, "
                f"equipment={event.equipment_track_id}"
            )
        return record

    @staticmethod
    def _find_event(
        session: Session,
        event: RiskEvent,
    ) -> RiskEventRecord | None:
        statement = select(RiskEventRecord).where(
            RiskEventRecord.camera_id == event.camera_id,
            RiskEventRecord.person_track_id == event.person_track_id,
            RiskEventRecord.equipment_track_id == event.equipment_track_id,
            RiskEventRecord.started_utc == event.started_utc,
        )
        return session.scalar(statement)

    @staticmethod
    def _update_last_observation(
        record: RiskEventRecord,
        event: RiskEvent,
    ) -> None:
        x, y = event.person_bottom_center
        record.last_utc = event.last_utc
        record.last_source_timestamp_sec = event.last_source_timestamp_sec
        record.person_bottom_center_x = x
        record.person_bottom_center_y = y
