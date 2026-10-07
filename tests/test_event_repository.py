"""SQLite 위험 이벤트 저장과 갱신 동작을 검증한다."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from app.alerts.models import RiskEvent
from app.risk.models import RiskLevel
from app.storage.database import Database
from app.storage.event_repository import EventRepository
from app.zones.models import EquipmentType


def _event() -> RiskEvent:
    started = datetime(2026, 10, 3, 1, 2, 3, tzinfo=timezone.utc)
    return RiskEvent(
        camera_id="cam_01",
        started_utc=started,
        last_utc=started,
        source_timestamp_sec=10.0,
        last_source_timestamp_sec=10.0,
        person_track_id=11,
        person_bottom_center=(120.0, 300.0),
        equipment_track_id=22,
        equipment_type=EquipmentType.FORKLIFT,
        risk_level=RiskLevel.WARNING,
    )


def test_create_event_is_idempotent(tmp_path) -> None:
    database = Database(tmp_path / "events.db")
    database.initialize()
    repository = EventRepository(database)
    event = _event()

    first_id = repository.create_event(event)
    second_id = repository.create_event(event)
    records = repository.list_events()
    database.dispose()

    assert first_id == second_id
    assert len(records) == 1
    assert records[0].id == first_id
    assert records[0].risk_level == "warning"


def test_escalation_updates_same_row(tmp_path) -> None:
    database = Database(tmp_path / "events.db")
    database.initialize()
    repository = EventRepository(database)
    event = _event()
    event_id = repository.create_event(event)
    escalated = replace(
        event,
        last_utc=event.started_utc + timedelta(seconds=2),
        last_source_timestamp_sec=12.0,
        person_bottom_center=(130.0, 310.0),
        risk_level=RiskLevel.CRITICAL,
    )

    repository.escalate_event(escalated)
    records = repository.list_events()
    database.dispose()

    assert len(records) == 1
    assert records[0].id == event_id
    assert records[0].risk_level == "critical"
    assert records[0].last_source_timestamp_sec == 12.0
    assert records[0].person_bottom_center_x == 130.0
    assert records[0].person_bottom_center_y == 310.0


def test_resolve_updates_last_observation_without_new_row(tmp_path) -> None:
    database = Database(tmp_path / "events.db")
    database.initialize()
    repository = EventRepository(database)
    event = _event()
    repository.create_event(event)
    resolved = replace(
        event,
        last_utc=event.started_utc + timedelta(seconds=3),
        last_source_timestamp_sec=13.0,
        person_bottom_center=(140.0, 320.0),
    )

    repository.resolve_event(resolved)
    records = repository.list_events()
    database.dispose()

    assert len(records) == 1
    assert records[0].risk_level == "warning"
    assert records[0].last_source_timestamp_sec == 13.0
    assert records[0].person_bottom_center_x == 140.0
