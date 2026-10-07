"""SQLite 연결과 transaction 수명주기를 관리한다."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.storage.models import Base


class Database:
    """SQLAlchemy Engine과 Session의 생성·종료 경계를 제공한다."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{self.database_path.as_posix()}",
            future=True,
        )
        _enable_sqlite_safety_options(self._engine)
        self._session_factory = sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            class_=Session,
        )

    def initialize(self) -> None:
        """현재 등록된 ORM 테이블을 생성한다."""
        Base.metadata.create_all(self._engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """성공 시 commit하고 실패 시 rollback하는 Session을 제공한다."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self._engine.dispose()


def create_database_from_system_config(config_path: Path) -> Database:
    """system.yaml의 storage.database_path로 Database를 생성한다."""
    with config_path.open("r", encoding="utf-8") as config_file:
        try:
            config = yaml.safe_load(config_file) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid system config: {config_path}") from exc

    database_value = config.get("storage", {}).get("database_path")
    if not isinstance(database_value, str) or not database_value.strip():
        raise ValueError("storage.database_path must be a non-empty string")

    database_path = Path(database_value)
    if not database_path.is_absolute():
        database_path = config_path.resolve().parent.parent / database_path
    return Database(database_path)


def _enable_sqlite_safety_options(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()
