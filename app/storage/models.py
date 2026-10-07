"""이벤트 저장에 사용하는 SQLAlchemy 데이터베이스 모델."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Vision Guard ORM 모델의 공통 Base."""


class RiskEventRecord(Base):
    """하나의 작업자-설비 위험 상황을 저장하는 테이블."""

    __tablename__ = "risk_events"
    __table_args__ = (
        UniqueConstraint(
            "camera_id",
            "person_track_id",
            "equipment_track_id",
            "started_utc",
            name="uq_risk_event_natural_key",
        ),
        Index("ix_risk_events_camera_started", "camera_id", "started_utc"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    started_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    source_timestamp_sec: Mapped[float] = mapped_column(Float, nullable=False)
    last_source_timestamp_sec: Mapped[float] = mapped_column(Float, nullable=False)
    person_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    person_bottom_center_x: Mapped[float] = mapped_column(Float, nullable=False)
    person_bottom_center_y: Mapped[float] = mapped_column(Float, nullable=False)
    equipment_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    equipment_type: Mapped[str] = mapped_column(String(32), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
