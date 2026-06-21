"""
Async PostgreSQL persistence layer.

Tables
------
  sessions  one row per WebSocket call (started_at / ended_at)
  events    one row per risk score emitted during a session

Set DATABASE_URL in .env or environment to connect.
Set DB_ENABLED=0 to disable entirely (app still works without a DB).
"""

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from sqlalchemy import Float, ForeignKey, Integer, String, Text, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:password@localhost/voiceguard",
)
DB_ENABLED = os.getenv("DB_ENABLED", "1") == "1"

engine = None
AsyncSessionLocal = None


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id:         Mapped[str]             = mapped_column(String(36), primary_key=True)
    started_at: Mapped[datetime]        = mapped_column(DateTime(timezone=True))
    ended_at:   Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    meta:       Mapped[dict]            = mapped_column(JSONB, default=dict)

    events: Mapped[list["Event"]] = relationship(back_populates="session")


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        Index("ix_events_session_id", "session_id"),
        Index("ix_events_created_at", "created_at"),
    )

    id:         Mapped[int]            = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str]            = mapped_column(String(36), ForeignKey("sessions.id"))
    created_at: Mapped[datetime]       = mapped_column(DateTime(timezone=True))
    score:      Mapped[float]          = mapped_column(Float)
    band:       Mapped[str]            = mapped_column(String(10))
    action:     Mapped[str]            = mapped_column(String(20))
    cnn_prob:   Mapped[float]          = mapped_column(Float)
    transcript: Mapped[str | None]     = mapped_column(Text, nullable=True)
    keywords:   Mapped[list | None]    = mapped_column(JSONB, nullable=True)

    session: Mapped["Session"] = relationship(back_populates="events")


async def init_db() -> None:
    global engine, AsyncSessionLocal
    if not DB_ENABLED:
        print("[db] disabled (DB_ENABLED=0)")
        return
    try:
        engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
        AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print(f"[db] connected → {DATABASE_URL.split('@')[-1]}")
    except Exception as e:
        print(f"[db] unavailable ({e}) — running without persistence")
        engine = None
        AsyncSessionLocal = None


async def close_db() -> None:
    global engine
    if engine:
        await engine.dispose()
        engine = None


def db_available() -> bool:
    return AsyncSessionLocal is not None


@asynccontextmanager
async def get_db():
    if not db_available():
        yield None
        return
    async with AsyncSessionLocal() as session:
        yield session


def new_session_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
