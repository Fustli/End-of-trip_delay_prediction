"""
PostgreSQL database connection and models for GTFS data.
Uses SQLAlchemy for ORM and connection pooling.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import (
    Column,
    Float,
    Index,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# Database URL from environment or default to local PostgreSQL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://gtfs_user:gtfs_password@localhost:5432/gtfs_db"
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Stop(Base):
    __tablename__ = "stops"

    stop_id = Column(String, primary_key=True, index=True)
    stop_name = Column(String)
    stop_lat = Column(Float, nullable=False)
    stop_lon = Column(Float, nullable=False)
    stop_code = Column(String)
    location_type = Column(String)
    parent_station = Column(String)
    wheelchair_boarding = Column(String)

    __table_args__ = (
        Index("idx_stops_lat_lon", "stop_lat", "stop_lon"),
    )


class Route(Base):
    __tablename__ = "routes"

    route_id = Column(String, primary_key=True, index=True)
    agency_id = Column(String)
    route_short_name = Column(String)
    route_long_name = Column(String)
    route_type = Column(Integer)
    route_desc = Column(String)
    route_color = Column(String)
    route_text_color = Column(String)


class Trip(Base):
    __tablename__ = "trips"

    trip_id = Column(String, primary_key=True, index=True)
    route_id = Column(String, index=True)
    service_id = Column(String, index=True)
    trip_headsign = Column(String)
    direction_id = Column(Integer)
    block_id = Column(String)
    shape_id = Column(String)
    wheelchair_accessible = Column(Integer)
    bikes_allowed = Column(Integer)

    __table_args__ = (
        Index("idx_trips_route_id", "route_id"),
    )


class StopTime(Base):
    __tablename__ = "stop_times"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trip_id = Column(String, nullable=False, index=True)
    stop_id = Column(String, nullable=False, index=True)
    arrival_time = Column(String)
    departure_time = Column(String)
    arrival_secs = Column(Integer, index=True)
    departure_secs = Column(Integer, index=True)
    stop_sequence = Column(Integer, nullable=False)
    stop_headsign = Column(String)
    pickup_type = Column(Integer)
    drop_off_type = Column(Integer)

    __table_args__ = (
        Index("idx_stop_times_trip_stop", "trip_id", "stop_id"),
        Index("idx_stop_times_stop_departure", "stop_id", "departure_secs"),
        Index("idx_stop_times_trip_seq", "trip_id", "stop_sequence"),
    )


def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all tables from the database."""
    Base.metadata.drop_all(bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """Get a database session (non-context manager version)."""
    return SessionLocal()


def check_database_populated() -> bool:
    """Check if the database has been populated with GTFS data."""
    try:
        with get_db() as db:
            count = db.query(Stop).limit(1).count()
            return count > 0
    except Exception:
        return False


def get_routable_stop_ids(db: Session) -> set[str]:
    """Get all stop IDs that have scheduled stops (appear in stop_times)."""
    result = db.execute(
        text("SELECT DISTINCT stop_id FROM stop_times")
    ).fetchall()
    return {row[0] for row in result}
