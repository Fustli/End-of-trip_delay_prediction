#!/usr/bin/env python3
"""
Script to load GTFS data from CSV files into PostgreSQL database.

Usage:
    python scripts/load_gtfs_to_db.py [--gtfs-dir PATH] [--drop-existing]

This script:
1. Creates the database tables if they don't exist
2. Loads GTFS CSV files into the database
3. Creates optimized indexes for fast queries

Prerequisites:
1. PostgreSQL server running
2. Database and user created (see setup instructions below)

PostgreSQL Setup (run these commands in psql as superuser):
    CREATE USER gtfs_user WITH PASSWORD 'gtfs_password';
    CREATE DATABASE gtfs_db OWNER gtfs_user;
    GRANT ALL PRIVILEGES ON DATABASE gtfs_db TO gtfs_user;

Or using Docker:
    docker run -d --name gtfs-postgres \\
        -e POSTGRES_USER=gtfs_user \\
        -e POSTGRES_PASSWORD=gtfs_password \\
        -e POSTGRES_DB=gtfs_db \\
        -p 5432:5432 \\
        postgres:15
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

from app.backend.database import (
    Base,
    Route,
    Stop,
    StopTime,
    Trip,
    create_tables,
    drop_tables,
    engine,
    get_db,
)


def parse_hhmmss_to_seconds(value: str) -> int:
    """Parse GTFS time format (HH:MM:SS) to seconds since midnight."""
    if pd.isna(value) or not value:
        return 0
    s = str(value).strip()
    parts = s.split(":")
    if len(parts) != 3:
        return 0
    try:
        h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600 + m * 60 + sec
    except ValueError:
        return 0


def load_stops(gtfs_dir: str, db_session) -> int:
    """Load stops.txt into the database."""
    path = os.path.join(gtfs_dir, "stops.txt")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping")
        return 0

    df = pd.read_csv(path, dtype=str)
    df = df.fillna("")

    # Convert lat/lon to float
    df["stop_lat"] = pd.to_numeric(df["stop_lat"], errors="coerce")
    df["stop_lon"] = pd.to_numeric(df["stop_lon"], errors="coerce")

    # Filter out rows with invalid coordinates
    df = df.dropna(subset=["stop_lat", "stop_lon"])

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing stops"):
        records.append(
            Stop(
                stop_id=str(row["stop_id"]),
                stop_name=row.get("stop_name", ""),
                stop_lat=float(row["stop_lat"]),
                stop_lon=float(row["stop_lon"]),
                stop_code=row.get("stop_code", ""),
                location_type=row.get("location_type", ""),
                parent_station=row.get("parent_station", ""),
                wheelchair_boarding=row.get("wheelchair_boarding", ""),
            )
        )

    db_session.bulk_save_objects(records)
    db_session.commit()
    return len(records)


def load_routes(gtfs_dir: str, db_session) -> int:
    """Load routes.txt into the database."""
    path = os.path.join(gtfs_dir, "routes.txt")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping")
        return 0

    df = pd.read_csv(path, dtype=str)
    df = df.fillna("")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing routes"):
        route_type = None
        if row.get("route_type"):
            try:
                route_type = int(row["route_type"])
            except ValueError:
                pass

        records.append(
            Route(
                route_id=str(row["route_id"]),
                agency_id=row.get("agency_id", ""),
                route_short_name=row.get("route_short_name", ""),
                route_long_name=row.get("route_long_name", ""),
                route_type=route_type,
                route_desc=row.get("route_desc", ""),
                route_color=row.get("route_color", ""),
                route_text_color=row.get("route_text_color", ""),
            )
        )

    db_session.bulk_save_objects(records)
    db_session.commit()
    return len(records)


def load_trips(gtfs_dir: str, db_session) -> int:
    """Load trips.txt into the database."""
    path = os.path.join(gtfs_dir, "trips.txt")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping")
        return 0

    df = pd.read_csv(path, dtype=str)
    df = df.fillna("")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing trips"):
        direction_id = None
        if row.get("direction_id"):
            try:
                direction_id = int(row["direction_id"])
            except ValueError:
                pass

        wheelchair = None
        if row.get("wheelchair_accessible"):
            try:
                wheelchair = int(row["wheelchair_accessible"])
            except ValueError:
                pass

        bikes = None
        if row.get("bikes_allowed"):
            try:
                bikes = int(row["bikes_allowed"])
            except ValueError:
                pass

        records.append(
            Trip(
                trip_id=str(row["trip_id"]),
                route_id=str(row.get("route_id", "")),
                service_id=row.get("service_id", ""),
                trip_headsign=row.get("trip_headsign", ""),
                direction_id=direction_id,
                block_id=row.get("block_id", ""),
                shape_id=row.get("shape_id", ""),
                wheelchair_accessible=wheelchair,
                bikes_allowed=bikes,
            )
        )

    db_session.bulk_save_objects(records)
    db_session.commit()
    return len(records)


def load_stop_times(gtfs_dir: str, db_session, batch_size: int = 50000) -> int:
    """Load stop_times.txt into the database in batches."""
    path = os.path.join(gtfs_dir, "stop_times.txt")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping")
        return 0

    # Count total rows for progress bar
    with open(path, "r") as f:
        total_rows = sum(1 for _ in f) - 1  # Subtract header

    print(f"  Loading {total_rows:,} stop_times in batches of {batch_size:,}...")

    total_loaded = 0
    chunk_iter = pd.read_csv(path, dtype=str, chunksize=batch_size)

    with tqdm(total=total_rows, desc="  Processing stop_times") as pbar:
        for chunk in chunk_iter:
            chunk = chunk.fillna("")
            records = []

            for _, row in chunk.iterrows():
                stop_seq = 0
                if row.get("stop_sequence"):
                    try:
                        stop_seq = int(row["stop_sequence"])
                    except ValueError:
                        pass

                pickup = None
                if row.get("pickup_type"):
                    try:
                        pickup = int(row["pickup_type"])
                    except ValueError:
                        pass

                dropoff = None
                if row.get("drop_off_type"):
                    try:
                        dropoff = int(row["drop_off_type"])
                    except ValueError:
                        pass

                arrival_secs = parse_hhmmss_to_seconds(row.get("arrival_time", ""))
                departure_secs = parse_hhmmss_to_seconds(row.get("departure_time", ""))

                records.append(
                    StopTime(
                        trip_id=str(row["trip_id"]),
                        stop_id=str(row["stop_id"]),
                        arrival_time=row.get("arrival_time", ""),
                        departure_time=row.get("departure_time", ""),
                        arrival_secs=arrival_secs,
                        departure_secs=departure_secs,
                        stop_sequence=stop_seq,
                        stop_headsign=row.get("stop_headsign", ""),
                        pickup_type=pickup,
                        drop_off_type=dropoff,
                    )
                )

            db_session.bulk_save_objects(records)
            db_session.commit()
            total_loaded += len(records)
            pbar.update(len(records))

    return total_loaded


def create_indexes(db_session):
    """Create additional indexes for performance."""
    print("\nCreating additional indexes...")

    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_stop_times_stop_dep ON stop_times(stop_id, departure_secs)",
        "CREATE INDEX IF NOT EXISTS idx_stop_times_trip_seq ON stop_times(trip_id, stop_sequence)",
        "CREATE INDEX IF NOT EXISTS idx_trips_route ON trips(route_id)",
    ]

    for idx_sql in indexes:
        try:
            db_session.execute(text(idx_sql))
            db_session.commit()
            print(f"  Created: {idx_sql.split('idx_')[1].split(' ')[0]}")
        except Exception as e:
            print(f"  Warning: Could not create index: {e}")


def main():
    parser = argparse.ArgumentParser(description="Load GTFS data into PostgreSQL")
    parser.add_argument(
        "--gtfs-dir",
        default=os.path.join("data", "gtfs"),
        help="Path to GTFS directory (default: data/gtfs)",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing tables before loading",
    )
    args = parser.parse_args()

    gtfs_dir = args.gtfs_dir
    if not os.path.exists(gtfs_dir):
        print(f"Error: GTFS directory not found: {gtfs_dir}")
        sys.exit(1)

    print("=" * 60)
    print("GTFS Data Loader for PostgreSQL")
    print("=" * 60)
    print(f"\nGTFS Directory: {gtfs_dir}")
    print(f"Database URL: {engine.url}")

    # Test database connection
    print("\nTesting database connection...")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("  Database connection successful!")
    except Exception as e:
        print(f"\nError: Could not connect to database: {e}")
        print("\nMake sure PostgreSQL is running and the database exists.")
        print("You can set DATABASE_URL environment variable to configure the connection.")
        print("\nQuick setup with Docker:")
        print("  docker run -d --name gtfs-postgres \\")
        print("    -e POSTGRES_USER=gtfs_user \\")
        print("    -e POSTGRES_PASSWORD=gtfs_password \\")
        print("    -e POSTGRES_DB=gtfs_db \\")
        print("    -p 5432:5432 \\")
        print("    postgres:15")
        sys.exit(1)

    start_time = time.time()

    if args.drop_existing:
        print("\nDropping existing tables...")
        drop_tables()

    print("\nCreating tables...")
    create_tables()

    with get_db() as db:
        print("\nLoading GTFS data...")

        # Load in order of dependencies
        print("\n1. Loading stops...")
        stops_count = load_stops(gtfs_dir, db)
        print(f"   Loaded {stops_count:,} stops")

        print("\n2. Loading routes...")
        routes_count = load_routes(gtfs_dir, db)
        print(f"   Loaded {routes_count:,} routes")

        print("\n3. Loading trips...")
        trips_count = load_trips(gtfs_dir, db)
        print(f"   Loaded {trips_count:,} trips")

        print("\n4. Loading stop_times (this may take a while)...")
        stop_times_count = load_stop_times(gtfs_dir, db)
        print(f"   Loaded {stop_times_count:,} stop_times")

        # Create additional indexes
        create_indexes(db)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Loading complete!")
    print(f"Total time: {elapsed:.1f} seconds")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Stops:      {stops_count:,}")
    print(f"  Routes:     {routes_count:,}")
    print(f"  Trips:      {trips_count:,}")
    print(f"  Stop Times: {stop_times_count:,}")
    print("\nYou can now start the backend with:")
    print("  venv/bin/python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000")


if __name__ == "__main__":
    main()
