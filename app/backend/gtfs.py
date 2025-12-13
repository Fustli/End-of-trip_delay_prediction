from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Optional

import pandas as pd


# Flag to control data source (set by settings or environment)
USE_DATABASE = os.getenv("USE_DATABASE", "false").lower() in ("true", "1", "yes")


def _read_gtfs_csv(gtfs_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(gtfs_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing GTFS file: {path}")
    return pd.read_csv(path, low_memory=False)


def parse_hhmmss_to_seconds(value: str) -> int:
    # GTFS times can exceed 24:00:00
    if pd.isna(value):
        return 0
    s = str(value).strip()
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid HH:MM:SS time: {value}")
    h, m, sec = (int(parts[0]), int(parts[1]), int(parts[2]))
    return h * 3600 + m * 60 + sec


@dataclass(frozen=True)
class GtfsData:
    stops: pd.DataFrame
    routes: pd.DataFrame
    trips: pd.DataFrame
    stop_times: pd.DataFrame


def _load_gtfs_from_database() -> GtfsData:
    """Load GTFS data from PostgreSQL database."""
    from .database import engine
    
    print("Loading GTFS data from database...")
    
    # Load each table into a DataFrame
    stops = pd.read_sql_table("stops", engine)
    routes = pd.read_sql_table("routes", engine)
    trips = pd.read_sql_table("trips", engine)
    stop_times = pd.read_sql_table("stop_times", engine)
    
    # Ensure string types for IDs
    stops["stop_id"] = stops["stop_id"].astype(str)
    trips["trip_id"] = trips["trip_id"].astype(str)
    trips["route_id"] = trips["route_id"].astype(str)
    stop_times["trip_id"] = stop_times["trip_id"].astype(str)
    stop_times["stop_id"] = stop_times["stop_id"].astype(str)
    
    print(f"Loaded from DB: {len(stops)} stops, {len(routes)} routes, {len(trips)} trips, {len(stop_times)} stop_times")
    
    return GtfsData(stops=stops, routes=routes, trips=trips, stop_times=stop_times)


def _load_gtfs_from_csv(gtfs_dir: str) -> GtfsData:
    """Load GTFS data from CSV files."""
    stops = _read_gtfs_csv(gtfs_dir, "stops.txt")
    routes = _read_gtfs_csv(gtfs_dir, "routes.txt")
    trips = _read_gtfs_csv(gtfs_dir, "trips.txt")
    stop_times = _read_gtfs_csv(gtfs_dir, "stop_times.txt")

    # Normalize dtypes we rely on
    for c in ("stop_id",):
        if c in stops.columns:
            stops[c] = stops[c].astype(str)

    for c in ("trip_id", "route_id"):
        if c in trips.columns:
            trips[c] = trips[c].astype(str)

    for c in ("trip_id", "stop_id"):
        if c in stop_times.columns:
            stop_times[c] = stop_times[c].astype(str)

    if "stop_sequence" in stop_times.columns:
        stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce").fillna(0).astype(int)

    # Parse times to seconds for fast routing
    if "departure_time" in stop_times.columns:
        stop_times["departure_secs"] = stop_times["departure_time"].map(parse_hhmmss_to_seconds)
    else:
        stop_times["departure_secs"] = 0

    if "arrival_time" in stop_times.columns:
        stop_times["arrival_secs"] = stop_times["arrival_time"].map(parse_hhmmss_to_seconds)
    else:
        stop_times["arrival_secs"] = stop_times["departure_secs"]

    return GtfsData(stops=stops, routes=routes, trips=trips, stop_times=stop_times)


@lru_cache(maxsize=1)
def load_gtfs(gtfs_dir: str) -> GtfsData:
    """
    Load GTFS data from either database or CSV files.
    
    Set USE_DATABASE=true environment variable to load from PostgreSQL.
    """
    if USE_DATABASE:
        try:
            from .database import check_database_populated
            if check_database_populated():
                return _load_gtfs_from_database()
            else:
                print("Database is empty, falling back to CSV files...")
        except Exception as e:
            print(f"Database error, falling back to CSV files: {e}")
    
    return _load_gtfs_from_csv(gtfs_dir)


def get_stop_coords(stops: pd.DataFrame, stop_id: str) -> Optional[tuple[float, float]]:
    row = stops.loc[stops["stop_id"] == str(stop_id)]
    if row.empty:
        return None
    lat = float(row.iloc[0].get("stop_lat"))
    lon = float(row.iloc[0].get("stop_lon"))
    return (lat, lon)
