from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Paths
    gtfs_dir: str = os.getenv("GTFS_DIR", os.path.join("data", "gtfs"))

    # GNN Model V4 artifacts (exported from notebook)
    gnn_v4_artifacts_dir: str = os.getenv("GNN_V4_ARTIFACTS_DIR", os.path.join("models", "gnn_v4"))

    # Realtime (optional)
    gtfs_rt_vehicle_positions_url: str | None = os.getenv("GTFS_RT_VEHICLE_POSITIONS_URL") or None

    # If your provider requires an auth header (optional)
    gtfs_rt_api_key: str | None = os.getenv("GTFS_RT_API_KEY") or None
    gtfs_rt_api_key_header: str = os.getenv("GTFS_RT_API_KEY_HEADER", "Authorization")

    # Timezone for schedule interpretation
    timezone: str = os.getenv("GTFS_TIMEZONE", "Europe/Budapest")


SETTINGS = Settings()
