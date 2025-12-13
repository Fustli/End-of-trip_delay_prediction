from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .settings import SETTINGS
from .gtfs import load_gtfs
from .router import ScheduledLeg, ScheduledPlan, find_fastest_scheduled_plan, reachable_stop_ids
from .realtime import RealtimeDelay, estimate_delay_for_trip_from_vehicle_positions
from .gnn_v4_inference import V4Predictor


@lru_cache(maxsize=4096)
def _reachable_cached(
    gtfs_dir: str,
    tz_name: str,
    from_sid: str,
    mt: int,
    hz: int,
    time_bucket: int,
) -> tuple[str, ...]:
    # Cache in 5-minute buckets because reachability depends on time-of-day.
    gtfs = load_gtfs(gtfs_dir)
    import pytz

    tz = pytz.timezone(tz_name)
    dt = datetime.fromtimestamp(time_bucket * 300, tz)
    ids = reachable_stop_ids(
        gtfs,
        from_sid,
        now_local=dt,
        max_transfers=int(mt),
        horizon_sec=int(hz),
        min_transfer_sec=120,
        max_boardings_per_stop=200,
    )
    ids.discard(str(from_sid))
    return tuple(sorted(ids))


class StopOut(BaseModel):
    stop_id: str
    stop_name: Optional[str] = None
    stop_lat: float
    stop_lon: float


class JourneyOut(BaseModel):
    class LegOut(BaseModel):
        from_stop_id: str
        to_stop_id: str
        trip_id: str
        route_id: str
        route_short_name: Optional[str] = None
        route_long_name: Optional[str] = None
        route_type: Optional[int] = None
        route_label: Optional[str] = None
        trip_headsign: Optional[str] = None
        scheduled_departure_time: str
        scheduled_arrival_time: str
        scheduled_travel_time_sec: int
        model_predicted_stop_delay_seconds: Optional[float] = None
        model_predicted_leg_delay_seconds: Optional[float] = None
        model_status: str
        realtime_delay_seconds: Optional[int] = None
        realtime_method: Optional[str] = None
        realtime_status: str

    legs: list[LegOut]
    total_scheduled_time_sec: int


app = FastAPI(title="Stop-to-stop routing + current delay")

# Serve frontend
app.mount("/static", StaticFiles(directory="app/frontend", html=True), name="frontend_static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse("app/frontend/index.html")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/stops", response_model=list[StopOut])
def stops():
    gtfs = load_gtfs(SETTINGS.gtfs_dir)
    s = gtfs.stops
    st = gtfs.stop_times

    required = {"stop_id", "stop_lat", "stop_lon"}
    if not required.issubset(set(s.columns)):
        raise HTTPException(status_code=500, detail=f"stops.txt missing columns: {sorted(required - set(s.columns))}")

    routable_stop_ids = set(st["stop_id"].astype(str).unique()) if "stop_id" in st.columns else set()

    out: list[StopOut] = []
    for _, row in s.iterrows():
        try:
            stop_id = str(row["stop_id"])
            if routable_stop_ids and stop_id not in routable_stop_ids:
                continue
            out.append(
                StopOut(
                    stop_id=stop_id,
                    stop_name=str(row["stop_name"]) if "stop_name" in s.columns and row.get("stop_name") is not None else None,
                    stop_lat=float(row["stop_lat"]),
                    stop_lon=float(row["stop_lon"]),
                )
            )
        except Exception:
            continue
    return out


@app.get("/api/reachable")
def reachable(
    from_stop_id: str = Query(..., min_length=1),
    max_transfers: int = Query(3, ge=0, le=6),
    horizon_sec: int = Query(6 * 3600, ge=300, le=24 * 3600),
    departure_time: Optional[str] = Query(None, description="Departure time in HH:MM:SS format (uses current time if not provided)"),
):
    try:
        import pytz

        tz = pytz.timezone(SETTINGS.timezone)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid timezone config: {SETTINGS.timezone}") from e

    # Parse custom departure time or use current time
    now_local = datetime.now(tz)
    if departure_time:
        try:
            # Parse HH:MM:SS and combine with today's date
            parts = departure_time.split(":")
            hour, minute = int(parts[0]), int(parts[1])
            second = int(parts[2]) if len(parts) > 2 else 0
            now_local = now_local.replace(hour=hour, minute=minute, second=second)
        except (ValueError, IndexError):
            pass  # Fall back to current time if parsing fails
    
    # In-memory cache: reachability depends on time-of-day. Cache in 5-minute buckets.
    # This keeps the UI snappy without introducing a database.
    bucket = int(now_local.timestamp() // 300)

    reachable_ids = _reachable_cached(
        SETTINGS.gtfs_dir,
        SETTINGS.timezone,
        str(from_stop_id),
        int(max_transfers),
        int(horizon_sec),
        bucket,
    )
    # Don't include the origin as a valid destination choice.
    return {
        "from_stop_id": str(from_stop_id),
        "max_transfers": int(max_transfers),
        "horizon_sec": int(horizon_sec),
        "reachable_stop_ids": list(reachable_ids),
        "cache_bucket": bucket,
    }


@app.get("/api/journey", response_model=JourneyOut)
def journey(
    from_stop_id: str = Query(..., min_length=1),
    to_stop_id: str = Query(..., min_length=1),
    max_transfers: int = Query(3, ge=0, le=6),
    departure_time: Optional[str] = Query(None, description="Departure time in HH:MM:SS format (uses current time if not provided)"),
):
    if from_stop_id == to_stop_id:
        raise HTTPException(status_code=400, detail="from_stop_id and to_stop_id must be different")

    gtfs = load_gtfs(SETTINGS.gtfs_dir)

    try:
        import pytz

        tz = pytz.timezone(SETTINGS.timezone)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid timezone config: {SETTINGS.timezone}") from e

    now_local = datetime.now(tz)
    
    # Parse custom departure time if provided
    if departure_time:
        try:
            parts = departure_time.split(":")
            hour, minute = int(parts[0]), int(parts[1])
            second = int(parts[2]) if len(parts) > 2 else 0
            now_local = now_local.replace(hour=hour, minute=minute, second=second)
        except (ValueError, IndexError):
            pass  # Fall back to current time if parsing fails

    plan = find_fastest_scheduled_plan(
        gtfs,
        from_stop_id,
        to_stop_id,
        now_local=now_local,
        max_transfers=int(max_transfers),
        horizon_sec=6 * 3600,
    )
    if plan is None:
        raise HTTPException(status_code=404, detail="No scheduled journey found within constraints")

    # Prepare V4 predictor (optional; requires exported artifacts)
    v4 = V4Predictor(SETTINGS.gnn_v4_artifacts_dir)
    v4_available = v4.available()
    if not v4_available:
        v4_nb = V4Predictor(os.path.join("notebook", "models", "gnn_v4"))
        if v4_nb.available():
            v4 = v4_nb
            v4_available = True

    # Current estimate: VehiclePositions can only give us a *current* delay for a trip the feed reports.
    # We attach realtime delay to the first leg if available.
    rt_delay_seconds: Optional[int] = None
    rt_method: Optional[str] = None
    rt_status = "unavailable"
    if SETTINGS.gtfs_rt_vehicle_positions_url:
        try:
            rt = estimate_delay_for_trip_from_vehicle_positions(
                gtfs=gtfs,
                vehicle_positions_url=SETTINGS.gtfs_rt_vehicle_positions_url,
                trip_id=plan.legs[0].trip_id,
                tz=tz,
                api_key=SETTINGS.gtfs_rt_api_key,
                api_key_header=SETTINGS.gtfs_rt_api_key_header,
            )
            if rt is None:
                rt_status = "no_vehicle_for_first_leg"
            else:
                rt_delay_seconds = rt.estimated_delay_seconds
                rt_method = rt.method
                rt_status = "ok"
        except Exception:
            rt_status = "error"

    # V4 model prediction
    # We predict an absolute delay (seconds) at the leg's arrival stop using a minimal 1-step context window.
    # Then we turn that into stop-to-stop delay by differencing with the best available previous delay.
    def cyclical_time(now_l):
        hour = now_l.hour
        dow = now_l.weekday()
        hour_sin = float(np.sin(2 * np.pi * hour / 24))
        hour_cos = float(np.cos(2 * np.pi * hour / 24))
        day_sin = float(np.sin(2 * np.pi * dow / 7))
        day_cos = float(np.cos(2 * np.pi * dow / 7))
        return hour_sin, hour_cos, day_sin, day_cos

    hour_sin, hour_cos, day_sin, day_cos = cyclical_time(now_local)

    # Progress is approximated from GTFS stop_sequence within the trip when possible.
    def progress_for_trip_stop(trip_id: str, stop_id: str) -> float:
        st = gtfs.stop_times
        rows = st.loc[st["trip_id"] == str(trip_id), ["stop_id", "stop_sequence"]].copy()
        if rows.empty:
            return 0.0
        rows["stop_sequence"] = rows["stop_sequence"].astype(int)
        trip_len = int(rows.shape[0])
        if trip_len <= 1:
            return 0.0
        m = rows.loc[rows["stop_id"] == str(stop_id)]
        if m.empty:
            return 0.0
        seq = int(m.iloc[0]["stop_sequence"])
        # stop_sequence in GTFS is typically 1-based
        seq0 = max(0, seq - 1)
        return float(seq0 / (trip_len - 1))

    legs_out: list[JourneyOut.LegOut] = []

    # Seed previous delay for leg1 from realtime if available; otherwise 0.
    prev_delay_for_next_leg = float(rt_delay_seconds) if rt_status == "ok" and rt_delay_seconds is not None else 0.0

    for i, leg in enumerate(plan.legs):
        # Approximate time delta from schedule (leg travel time)
        time_delta_sec = float(max(0, int(leg.scheduled_travel_time_sec)))
        prog = progress_for_trip_stop(leg.trip_id, leg.to_stop_id)

        model_abs_delay: Optional[float] = None
        model_leg_delay: Optional[float] = None
        model_status = "unavailable"
        if v4_available:
            try:
                model_abs_delay = v4.predict_stop_delay_seconds(
                    stop_id=leg.to_stop_id,
                    hour_sin=hour_sin,
                    hour_cos=hour_cos,
                    day_sin=day_sin,
                    day_cos=day_cos,
                    prev_delay_seconds=prev_delay_for_next_leg,
                    rolling_prev_delay_seconds=prev_delay_for_next_leg,
                    time_delta_seconds=time_delta_sec,
                    progress=prog,
                )
                if model_abs_delay is None:
                    model_status = "stop_not_in_v4_mapping"
                else:
                    model_leg_delay = float(model_abs_delay - prev_delay_for_next_leg)
                    model_status = "ok"
                    # propagate absolute delay forward as the context for next leg
                    prev_delay_for_next_leg = float(model_abs_delay)
            except Exception:
                model_status = "error"
        else:
            model_status = "artifacts_missing"
        legs_out.append(
            JourneyOut.LegOut(
                from_stop_id=leg.from_stop_id,
                to_stop_id=leg.to_stop_id,
                trip_id=leg.trip_id,
                route_id=leg.route_id,
                route_short_name=leg.route_short_name,
                route_long_name=leg.route_long_name,
                route_type=leg.route_type,
                route_label=(leg.route_short_name or leg.route_id),
                trip_headsign=leg.trip_headsign,
                scheduled_departure_time=leg.scheduled_departure_time,
                scheduled_arrival_time=leg.scheduled_arrival_time,
                scheduled_travel_time_sec=leg.scheduled_travel_time_sec,
                model_predicted_stop_delay_seconds=model_abs_delay,
                model_predicted_leg_delay_seconds=model_leg_delay,
                model_status=model_status,
                realtime_delay_seconds=(rt_delay_seconds if i == 0 and rt_status == "ok" else None),
                realtime_method=(rt_method if i == 0 and rt_status == "ok" else None),
                realtime_status=(rt_status if i == 0 else "n/a"),
            )
        )

    return JourneyOut(
        legs=legs_out,
        total_scheduled_time_sec=plan.total_scheduled_time_sec,
    )
