from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import requests

from .gtfs import GtfsData


@dataclass(frozen=True)
class RealtimeDelay:
    trip_id: str
    current_stop_id: str | None
    vehicle_timestamp: int
    estimated_delay_seconds: int
    method: str


def _scheduled_epoch_candidates(local_midnight_utc_epoch: int, scheduled_secs: int) -> list[int]:
    # candidate schedule timestamps for today, yesterday, tomorrow (to handle >24h and feed date mismatch)
    base = local_midnight_utc_epoch + scheduled_secs
    one_day = 86400
    return [base - one_day, base, base + one_day]


def estimate_delay_for_trip_from_vehicle_positions(
    gtfs: GtfsData,
    vehicle_positions_url: str,
    trip_id: str,
    tz,
    api_key: str | None = None,
    api_key_header: str = "Authorization",
    timeout_sec: int = 10,
) -> Optional[RealtimeDelay]:
    try:
        from google.transit import gtfs_realtime_pb2  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "gtfs-realtime-bindings is required for realtime parsing. Install it via pip."
        ) from e

    headers = {}
    if api_key:
        headers[api_key_header] = api_key

    resp = requests.get(vehicle_positions_url, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)

    trip_id = str(trip_id)

    # Find matching vehicle
    matched_vehicle = None
    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue
        v = entity.vehicle
        if not v.HasField("trip"):
            continue
        if v.trip.trip_id == trip_id:
            matched_vehicle = v
            break

    if matched_vehicle is None:
        return None

    vehicle_timestamp = int(matched_vehicle.timestamp) if matched_vehicle.HasField("timestamp") else 0
    current_stop_id = None
    if matched_vehicle.HasField("stop_id"):
        current_stop_id = str(matched_vehicle.stop_id)

    # If we can't locate a stop, we can't estimate schedule deviation.
    if not current_stop_id:
        return RealtimeDelay(
            trip_id=trip_id,
            current_stop_id=None,
            vehicle_timestamp=vehicle_timestamp,
            estimated_delay_seconds=0,
            method="vehicle_positions_no_stop_id",
        )

    st = gtfs.stop_times
    trip_times = st.loc[(st["trip_id"] == trip_id) & (st["stop_id"] == current_stop_id)]
    if trip_times.empty:
        return RealtimeDelay(
            trip_id=trip_id,
            current_stop_id=current_stop_id,
            vehicle_timestamp=vehicle_timestamp,
            estimated_delay_seconds=0,
            method="stop_not_found_in_static_gtfs",
        )

    scheduled_secs = int(trip_times.iloc[0]["departure_secs"]) if "departure_secs" in trip_times.columns else 0

    # Use local midnight in the feed timezone, converted to UTC epoch.
    import pytz

    now_local = datetime.now(tz)
    midnight_local = tz.localize(datetime(now_local.year, now_local.month, now_local.day, 0, 0, 0))
    midnight_utc = midnight_local.astimezone(pytz.utc)
    midnight_utc_epoch = int(midnight_utc.timestamp())

    if vehicle_timestamp <= 0:
        return RealtimeDelay(
            trip_id=trip_id,
            current_stop_id=current_stop_id,
            vehicle_timestamp=0,
            estimated_delay_seconds=0,
            method="vehicle_positions_no_timestamp",
        )

    candidates = _scheduled_epoch_candidates(midnight_utc_epoch, scheduled_secs)
    scheduled_epoch = min(candidates, key=lambda t: abs(vehicle_timestamp - t))

    delay = int(vehicle_timestamp - scheduled_epoch)
    # Clamp to reasonable range to avoid pathological results due to mismatched calendars
    delay = max(min(delay, 6 * 3600), -6 * 3600)

    return RealtimeDelay(
        trip_id=trip_id,
        current_stop_id=current_stop_id,
        vehicle_timestamp=vehicle_timestamp,
        estimated_delay_seconds=delay,
        method="vehicle_timestamp_minus_scheduled_departure",
    )
