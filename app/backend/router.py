from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import heapq
from typing import Optional

import pandas as pd

from .gtfs import GtfsData


@dataclass(frozen=True)
class ScheduledLeg:
    trip_id: str
    route_id: str
    route_short_name: str | None
    route_long_name: str | None
    route_type: int | None
    trip_headsign: str | None
    from_stop_id: str
    to_stop_id: str
    depart_secs: int
    arrive_secs: int

    @property
    def scheduled_departure_time(self) -> str:
        return _secs_to_hhmmss(self.depart_secs)

    @property
    def scheduled_arrival_time(self) -> str:
        return _secs_to_hhmmss(self.arrive_secs)

    @property
    def scheduled_travel_time_sec(self) -> int:
        return int(self.arrive_secs - self.depart_secs)


@dataclass(frozen=True)
class ScheduledPlan:
    legs: list[ScheduledLeg]

    @property
    def arrival_secs(self) -> int:
        return self.legs[-1].arrive_secs

    @property
    def departure_secs(self) -> int:
        return self.legs[0].depart_secs

    @property
    def total_scheduled_time_sec(self) -> int:
        return int(self.arrival_secs - self.departure_secs)


def _secs_to_hhmmss(secs: int) -> str:
    # Keep GTFS-style hours (can exceed 24)
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def find_fastest_scheduled_plan(
    gtfs: GtfsData,
    from_stop_id: str,
    to_stop_id: str,
    now_local: datetime,
    *,
    max_transfers: int = 1,
    horizon_sec: int = 2 * 3600,
    min_transfer_sec: int = 120,
    max_first_leg_trips: int = 200,
) -> Optional[ScheduledPlan]:
    # For >1 transfers we use a bounded earliest-arrival search.
    if max_transfers > 1:
        return find_fastest_scheduled_plan_multi_transfer(
            gtfs,
            from_stop_id,
            to_stop_id,
            now_local=now_local,
            max_transfers=max_transfers,
            horizon_sec=horizon_sec,
            min_transfer_sec=min_transfer_sec,
            max_boardings_per_stop=max_first_leg_trips,
        )

    # 0 transfers (direct)
    direct = find_fastest_direct_leg(gtfs, from_stop_id, to_stop_id, now_local=now_local)
    if direct is not None:
        return ScheduledPlan(legs=[direct])

    if max_transfers <= 0:
        return None

    # 1 transfer (two legs)
    st = gtfs.stop_times
    now_secs = now_local.hour * 3600 + now_local.minute * 60 + now_local.second
    latest_depart = now_secs + int(horizon_sec)

    from_stop_id = str(from_stop_id)
    to_stop_id = str(to_stop_id)

    from_rows = st.loc[
        (st["stop_id"] == from_stop_id),
        ["trip_id", "stop_sequence", "departure_secs"],
    ].copy()

    if from_rows.empty:
        return None

    one_day = 86400
    from_rows["depart_mod"] = from_rows["departure_secs"].astype(int)
    # Treat departures earlier than now as next-day departures
    from_rows.loc[from_rows["depart_mod"] < now_secs, "depart_mod"] += one_day

    from_rows = from_rows.loc[(from_rows["depart_mod"] >= now_secs) & (from_rows["depart_mod"] <= latest_depart)]
    from_rows = from_rows.sort_values(["depart_mod"]).head(max(1, int(max_first_leg_trips)))

    if from_rows.empty:
        return None

    # Build best (earliest) arrival to each potential transfer stop, along with the first leg info.
    best_to_transfer: dict[str, tuple[int, ScheduledLeg]] = {}
    trips = gtfs.trips
    routes = gtfs.routes

    def _route_trip_info(trip_id: str) -> tuple[str, str | None, str | None, int | None, str | None]:
        trip_row = trips.loc[trips["trip_id"] == trip_id]
        if trip_row.empty:
            return ("", None, None, None, None)
        route_id = str(trip_row.iloc[0].get("route_id"))
        trip_headsign = trip_row.iloc[0].get("trip_headsign")

        route_row = routes.loc[routes["route_id"].astype(str) == route_id]
        route_short_name = None
        route_long_name = None
        route_type = None
        if not route_row.empty:
            route_short_name = route_row.iloc[0].get("route_short_name")
            route_long_name = route_row.iloc[0].get("route_long_name")
            try:
                rt = route_row.iloc[0].get("route_type")
                route_type = int(rt) if rt is not None and not pd.isna(rt) else None
            except Exception:
                route_type = None

        return (
            route_id,
            str(route_short_name) if route_short_name is not None and not pd.isna(route_short_name) else None,
            str(route_long_name) if route_long_name is not None and not pd.isna(route_long_name) else None,
            route_type,
            str(trip_headsign) if trip_headsign is not None and not pd.isna(trip_headsign) else None,
        )

    for _, fr in from_rows.iterrows():
        trip_id = str(fr["trip_id"])
        from_seq = int(fr["stop_sequence"])
        depart_secs = int(fr["depart_mod"])

        # All later stops on that trip can be transfer candidates
        later = st.loc[
            (st["trip_id"] == trip_id) & (st["stop_sequence"] > from_seq),
            ["stop_id", "arrival_secs", "stop_sequence"],
        ].copy()
        if later.empty:
            continue

        route_id, rsn, rln, rtype, headsign = _route_trip_info(trip_id)
        if not route_id:
            continue

        later["arrive_mod"] = later["arrival_secs"].astype(int)
        later.loc[later["arrive_mod"] < depart_secs, "arrive_mod"] += one_day

        for _, lr in later.iterrows():
            transfer_stop_id = str(lr["stop_id"])
            if transfer_stop_id == from_stop_id:
                continue
            arrive_transfer = int(lr["arrive_mod"])
            # Skip degenerate transfer if it already is destination; handled by direct search
            if transfer_stop_id == to_stop_id:
                continue

            leg1 = ScheduledLeg(
                trip_id=trip_id,
                route_id=route_id,
                route_short_name=rsn,
                route_long_name=rln,
                route_type=rtype,
                trip_headsign=headsign,
                from_stop_id=from_stop_id,
                to_stop_id=transfer_stop_id,
                depart_secs=depart_secs,
                arrive_secs=arrive_transfer,
            )

            prev = best_to_transfer.get(transfer_stop_id)
            if prev is None or arrive_transfer < prev[0]:
                best_to_transfer[transfer_stop_id] = (arrive_transfer, leg1)

    if not best_to_transfer:
        return None

    # Try transfer stops by earliest arrival to keep it fast.
    transfer_candidates = sorted(best_to_transfer.items(), key=lambda kv: kv[1][0])[:500]

    best_plan: Optional[ScheduledPlan] = None
    best_arrival = None
    for transfer_stop_id, (arrive_transfer, leg1) in transfer_candidates:
        depart2_not_before = arrive_transfer + int(min_transfer_sec)
        leg2 = find_fastest_direct_leg_by_secs(gtfs, transfer_stop_id, to_stop_id, now_secs=depart2_not_before)
        if leg2 is None:
            continue
        plan = ScheduledPlan(legs=[leg1, leg2])
        if best_arrival is None or plan.arrival_secs < best_arrival:
            best_plan = plan
            best_arrival = plan.arrival_secs

    return best_plan


def reachable_stop_ids(
    gtfs: GtfsData,
    from_stop_id: str,
    now_local: datetime,
    *,
    max_transfers: int = 3,
    horizon_sec: int = 2 * 3600,
    min_transfer_sec: int = 120,
    max_boardings_per_stop: int = 200,
) -> set[str]:
    from_stop_id = str(from_stop_id)

    now_secs = now_local.hour * 3600 + now_local.minute * 60 + now_local.second
    latest_time = now_secs + int(horizon_sec)
    max_legs = max(1, int(max_transfers) + 1)
    one_day = 86400

    st = gtfs.stop_times
    if st.empty:
        return set()

    stop_index: dict[str, pd.DataFrame] = {
        sid: df.sort_values(["departure_secs", "stop_sequence"]).loc[:, ["trip_id", "stop_sequence", "departure_secs"]]
        for sid, df in st.groupby("stop_id", sort=False)
    }
    trip_index: dict[str, pd.DataFrame] = {
        tid: df.sort_values(["stop_sequence"]).loc[:, ["stop_id", "stop_sequence", "arrival_secs"]]
        for tid, df in st.groupby("trip_id", sort=False)
    }

    def _normalize_depart_times(dep_mod: pd.Series, ref_time: int) -> pd.Series:
        base_day = ref_time // one_day
        ref_mod = ref_time % one_day
        dep_mod_int = dep_mod.astype(int)
        dep_day = base_day + (dep_mod_int < ref_mod).astype(int)
        return dep_mod_int + dep_day * one_day

    def _normalize_arrival(arr_mod: int, depart_time: int) -> int:
        day = depart_time // one_day
        arrive = int(arr_mod) + day * one_day
        if arrive < depart_time:
            arrive += one_day
        return arrive

    best: dict[tuple[str, int], int] = {(from_stop_id, 0): int(now_secs)}
    pq: list[tuple[int, str, int]] = [(int(now_secs), from_stop_id, 0)]
    reached: set[str] = {from_stop_id}

    while pq:
        t, stop_id, legs_used = heapq.heappop(pq)
        key = (stop_id, legs_used)
        if best.get(key) != t:
            continue
        if t > latest_time:
            continue
        if legs_used >= max_legs:
            continue

        stop_rows = stop_index.get(stop_id)
        if stop_rows is None or stop_rows.empty:
            continue

        # Only allow boarding another vehicle after transfer buffer (except for the first leg).
        board_not_before = t if legs_used == 0 else t + int(min_transfer_sec)
        depart_times = _normalize_depart_times(stop_rows["departure_secs"], board_not_before)

        mask = (depart_times >= board_not_before) & (depart_times <= latest_time)
        if not mask.any():
            continue

        cand = stop_rows.loc[mask].copy()
        cand["depart_time"] = depart_times.loc[mask].astype(int)
        cand = cand.sort_values(["depart_time"]).head(max(1, int(max_boardings_per_stop)))

        for _, br in cand.iterrows():
            trip_id = str(br["trip_id"])
            board_seq = int(br["stop_sequence"])
            depart_time = int(br["depart_time"])

            trip_rows = trip_index.get(trip_id)
            if trip_rows is None or trip_rows.empty:
                continue

            later = trip_rows.loc[trip_rows["stop_sequence"] > board_seq]
            if later.empty:
                continue

            for _, lr in later.iterrows():
                nxt_stop = str(lr["stop_id"])
                arrive_time = _normalize_arrival(int(lr["arrival_secs"]), depart_time)
                if arrive_time > latest_time:
                    continue

                reached.add(nxt_stop)
                nk = (nxt_stop, legs_used + 1)
                prev_best = best.get(nk)
                if prev_best is None or arrive_time < prev_best:
                    best[nk] = arrive_time
                    heapq.heappush(pq, (arrive_time, nxt_stop, legs_used + 1))

    return reached


def find_fastest_scheduled_plan_multi_transfer(
    gtfs: GtfsData,
    from_stop_id: str,
    to_stop_id: str,
    now_local: datetime,
    *,
    max_transfers: int = 3,
    horizon_sec: int = 2 * 3600,
    min_transfer_sec: int = 120,
    max_boardings_per_stop: int = 200,
) -> Optional[ScheduledPlan]:
    from_stop_id = str(from_stop_id)
    to_stop_id = str(to_stop_id)
    if from_stop_id == to_stop_id:
        return None

    now_secs = now_local.hour * 3600 + now_local.minute * 60 + now_local.second
    latest_time = now_secs + int(horizon_sec)
    max_legs = max(1, int(max_transfers) + 1)
    one_day = 86400

    st = gtfs.stop_times
    if st.empty:
        return None

    stop_index: dict[str, pd.DataFrame] = {
        sid: df.sort_values(["departure_secs", "stop_sequence"]).loc[:, ["trip_id", "stop_sequence", "departure_secs"]]
        for sid, df in st.groupby("stop_id", sort=False)
    }
    trip_index: dict[str, pd.DataFrame] = {
        tid: df.sort_values(["stop_sequence"]).loc[:, ["stop_id", "stop_sequence", "arrival_secs"]]
        for tid, df in st.groupby("trip_id", sort=False)
    }

    trips = gtfs.trips
    routes = gtfs.routes

    def _route_trip_info(trip_id: str) -> tuple[str, str | None, str | None, int | None, str | None]:
        trip_row = trips.loc[trips["trip_id"] == trip_id]
        if trip_row.empty:
            return ("", None, None, None, None)
        route_id = str(trip_row.iloc[0].get("route_id"))
        trip_headsign = trip_row.iloc[0].get("trip_headsign")

        route_row = routes.loc[routes["route_id"].astype(str) == route_id]
        route_short_name = None
        route_long_name = None
        route_type = None
        if not route_row.empty:
            route_short_name = route_row.iloc[0].get("route_short_name")
            route_long_name = route_row.iloc[0].get("route_long_name")
            try:
                rt = route_row.iloc[0].get("route_type")
                route_type = int(rt) if rt is not None and not pd.isna(rt) else None
            except Exception:
                route_type = None

        return (
            route_id,
            str(route_short_name) if route_short_name is not None and not pd.isna(route_short_name) else None,
            str(route_long_name) if route_long_name is not None and not pd.isna(route_long_name) else None,
            route_type,
            str(trip_headsign) if trip_headsign is not None and not pd.isna(trip_headsign) else None,
        )

    def _normalize_depart_times(dep_mod: pd.Series, ref_time: int) -> pd.Series:
        base_day = ref_time // one_day
        ref_mod = ref_time % one_day
        dep_mod_int = dep_mod.astype(int)
        dep_day = base_day + (dep_mod_int < ref_mod).astype(int)
        return dep_mod_int + dep_day * one_day

    def _normalize_arrival(arr_mod: int, depart_time: int) -> int:
        day = depart_time // one_day
        arrive = int(arr_mod) + day * one_day
        if arrive < depart_time:
            arrive += one_day
        return arrive

    best: dict[tuple[str, int], int] = {(from_stop_id, 0): int(now_secs)}
    prev: dict[tuple[str, int], tuple[tuple[str, int], ScheduledLeg]] = {}
    pq: list[tuple[int, str, int]] = [(int(now_secs), from_stop_id, 0)]

    best_dest_key: Optional[tuple[str, int]] = None
    best_dest_arrival: Optional[int] = None

    while pq:
        t, stop_id, legs_used = heapq.heappop(pq)
        key = (stop_id, legs_used)
        if best.get(key) != t:
            continue
        if t > latest_time:
            continue

        if stop_id == to_stop_id and legs_used > 0:
            if best_dest_arrival is None or t < best_dest_arrival:
                best_dest_arrival = t
                best_dest_key = key

        if legs_used >= max_legs:
            continue
        if best_dest_arrival is not None and t >= best_dest_arrival:
            continue

        stop_rows = stop_index.get(stop_id)
        if stop_rows is None or stop_rows.empty:
            continue

        board_not_before = t if legs_used == 0 else t + int(min_transfer_sec)
        depart_times = _normalize_depart_times(stop_rows["departure_secs"], board_not_before)
        mask = (depart_times >= board_not_before) & (depart_times <= latest_time)
        if not mask.any():
            continue

        cand = stop_rows.loc[mask].copy()
        cand["depart_time"] = depart_times.loc[mask].astype(int)
        cand = cand.sort_values(["depart_time"]).head(max(1, int(max_boardings_per_stop)))

        for _, br in cand.iterrows():
            trip_id = str(br["trip_id"])
            board_seq = int(br["stop_sequence"])
            depart_time = int(br["depart_time"])

            trip_rows = trip_index.get(trip_id)
            if trip_rows is None or trip_rows.empty:
                continue

            later = trip_rows.loc[trip_rows["stop_sequence"] > board_seq]
            if later.empty:
                continue

            route_id, rsn, rln, rtype, headsign = _route_trip_info(trip_id)
            if not route_id:
                continue

            for _, lr in later.iterrows():
                nxt_stop = str(lr["stop_id"])
                arrive_time = _normalize_arrival(int(lr["arrival_secs"]), depart_time)
                if arrive_time > latest_time:
                    continue
                if best_dest_arrival is not None and arrive_time >= best_dest_arrival:
                    continue

                nk = (nxt_stop, legs_used + 1)
                prev_best = best.get(nk)
                if prev_best is None or arrive_time < prev_best:
                    best[nk] = arrive_time
                    leg = ScheduledLeg(
                        trip_id=trip_id,
                        route_id=route_id,
                        route_short_name=rsn,
                        route_long_name=rln,
                        route_type=rtype,
                        trip_headsign=headsign,
                        from_stop_id=stop_id,
                        to_stop_id=nxt_stop,
                        depart_secs=depart_time,
                        arrive_secs=arrive_time,
                    )
                    prev[nk] = (key, leg)
                    heapq.heappush(pq, (arrive_time, nxt_stop, legs_used + 1))

    if best_dest_key is None:
        return None

    # Reconstruct best plan
    legs: list[ScheduledLeg] = []
    cur = best_dest_key
    while cur != (from_stop_id, 0):
        p = prev.get(cur)
        if p is None:
            return None
        prev_key, leg = p
        legs.append(leg)
        cur = prev_key
    legs.reverse()
    return ScheduledPlan(legs=legs)


def find_fastest_direct_leg(
    gtfs: GtfsData,
    from_stop_id: str,
    to_stop_id: str,
    now_local: datetime,
) -> Optional[ScheduledLeg]:
    now_secs = now_local.hour * 3600 + now_local.minute * 60 + now_local.second
    return find_fastest_direct_leg_by_secs(gtfs, from_stop_id, to_stop_id, now_secs=now_secs)


def find_fastest_direct_leg_by_secs(
    gtfs: GtfsData,
    from_stop_id: str,
    to_stop_id: str,
    *,
    now_secs: int,
    limit: int = 1,
) -> Optional[ScheduledLeg]:
    st = gtfs.stop_times
    trips = gtfs.trips
    routes = gtfs.routes

    from_stop_id = str(from_stop_id)
    to_stop_id = str(to_stop_id)

    from_rows = st.loc[st["stop_id"] == from_stop_id, ["trip_id", "stop_sequence", "departure_secs"]]
    to_rows = st.loc[st["stop_id"] == to_stop_id, ["trip_id", "stop_sequence", "arrival_secs"]]

    if from_rows.empty or to_rows.empty:
        return None

    merged = from_rows.merge(to_rows, on="trip_id", suffixes=("_from", "_to"))
    merged = merged.loc[merged["stop_sequence_from"] < merged["stop_sequence_to"]].copy()
    if merged.empty:
        return None

    one_day = 86400
    merged["depart_mod"] = merged["departure_secs"].astype(int)
    merged.loc[merged["depart_mod"] < now_secs, "depart_mod"] += one_day
    merged["arrive_mod"] = merged["arrival_secs"].astype(int)
    merged.loc[merged["arrive_mod"] < merged["depart_mod"], "arrive_mod"] += one_day

    merged = merged.loc[merged["depart_mod"] >= now_secs]
    if merged.empty:
        return None

    merged["arrive_secs"] = merged["arrive_mod"].astype(int)
    merged["depart_secs"] = merged["depart_mod"].astype(int)
    merged["travel_sec"] = merged["arrive_secs"] - merged["depart_secs"]

    merged = merged.sort_values(["arrive_secs", "travel_sec"]).head(max(1, int(limit)))

    for _, row in merged.iterrows():
        trip_id = str(row["trip_id"])
        trip_row = trips.loc[trips["trip_id"] == trip_id]
        if trip_row.empty:
            continue

        route_id = str(trip_row.iloc[0].get("route_id"))
        trip_headsign = trip_row.iloc[0].get("trip_headsign")

        route_row = routes.loc[routes["route_id"].astype(str) == route_id]
        route_short_name = None
        route_long_name = None
        route_type = None
        if not route_row.empty:
            route_short_name = route_row.iloc[0].get("route_short_name")
            route_long_name = route_row.iloc[0].get("route_long_name")
            try:
                rt = route_row.iloc[0].get("route_type")
                route_type = int(rt) if rt is not None and not pd.isna(rt) else None
            except Exception:
                route_type = None

        depart_secs = int(row["depart_secs"])
        arrive_secs = int(row["arrive_secs"])

        return ScheduledLeg(
            trip_id=trip_id,
            route_id=route_id,
            route_short_name=str(route_short_name) if route_short_name is not None and not pd.isna(route_short_name) else None,
            route_long_name=str(route_long_name) if route_long_name is not None and not pd.isna(route_long_name) else None,
            route_type=route_type,
            trip_headsign=str(trip_headsign) if trip_headsign is not None and not pd.isna(trip_headsign) else None,
            from_stop_id=from_stop_id,
            to_stop_id=to_stop_id,
            depart_secs=depart_secs,
            arrive_secs=arrive_secs,
        )

    return None
