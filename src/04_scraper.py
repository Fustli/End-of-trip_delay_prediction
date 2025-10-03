import os
import time
import datetime
import csv
import requests
import pytz
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import VehicleUpdate, Base
from google.transit import gtfs_realtime_pb2

# Load environment variables
load_dotenv()
API_KEY = os.getenv("BKK_API_KEY")
BASE_URL = os.getenv("BKK_BASE_URL")
VEHICLE_URL = f"{BASE_URL}/gtfs-rt/full/VehiclePositions.pb?key={API_KEY}"

# Path to GTFS stop_times
STOP_TIMES_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "gtfs", "stop_times.txt")

# Setup database
db_path = os.path.join(os.path.dirname(__file__), "..", "data", "transit.db")
engine = create_engine(f"sqlite:///{db_path}")
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

# Budapest timezone
BUDAPEST = pytz.timezone("Europe/Budapest")

# Load stop times into memory as (trip_id, stop_sequence) -> departure_seconds
print("[INFO] Loading scheduled stop times...")
scheduled_stop_times = {}
with open(STOP_TIMES_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        trip_id = row["trip_id"]
        stop_sequence = int(row["stop_sequence"])
        departure_time = row["departure_time"]
        h, m, s = map(int, departure_time.split(":"))
        departure_seconds = h * 3600 + m * 60 + s
        scheduled_stop_times[(trip_id, stop_sequence)] = departure_seconds
print(f"[INFO] Loaded {len(scheduled_stop_times)} stop times.")

def fetch_vehicle_positions():
    """Fetch GTFS-realtime VehiclePositions and parse protobuf."""
    feed = gtfs_realtime_pb2.FeedMessage()
    try:
        resp = requests.get(VEHICLE_URL, timeout=10)
        resp.raise_for_status()
        feed.ParseFromString(resp.content)
        return feed.entity
    except Exception as e:
        print(f"[ERROR] Failed to fetch vehicle positions: {e}")
        return []

def calculate_delay(trip_id, stop_sequence, real_timestamp):
    """Compute delay in seconds using stop sequence."""
    key = (trip_id, stop_sequence)
    if key not in scheduled_stop_times:
        return None

    scheduled_seconds = scheduled_stop_times[key]
    dt_local = datetime.datetime.fromtimestamp(real_timestamp, BUDAPEST)
    real_seconds = dt_local.hour * 3600 + dt_local.minute * 60 + dt_local.second

    # Handle >24:00 scheduled times
    if scheduled_seconds >= 24 * 3600:
        scheduled_seconds -= 24 * 3600
        if real_seconds < 12 * 3600:
            real_seconds += 24 * 3600

    return real_seconds - scheduled_seconds

def save_to_db(entities):
    """Save vehicle positions and delays into SQLite."""
    if not entities:
        return

    session = Session()
    timestamp = datetime.datetime.now(BUDAPEST)
    count = 0

    for entity in entities:
        if entity.HasField("vehicle"):
            v = entity.vehicle
            trip_id = v.trip.trip_id if v.HasField("trip") else None
            vehicle_id = v.vehicle.id if v.HasField("vehicle") else None
            stop_sequence = v.current_stop_sequence if v.HasField("current_stop_sequence") else None
            latitude = v.position.latitude if v.HasField("position") else None
            longitude = v.position.longitude if v.HasField("position") else None

            delay_seconds = calculate_delay(trip_id, stop_sequence, v.timestamp) if trip_id and stop_sequence is not None else None

            update = VehicleUpdate(
                timestamp=timestamp,
                trip_id=trip_id,
                vehicle_id=vehicle_id,
                last_stop_id=v.stop_id if v.HasField("stop_id") else None,
                delay_seconds=delay_seconds,
                latitude=latitude,
                longitude=longitude
            )
            session.add(update)
            count += 1

    session.commit()
    session.close()
    print(f"[INFO] Saved {count} vehicle updates at {timestamp.isoformat()} ✅")

def run_scraper(interval_sec=60):
    print("[INFO] Starting GTFS-realtime scraper using stop sequence for delays...")
    while True:
        vehicle_entities = fetch_vehicle_positions()
        save_to_db(vehicle_entities)
        time.sleep(interval_sec)

if __name__ == "__main__":
    run_scraper(interval_sec=60)
