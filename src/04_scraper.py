"""
BKK Real-time Vehicle Data Scraper
----------------------------------
Collects real-time bus positions and calculates delays using GTFS schedules.
Designed for GNN training data collection.
"""

import time
import datetime
import csv
import requests
import pytz
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from google.transit import gtfs_realtime_pb2

# Import configuration
import config

# =============================================================================
# DATABASE MODEL
# =============================================================================

Base = declarative_base()

class VehicleUpdate(Base):
    """
    Database model for storing vehicle position and delay data.
    Contains only the 6 fields required for GNN training.
    """
    __tablename__ = 'vehicle_updates'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)  # Query timestamp
    trip_id = Column(String)                      # Journey identifier
    vehicle_id = Column(String)                   # Vehicle identifier
    last_stop_id = Column(String)                 # Last visited stop ID
    delay_seconds = Column(Integer)               # Calculated delay in seconds
    latitude = Column(Float)                      # Vehicle position
    longitude = Column(Float)                     # Vehicle position

# Database setup
engine = create_engine(config.DATABASE_URL)
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

# Timezone setup
BUDAPEST = pytz.timezone(config.TIMEZONE)

# =============================================================================
# DELAY CALCULATOR
# =============================================================================

class DelayCalculator:
    """
    Calculates vehicle delays by comparing real-time positions with GTFS schedule.
    Handles midnight crossovers and filters unrealistic delay values.
    """
    
    def __init__(self, stop_times_path):
        """Initialize with GTFS stop_times data."""
        self.schedule = self._load_stop_times(stop_times_path)
        
    def _load_stop_times(self, stop_times_path):
        """
        Load GTFS stop_times.txt and build schedule lookup dictionary.
        """
        print("[INFO] Loading GTFS stop_times.txt...")
        schedule = {}
        
        try:
            with open(stop_times_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trip_id = row['trip_id']
                    stop_sequence = int(row['stop_sequence'])
                    departure_time = row['departure_time']
                    
                    # Convert HH:MM:SS to seconds since midnight
                    h, m, s = map(int, departure_time.split(':'))
                    departure_seconds = h * 3600 + m * 60 + s
                    
                    # Build nested dictionary
                    if trip_id not in schedule:
                        schedule[trip_id] = {}
                    schedule[trip_id][stop_sequence] = departure_seconds
            
            print(f"[INFO] Loaded {len(schedule)} trips with {sum(len(stops) for stops in schedule.values())} stop sequences")
            return schedule
            
        except Exception as e:
            print(f"[ERROR] Failed to load stop_times.txt: {e}")
            return {}
    
    def calculate_delay(self, trip_id, current_stop_sequence, vehicle_timestamp):
        """
        Calculate delay by comparing scheduled vs actual departure time.
        """
        # Validate input parameters
        if not trip_id or current_stop_sequence is None:
            return None
            
        # Check if trip exists in schedule
        if trip_id not in self.schedule:
            return None
        
        # Check if stop sequence exists for this trip
        if current_stop_sequence not in self.schedule[trip_id]:
            return None
        
        # Get scheduled departure time
        scheduled_departure = self.schedule[trip_id][current_stop_sequence]
        
        # Convert vehicle timestamp to seconds since midnight
        vehicle_time = datetime.datetime.fromtimestamp(vehicle_timestamp, BUDAPEST)
        current_seconds = vehicle_time.hour * 3600 + vehicle_time.minute * 60 + vehicle_time.second
        
        # Calculate raw time difference
        time_diff = current_seconds - scheduled_departure
        
        # Handle midnight crossover
        if time_diff > 12 * 3600:
            time_diff -= 24 * 3600
        elif time_diff < -12 * 3600:
            time_diff += 24 * 3600
        
        if time_diff < config.MIN_REALISTIC_DELAY or time_diff > config.MAX_REALISTIC_DELAY:
            if config.VERBOSE_FILTERING:
                print(f"[FILTER] Unrealistic delay filtered: {time_diff}s - Trip: {trip_id}")
            return None
    
        return time_diff

# Initialize global delay calculator
delay_calculator = DelayCalculator(config.STOP_TIMES_FILE)

# =============================================================================
# DATA COLLECTION FUNCTIONS
# =============================================================================

def fetch_gtfs_rt_feed():
    """
    Fetch and parse GTFS-realtime protobuf feed.
    """
    feed = gtfs_realtime_pb2.FeedMessage()
    try:
        response = requests.get(config.VEHICLE_URL, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        feed.ParseFromString(response.content)
        return feed.entity
    except Exception as e:
        print(f"[ERROR] Failed to fetch GTFS-RT feed: {e}")
        return []

def get_vehicle_positions_with_calculated_delays():
    """
    Extract vehicle positions and calculate delays.
    """
    vehicle_entities = fetch_gtfs_rt_feed()
    vehicles_data = []
    current_timestamp = datetime.datetime.now(BUDAPEST)
    
    for entity in vehicle_entities:
        if entity.HasField("vehicle"):
            v = entity.vehicle
            
            # Extract required fields
            trip_id = v.trip.trip_id if v.HasField("trip") and v.trip.trip_id else None
            vehicle_id = v.vehicle.id if v.HasField("vehicle") else None
            last_stop_id = v.stop_id if v.HasField("stop_id") else None
            current_stop_sequence = v.current_stop_sequence if v.HasField("current_stop_sequence") else None
            
            # Calculate delay
            delay_seconds = delay_calculator.calculate_delay(
                trip_id, current_stop_sequence, v.timestamp
            )
            
            vehicle_data = {
                'timestamp': current_timestamp,
                'trip_id': trip_id,
                'vehicle_id': vehicle_id,
                'last_stop_id': last_stop_id,
                'delay_seconds': delay_seconds,
                'latitude': v.position.latitude if v.HasField("position") else None,
                'longitude': v.position.longitude if v.HasField("position") else None
            }
            
            # Only store vehicles with basic identification
            if vehicle_data['trip_id'] and vehicle_data['vehicle_id']:
                vehicles_data.append(vehicle_data)
    
    return vehicles_data

def save_vehicle_updates(vehicles_data):
    """
    Save vehicle data to SQLite database.
    """
    if not vehicles_data:
        print("[INFO] No vehicle data to save")
        return 0
    
    session = Session()
    count = 0
    delays_calculated = 0
    
    try:
        for vehicle in vehicles_data:
            update = VehicleUpdate(
                timestamp=vehicle['timestamp'],
                trip_id=vehicle['trip_id'],
                vehicle_id=vehicle['vehicle_id'],
                last_stop_id=vehicle['last_stop_id'],
                delay_seconds=vehicle['delay_seconds'],
                latitude=vehicle['latitude'],
                longitude=vehicle['longitude']
            )
            session.add(update)
            count += 1
            
            if vehicle['delay_seconds'] is not None:
                delays_calculated += 1
        
        session.commit()
        
        # Print statistics
        total_vehicles = len(vehicles_data)
        vehicles_with_position = len([v for v in vehicles_data if v['latitude'] and v['longitude']])
        
        print(f"[SUCCESS] Saved {count} vehicle updates at {datetime.datetime.now(BUDAPEST).isoformat()}")
        print(f"[STATS] Vehicles with calculated delays: {delays_calculated}/{total_vehicles} ({delays_calculated/total_vehicles*100:.1f}%)")
        print(f"[STATS] With position data: {vehicles_with_position}/{total_vehicles}")
        
        # Delay distribution
        if delays_calculated > 0:
            delays = [v['delay_seconds'] for v in vehicles_data if v['delay_seconds'] is not None]
            early_count = len([d for d in delays if d < -60])
            on_time_count = len([d for d in delays if -60 <= d <= 60])
            late_count = len([d for d in delays if d > 60])
            
            print(f"[DELAYS] Early (>1min): {early_count}, On time: {on_time_count}, Late (>1min): {late_count}")
            print(f"[DELAYS] Average delay: {sum(delays)/len(delays):.1f}s")
        
    except Exception as e:
        session.rollback()
        print(f"[ERROR] Database save failed: {e}")
    finally:
        session.close()
    
    return count

# =============================================================================
# MAIN SCRAPER LOOP
# =============================================================================

def run_scraper():
    """
    Main scraper loop - runs continuously with configured interval.
    """
    print("[INFO] Starting BKK Real-time Vehicle Data Scraper")
    print("[INFO] ===========================================")
    print(f"[INFO] Interval: {config.SCRAPER_INTERVAL_SEC} seconds")
    print(f"[INFO] Delay range: {config.MIN_REALISTIC_DELAY}s to {config.MAX_REALISTIC_DELAY}s")
    print(f"[INFO] Database: {config.DB_PATH}")
    print("[INFO] ===========================================")
    
    while True:
        try:
            cycle_start = time.time()
            
            vehicles_data = get_vehicle_positions_with_calculated_delays()
            save_vehicle_updates(vehicles_data)
            
            processing_time = time.time() - cycle_start
            sleep_time = max(1, config.SCRAPER_INTERVAL_SEC - processing_time)
            print(f"[INFO] Next collection in {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            print("\n[INFO] Scraper stopped by user - Graceful shutdown")
            break
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            print(f"[INFO] Retrying in {config.SCRAPER_INTERVAL_SEC} seconds...")
            time.sleep(config.SCRAPER_INTERVAL_SEC)

if __name__ == "__main__":
    run_scraper()