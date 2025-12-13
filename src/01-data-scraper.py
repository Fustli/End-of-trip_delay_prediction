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
import logging
from sqlalchemy.exc import SQLAlchemyError
from google.transit import gtfs_realtime_pb2

# Import configuration and database
import config
from models import Base, VehicleUpdate, get_engine, get_session, init_database

# =============================================================================
# INITIALIZATION
# =============================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Database setup
try:
    engine = get_engine()
    Session = get_session()  # This should be the Session CLASS
    logger.info("Database connection established")
    logger.info(f"DEBUG: Session type: {type(Session)}")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    raise

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
        self.final_stops = self._identify_final_stops()
        
    def _identify_final_stops(self):
        """Identify the final stop for each trip to detect endpoints."""
        logger.info("Identifying final stops for each trip...")
        final_stops = {}
        
        for trip_id, stops in self.schedule.items():
            if stops:  # Only if trip has stops
                max_sequence = max(stops.keys())
                final_stops[trip_id] = max_sequence
        
        logger.info(f"Identified final stops for {len(final_stops)} trips")
        return final_stops
        
    def _load_stop_times(self, stop_times_path):
        """
        Load GTFS stop_times.txt and build schedule lookup dictionary.
        """
        logger.info("Loading GTFS stop_times.txt...")
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
            
            logger.info(f"Loaded {len(schedule)} trips with {sum(len(stops) for stops in schedule.values())} stop sequences")
            return schedule
            
        except Exception as e:
            logger.error(f"Failed to load stop_times.txt: {e}")
            return {}
    
    def calculate_delay(self, trip_id, current_stop_sequence, vehicle_timestamp):
        """
        Calculate delay by comparing scheduled vs actual departure time.
        EXCLUDES delays at final stops (endpoints) to avoid counting waiting time.
        Returns tuple: (delay_seconds, is_endpoint, delay_calculated)
        """
        # Validate input parameters
        if not trip_id or current_stop_sequence is None:
            return None, False, False
            
        # Check if trip exists in schedule
        if trip_id not in self.schedule:
            return None, False, False
        
        # Check if stop sequence exists for this trip
        if current_stop_sequence not in self.schedule[trip_id]:
            return None, False, False
        
        # 🚨 CRITICAL FIX: Skip delay calculation at FINAL STOPS
        if trip_id in self.final_stops and current_stop_sequence == self.final_stops[trip_id]:
            if config.VERBOSE_FILTERING:
                logger.info(f"Endpoint detected - Trip: {trip_id}, Stop: {current_stop_sequence}")
            return None, True, False  # No delay, is endpoint, not calculated
        
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
        
        # Filter unrealistic delay values
        if time_diff < config.MIN_REALISTIC_DELAY or time_diff > config.MAX_REALISTIC_DELAY:
            if config.VERBOSE_FILTERING:
                logger.info(f"Unrealistic delay filtered: {time_diff}s - Trip: {trip_id}")
            return None, False, False  # No delay, not endpoint, not calculated
    
        return time_diff, False, True  # Valid delay, not endpoint, calculated

# Initialize global delay calculator
try:
    delay_calculator = DelayCalculator(config.STOP_TIMES_FILE)
    logger.info("Delay calculator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize delay calculator: {e}")
    raise

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
        logger.debug(f"Fetched {len(feed.entity)} entities from GTFS-RT feed")
        return feed.entity
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching GTFS-RT feed: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching GTFS-RT feed: {e}")
        return []

def get_vehicle_positions_with_calculated_delays():
    """
    Extract vehicle positions and calculate delays with quality flags.
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
            
            # Calculate delay with quality information
            delay_seconds, is_endpoint, delay_calculated = delay_calculator.calculate_delay(
                trip_id, current_stop_sequence, v.timestamp
            )
            
            # 🚨 FIX: Data quality flags - CORRECT VERSION
            has_position = v.HasField("position") and v.position.latitude is not None and v.position.longitude is not None
            has_stop_info = bool(last_stop_id)
            
            vehicle_data = {
                'timestamp': current_timestamp,
                'trip_id': trip_id,
                'vehicle_id': vehicle_id,
                'last_stop_id': last_stop_id,
                'delay_seconds': delay_seconds,
                'latitude': v.position.latitude if v.HasField("position") else None,
                'longitude': v.position.longitude if v.HasField("position") else None,
                # 🚨 FIXED: Data quality flags
                'has_position': has_position,  # Should be boolean, not longitude!
                'has_stop_info': has_stop_info,
                'is_endpoint': is_endpoint,
                'delay_calculated': delay_calculated
            }
            
            # Only store vehicles with basic identification
            if vehicle_data['trip_id'] and vehicle_data['vehicle_id']:
                vehicles_data.append(vehicle_data)
    
    logger.info(f"Processed {len(vehicles_data)} valid vehicle positions")
    return vehicles_data

def save_vehicle_updates(vehicles_data):
    """
    Save vehicle data to SQLite database with quality flags.
    """
    if not vehicles_data:
        logger.info("No vehicle data to save")
        return 0
    
    session = Session()
    count = 0
    delays_calculated = 0
    endpoints_detected = 0
    
    try:
        for vehicle in vehicles_data:
            update = VehicleUpdate(
                timestamp=vehicle['timestamp'],
                trip_id=vehicle['trip_id'],
                vehicle_id=vehicle['vehicle_id'],
                last_stop_id=vehicle['last_stop_id'],
                delay_seconds=vehicle['delay_seconds'],
                latitude=vehicle['latitude'],
                longitude=vehicle['longitude'],
                # Data quality flags
                has_position=vehicle['has_position'],
                has_stop_info=vehicle['has_stop_info'],
                is_endpoint=vehicle['is_endpoint'],
                delay_calculated=vehicle['delay_calculated']
            )
            session.add(update)
            count += 1
            
            if vehicle['delay_calculated']:
                delays_calculated += 1
                
            if vehicle['is_endpoint']:
                endpoints_detected += 1
        
        session.commit()
        
        # Enhanced statistics logging
        total_vehicles = len(vehicles_data)
        vehicles_with_position = len([v for v in vehicles_data if v['has_position']])
        vehicles_with_stop_info = len([v for v in vehicles_data if v['has_stop_info']])
        
        logger.info(f"Saved {count} vehicle updates")
        logger.info(f"Delay calculation: {delays_calculated}/{total_vehicles} ({delays_calculated/total_vehicles*100:.1f}%)")
        logger.info(f"Endpoints detected: {endpoints_detected}")
        logger.info(f"Position data: {vehicles_with_position}/{total_vehicles}")
        logger.info(f"Stop info: {vehicles_with_stop_info}/{total_vehicles}")
        
        # Delay distribution (only for calculated delays)
        if delays_calculated > 0:
            delays = [v['delay_seconds'] for v in vehicles_data if v['delay_calculated']]
            early_count = len([d for d in delays if d < -60])
            on_time_count = len([d for d in delays if -60 <= d <= 60])
            late_count = len([d for d in delays if d > 60])
            
            logger.info(f"⏰ Delays - Early: {early_count}, On-time: {on_time_count}, Late: {late_count}")
            logger.info(f"⏰ Average delay: {sum(delays)/len(delays):.1f}s")
        
        return count
        
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error during save: {e}")
        return 0
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error during save: {e}")
        return 0
    finally:
        session.close()

# =============================================================================
# MAIN SCRAPER LOOP
# =============================================================================

def run_scraper():
    """
    Main scraper loop - runs continuously with configured interval.
    """
    logger.info("🚌 Starting BKK Real-time Vehicle Data Scraper")
    logger.info("=" * 50)
    logger.info(f"📅 Interval: {config.SCRAPER_INTERVAL_SEC} seconds")
    logger.info(f"🎯 Delay range: {config.MIN_REALISTIC_DELAY}s to {config.MAX_REALISTIC_DELAY}s")
    logger.info(f"💾 Database: {config.DB_PATH}")
    logger.info("=" * 50)
    
    # Initialize database on startup
    try:
        init_database()
        logger.info("Database initialization completed")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return
    
    while True:
        try:
            cycle_start = time.time()
            
            vehicles_data = get_vehicle_positions_with_calculated_delays()
            records_saved = save_vehicle_updates(vehicles_data)
            
            processing_time = time.time() - cycle_start
            sleep_time = max(1, config.SCRAPER_INTERVAL_SEC - processing_time)
            
            if records_saved > 0:
                logger.info(f"⏰ Next collection in {sleep_time:.1f} seconds...")
            else:
                logger.warning(f"No records saved, retrying in {sleep_time:.1f} seconds...")
                
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logger.info("🛑 Scraper stopped by user - Graceful shutdown")
            break
        except Exception as e:
            logger.error(f"💥 Unexpected error in main loop: {e}")
            logger.info(f"🔄 Retrying in {config.SCRAPER_INTERVAL_SEC} seconds...")
            time.sleep(config.SCRAPER_INTERVAL_SEC)

if __name__ == "__main__":
    run_scraper()