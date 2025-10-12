"""
Configuration settings for BKK Real-time Vehicle Data Scraper
All sensitive and environment-specific settings go here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API CONFIGURATION
# =============================================================================
BKK_API_KEY = os.getenv("BKK_API_KEY")
BKK_BASE_URL = os.getenv("BKK_BASE_URL")
VEHICLE_URL = f"{BKK_BASE_URL}/gtfs-rt/full/VehiclePositions.pb?key={BKK_API_KEY}"

# =============================================================================
# FILE PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
GTFS_DIR = os.path.join(DATA_DIR, "gtfs")

STOP_TIMES_FILE = os.path.join(GTFS_DIR, "stop_times.txt")
DB_PATH = os.path.join(DATA_DIR, "transit.db")

# =============================================================================
# SCRAPER SETTINGS
# =============================================================================
SCRAPER_INTERVAL_SEC = int(os.getenv("SCRAPER_INTERVAL_SEC", "60"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))

# =============================================================================
# DELAY CALCULATION SETTINGS
# =============================================================================
MIN_REALISTIC_DELAY = int(os.getenv("MIN_REALISTIC_DELAY", "-1800"))  # -30 minutes
MAX_REALISTIC_DELAY = int(os.getenv("MAX_REALISTIC_DELAY", "5400"))   # +90 minutes

# =============================================================================
# DATABASE SETTINGS
# =============================================================================
DATABASE_URL = f"sqlite:///{DB_PATH}"

# =============================================================================
# TIMEZONE
# =============================================================================
TIMEZONE = "Europe/Budapest"

# Optional: Set to False to reduce console noise
VERBOSE_FILTERING = False