"""
Configuration settings for BKK End-of-Trip Delay Prediction Project.
Combined configuration for Data Scraper, Preprocessing, Baseline Models, and GNN Training.
"""

import os
import torch
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# 1. PATH CONFIGURATION (Project Structure)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Move up one level if src/ is the current dir, to reach root for data/, log/, and plots/
PROJECT_ROOT = os.path.dirname(BASE_DIR) 

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOG_DIR = os.path.join(PROJECT_ROOT, "log")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")  # <-- NEW: For saving diagrams

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Specific file paths
DB_PATH = os.path.join(DATA_DIR, "transit.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"
RAW_CSV_PATH = os.path.join(DATA_DIR, "vehicle_positions.csv")
CLEANED_CSV_PATH = os.path.join(DATA_DIR, "vehicle_positions_cleaned.csv") # <-- NEW: Explicit path for cleaned data
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_gnn_data.pt")
LOG_FILE_PATH = os.path.join(LOG_DIR, "run.log")

GTFS_DIR = os.path.join(DATA_DIR, "gtfs")
STOP_TIMES_FILE = os.path.join(GTFS_DIR, "stop_times.txt")

# =============================================================================
# 2. GLOBAL SETTINGS
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
RANDOM_STATE = SEED  # Alias for Scikit-Learn compatibility
TIMEZONE = "Europe/Budapest"

# Dataloader defaults (used by notebooks / training scripts)
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
PIN_MEMORY = os.getenv("PIN_MEMORY", "True").lower() == "true"

# =============================================================================
# 3. BASELINE MODEL CONFIGURATION (Linear Reg & Random Forest)
# =============================================================================
# Columns to use for the basic regression task
BASELINE_FEATURES = [
    'latitude', 
    'longitude', 
    'hour', 
    'minute', 
    'day_of_week'
]
TARGET_COL = "delay_seconds"

# Splitting
TEST_SIZE = 0.2

# Linear Regression Settings
LIN_REG_FIT_INTERCEPT = True

# Random Forest Hyperparameters
RF_N_ESTIMATORS = 50       # Keep low for rapid prototyping (Source: Baseline Notebook)
RF_N_JOBS = -1             # Use all CPU cores

# Ablation Study Depths (Step-by-step complexity increase)
RF_MAX_DEPTH = 10          # For Basic RF (Lat/Lon only)
RF_SMART_MAX_DEPTH = 12    # For Enhanced RF (Stop History)
RF_FINAL_MAX_DEPTH = 12    # For Context-Aware RF (Lag Features)


# =============================================================================
# 4. GNN MODEL HYPERPARAMETERS (Incremental Modeling)
# =============================================================================
# Architecture dimensions
# Start small as per "Incremental Modeling"
INPUT_DIM = 8        # Number of features per node/edge - ADJUST based on final preprocessing
HIDDEN_DIM = 16      # Small hidden dimension for the baseline "Baby GNN"
OUTPUT_DIM = 1       # Regression output (delay in seconds)

# Training Hyperparameters
BATCH_SIZE = 32      
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4  # L2 Regularization

# Epochs
NUM_EPOCHS = 10     

# Validation split used by notebook experiments (fraction of total rows)
VAL_SIZE = float(os.getenv("VAL_SIZE", "0.1"))

# Regularization
DROPOUT_RATE = 0.0   # Increase later if overfitting occurs

# Early Stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10

# "Overfit on Single Batch" Flag
OVERFIT_SINGLE_BATCH = False 


# =============================================================================
# 4B. SOTA EXPERIMENT SETTINGS (Notebook-driven)
# =============================================================================
# These are intentionally separate from the incremental baseline settings above.

# Leaderboard target (Context-Aware Random Forest)
CHAMPION_MAE = float(os.getenv("CHAMPION_MAE", "43.18"))

# SOTA V1 (GCN baseline notebook)
SOTA_V1_BATCH_SIZE = int(os.getenv("SOTA_V1_BATCH_SIZE", "16384"))
SOTA_V1_LR = float(os.getenv("SOTA_V1_LR", "0.005"))
SOTA_V1_NUM_EPOCHS = int(os.getenv("SOTA_V1_NUM_EPOCHS", "20"))


# =============================================================================
# 4C. GNN MODEL V1 SETTINGS (Preferred Names; Backwards Compatible)
# =============================================================================
# The notebook and logs refer to "GNN Model V1". Keep existing env var names
# (SOTA_V1_*) for compatibility, but expose alias constants without "SOTA".

GNN_V1_BATCH_SIZE = SOTA_V1_BATCH_SIZE
GNN_V1_LR = SOTA_V1_LR
GNN_V1_NUM_EPOCHS = SOTA_V1_NUM_EPOCHS

# SOTA V2 (ContextAwareGAT_V2 notebook)
SOTA_V2_BATCH_SIZE = int(os.getenv("SOTA_V2_BATCH_SIZE", "16384"))
SOTA_V2_EVAL_BATCH_SIZE = int(os.getenv("SOTA_V2_EVAL_BATCH_SIZE", str(SOTA_V2_BATCH_SIZE)))
SOTA_V2_LR = float(os.getenv("SOTA_V2_LR", "0.003"))
SOTA_V2_WEIGHT_DECAY = float(os.getenv("SOTA_V2_WEIGHT_DECAY", "0.0"))
SOTA_V2_NUM_EPOCHS = int(os.getenv("SOTA_V2_NUM_EPOCHS", "50"))

# ReduceLROnPlateau defaults (SOTA V2)
SOTA_V2_SCHED_FACTOR = float(os.getenv("SOTA_V2_SCHED_FACTOR", "0.5"))
SOTA_V2_SCHED_PATIENCE = int(os.getenv("SOTA_V2_SCHED_PATIENCE", "3"))
SOTA_V2_SCHED_THRESHOLD = float(os.getenv("SOTA_V2_SCHED_THRESHOLD", "1e-4"))
SOTA_V2_SCHED_MIN_LR = float(os.getenv("SOTA_V2_SCHED_MIN_LR", "1e-6"))
SOTA_V2_SCHED_COOLDOWN = int(os.getenv("SOTA_V2_SCHED_COOLDOWN", "0"))


# =============================================================================
# 4D. GNN MODEL V2 SETTINGS (Preferred Names; Backwards Compatible)
# =============================================================================
# The notebook and logs refer to "GNN Model V2". Keep existing env var names
# (SOTA_V2_*) for compatibility, but expose alias constants without "SOTA".

GNN_V2_BATCH_SIZE = SOTA_V2_BATCH_SIZE
GNN_V2_EVAL_BATCH_SIZE = SOTA_V2_EVAL_BATCH_SIZE
GNN_V2_LR = SOTA_V2_LR
GNN_V2_WEIGHT_DECAY = SOTA_V2_WEIGHT_DECAY
GNN_V2_NUM_EPOCHS = SOTA_V2_NUM_EPOCHS

GNN_V2_SCHED_FACTOR = SOTA_V2_SCHED_FACTOR
GNN_V2_SCHED_PATIENCE = SOTA_V2_SCHED_PATIENCE
GNN_V2_SCHED_THRESHOLD = SOTA_V2_SCHED_THRESHOLD
GNN_V2_SCHED_MIN_LR = SOTA_V2_SCHED_MIN_LR
GNN_V2_SCHED_COOLDOWN = SOTA_V2_SCHED_COOLDOWN


# =============================================================================
# 4E. GNN MODEL V3 SETTINGS (Temporal + Spatial)
# =============================================================================
# Defaults are conservative and derived from V2; override via environment vars.

GNN_V3_SEQ_LEN = int(os.getenv("GNN_V3_SEQ_LEN", "12"))
GNN_V3_GRU_HIDDEN = int(os.getenv("GNN_V3_GRU_HIDDEN", "64"))
GNN_V3_DROPOUT = float(os.getenv("GNN_V3_DROPOUT", "0.2"))

GNN_V3_BATCH_SIZE = int(os.getenv("GNN_V3_BATCH_SIZE", "4096"))
GNN_V3_EVAL_BATCH_SIZE = int(os.getenv("GNN_V3_EVAL_BATCH_SIZE", str(GNN_V3_BATCH_SIZE)))
GNN_V3_LR = float(os.getenv("GNN_V3_LR", str(GNN_V2_LR)))
GNN_V3_WEIGHT_DECAY = float(os.getenv("GNN_V3_WEIGHT_DECAY", str(GNN_V2_WEIGHT_DECAY)))
GNN_V3_NUM_EPOCHS = int(os.getenv("GNN_V3_NUM_EPOCHS", "50"))

GNN_V3_SCHED_FACTOR = float(os.getenv("GNN_V3_SCHED_FACTOR", str(GNN_V2_SCHED_FACTOR)))
GNN_V3_SCHED_PATIENCE = int(os.getenv("GNN_V3_SCHED_PATIENCE", str(GNN_V2_SCHED_PATIENCE)))
GNN_V3_SCHED_THRESHOLD = float(os.getenv("GNN_V3_SCHED_THRESHOLD", str(GNN_V2_SCHED_THRESHOLD)))
GNN_V3_SCHED_MIN_LR = float(os.getenv("GNN_V3_SCHED_MIN_LR", str(GNN_V2_SCHED_MIN_LR)))
GNN_V3_SCHED_COOLDOWN = int(os.getenv("GNN_V3_SCHED_COOLDOWN", str(GNN_V2_SCHED_COOLDOWN)))

# Feature engineering knobs for V3
GNN_V3_LAG_CLIP_MIN = int(os.getenv("GNN_V3_LAG_CLIP_MIN", "-1800"))
GNN_V3_LAG_CLIP_MAX = int(os.getenv("GNN_V3_LAG_CLIP_MAX", "3600"))
GNN_V3_TIME_DELTA_CLIP_SEC = int(os.getenv("GNN_V3_TIME_DELTA_CLIP_SEC", "900"))
GNN_V3_ROLLING_LAG_WINDOW = int(os.getenv("GNN_V3_ROLLING_LAG_WINDOW", "3"))

# Data splitting: prevent leakage by splitting on trip_id by default
GNN_V3_SPLIT_BY_TRIP = os.getenv("GNN_V3_SPLIT_BY_TRIP", "True").lower() == "true"


# =============================================================================
# 4F. GNN MODEL V4 SETTINGS (V3 + GATv2Conv)
# =============================================================================
# V4 is intentionally a minimal delta from V3: same temporal/windowing pipeline,
# but swapping the spatial layer from GATConv -> GATv2Conv.
# Defaults mirror V3 unless overridden via GNN_V4_* environment vars.

GNN_V4_SEQ_LEN = int(os.getenv("GNN_V4_SEQ_LEN", str(GNN_V3_SEQ_LEN)))
GNN_V4_GRU_HIDDEN = int(os.getenv("GNN_V4_GRU_HIDDEN", str(GNN_V3_GRU_HIDDEN)))
GNN_V4_DROPOUT = float(os.getenv("GNN_V4_DROPOUT", str(GNN_V3_DROPOUT)))

GNN_V4_BATCH_SIZE = int(os.getenv("GNN_V4_BATCH_SIZE", str(GNN_V3_BATCH_SIZE)))
GNN_V4_EVAL_BATCH_SIZE = int(os.getenv("GNN_V4_EVAL_BATCH_SIZE", str(GNN_V4_BATCH_SIZE)))
GNN_V4_LR = float(os.getenv("GNN_V4_LR", str(GNN_V3_LR)))
GNN_V4_WEIGHT_DECAY = float(os.getenv("GNN_V4_WEIGHT_DECAY", str(GNN_V3_WEIGHT_DECAY)))
GNN_V4_NUM_EPOCHS = int(os.getenv("GNN_V4_NUM_EPOCHS", str(GNN_V3_NUM_EPOCHS)))

GNN_V4_SCHED_FACTOR = float(os.getenv("GNN_V4_SCHED_FACTOR", str(GNN_V3_SCHED_FACTOR)))
GNN_V4_SCHED_PATIENCE = int(os.getenv("GNN_V4_SCHED_PATIENCE", str(GNN_V3_SCHED_PATIENCE)))
GNN_V4_SCHED_THRESHOLD = float(os.getenv("GNN_V4_SCHED_THRESHOLD", str(GNN_V3_SCHED_THRESHOLD)))
GNN_V4_SCHED_MIN_LR = float(os.getenv("GNN_V4_SCHED_MIN_LR", str(GNN_V3_SCHED_MIN_LR)))
GNN_V4_SCHED_COOLDOWN = int(os.getenv("GNN_V4_SCHED_COOLDOWN", str(GNN_V3_SCHED_COOLDOWN)))

GNN_V4_LAG_CLIP_MIN = int(os.getenv("GNN_V4_LAG_CLIP_MIN", str(GNN_V3_LAG_CLIP_MIN)))
GNN_V4_LAG_CLIP_MAX = int(os.getenv("GNN_V4_LAG_CLIP_MAX", str(GNN_V3_LAG_CLIP_MAX)))
GNN_V4_TIME_DELTA_CLIP_SEC = int(os.getenv("GNN_V4_TIME_DELTA_CLIP_SEC", str(GNN_V3_TIME_DELTA_CLIP_SEC)))
GNN_V4_ROLLING_LAG_WINDOW = int(os.getenv("GNN_V4_ROLLING_LAG_WINDOW", str(GNN_V3_ROLLING_LAG_WINDOW)))

GNN_V4_SPLIT_BY_TRIP = os.getenv("GNN_V4_SPLIT_BY_TRIP", str(GNN_V3_SPLIT_BY_TRIP)).lower() == "true"


# =============================================================================
# 5. SCRAPER & API CONFIGURATION (Legacy/Data Gen)
# =============================================================================
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
BKK_API_KEY = os.getenv("BKK_API_KEY")
BKK_BASE_URL = os.getenv("BKK_BASE_URL")
VEHICLE_URL = f"{BKK_BASE_URL}/gtfs-rt/full/VehiclePositions.pb?key={BKK_API_KEY}"

SCRAPER_INTERVAL_SEC = int(os.getenv("SCRAPER_INTERVAL_SEC", "60"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))

MIN_REALISTIC_DELAY = int(os.getenv("MIN_REALISTIC_DELAY", "-1800"))
MAX_REALISTIC_DELAY = int(os.getenv("MAX_REALISTIC_DELAY", "1800"))
VERBOSE_FILTERING = os.getenv("VERBOSE_FILTERING", "True").lower() == "true"

LAT_MIN, LAT_MAX = 47.30, 47.65
LON_MIN, LON_MAX = 18.90, 19.35