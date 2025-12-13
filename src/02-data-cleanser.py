import pandas as pd
import os
import sys
import logging

# --- IMPORT CONFIGURATION ---
try:
    import config
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config

# --- IMPORT LOGGER SETUP ---
try:
    from utils import setup_logger
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import setup_logger


logger = setup_logger(
    name="data_cleanser",
    log_dir=config.LOG_DIR,
    filename=os.path.basename(getattr(config, "LOG_FILE_PATH", "run.log")),
    level=logging.INFO,
)

def analyze_and_clean_data():
    input_path = getattr(config, "RAW_CSV_PATH", getattr(config, "FILE_PATH", None))
    output_path = getattr(config, "CLEANED_CSV_PATH", None)

    if not input_path:
        logger.error("No input path configured. Expected config.RAW_CSV_PATH (preferred) or config.FILE_PATH.")
        return

    logger.info(f"Target file: {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"File not found at: {input_path}")
        return

    logger.info("Loading dataset...")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    logger.info(f"Loaded {initial_count} rows.")

    # --- 1. DEFINE OUTLIER MASKS ---

    # A. Missing Data (NaN) - CRITICAL FIX
    mask_nan = df['delay_seconds'].isna()

    # B. Exact Zero Delays (excluding NaNs to avoid double counting)
    mask_zero = (df['delay_seconds'] == 0) & (~mask_nan)

    # C. Unrealistic Delays (excluding NaNs)
    # comparisons with NaN always return False, so we don't strictly need ~mask_nan here,
    # but it's good practice for clarity.
    mask_delay_outliers = (df['delay_seconds'] < config.MIN_REALISTIC_DELAY) | \
                          (df['delay_seconds'] > config.MAX_REALISTIC_DELAY)

    # D. Geolocation Outliers (config-driven)
    LAT_MIN = getattr(config, "LAT_MIN", 47.30)
    LAT_MAX = getattr(config, "LAT_MAX", 47.65)
    LON_MIN = getattr(config, "LON_MIN", 18.90)
    LON_MAX = getattr(config, "LON_MAX", 19.35)

    mask_geo_outliers = ~(
        (df['latitude'] >= LAT_MIN) & (df['latitude'] <= LAT_MAX) &
        (df['longitude'] >= LON_MIN) & (df['longitude'] <= LON_MAX)
    )

    # --- 2. REPORTING ---

    total_useless_mask = mask_nan | mask_zero | mask_delay_outliers | mask_geo_outliers
    useless_data = df[total_useless_mask]
    clean_data = df[~total_useless_mask]

    logger.info("=" * 40)
    logger.info("OUTLIER ANALYSIS REPORT")
    logger.info("=" * 40)
    logger.info(f"1. Missing Delays (NaN):     {mask_nan.sum():>8} rows")
    logger.info(f"2. Exact Zero Delays:        {mask_zero.sum():>8} rows")
    logger.info(f"3. Unrealistic Delays:       {mask_delay_outliers.sum():>8} rows")
    logger.info(f"   ( < {config.MIN_REALISTIC_DELAY}s or > {config.MAX_REALISTIC_DELAY}s )")
    logger.info(f"4. Outside Budapest bounds:  {mask_geo_outliers.sum():>8} rows")
    logger.info("-" * 40)
    logger.info(
        f"TOTAL USELESS ROWS:          {len(useless_data):>8} ({(len(useless_data)/initial_count)*100:.1f}%)"
    )
    logger.info(
        f"CLEAN DATA REMAINING:        {len(clean_data):>8} ({(len(clean_data)/initial_count)*100:.1f}%)"
    )
    logger.info("=" * 40)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        clean_data.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned dataset to: {output_path}")
    else:
        logger.warning("No output path configured (config.CLEANED_CSV_PATH). Cleaned data was not saved.")

    # --- 3. PREVIEW ---
    if not useless_data.empty:
        logger.info("Preview of useless data (first 5 rows):")
        logger.info(useless_data[['delay_seconds', 'latitude', 'longitude']].head().to_string())

if __name__ == "__main__":
    analyze_and_clean_data()