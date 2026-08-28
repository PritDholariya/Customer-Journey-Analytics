import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROCESSED_DATA_DIR / "figures"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Data files
RAW_EVENTS_FILE = RAW_DATA_DIR / "events.csv"
CANONICAL_EVENTS_FILE = PROCESSED_DATA_DIR / "events_canonical.parquet"
CLEAN_EVENTS_FILE = PROCESSED_DATA_DIR / "events_clean.parquet"

# Sampling and Filtering configurations
RANDOM_SEED = 42
MIN_EVENTS_PER_USER = 2  # Used in Phase 2
N_SAMPLE_VISITORS = 50000  # Number of visitors to sample (reduces data size for ML)

# Sessionization
SESSION_INACTIVITY_GAP_MINUTES = 30  # Standard session gap
