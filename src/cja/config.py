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
USER_FEATURES_FILE = PROCESSED_DATA_DIR / "user_features.parquet"
USER_SEGMENTS_FILE = PROCESSED_DATA_DIR / "user_segments.parquet"

# Sampling and Filtering configurations
RANDOM_SEED = 42
MIN_EVENTS_PER_USER = 2       # Used in Phase 2
N_SAMPLE_VISITORS = 50000     # Number of visitors to sample (reduces data size for ML)

# Sessionization
SESSION_INACTIVITY_GAP_MINUTES = 30  # Standard session gap

# ── Phase 4: Clustering ────────────────────────────────────────────────────────

# These are the exact feature columns fed into the clustering algorithm.
# We ONLY use the log-transformed versions of count features (they are less skewed)
# Plus ratio and time features that are already on a reasonable scale.
CLUSTERING_FEATURES = [
    'total_events_log',
    'n_views_log',
    'n_addtocart_log',
    'n_transactions_log',
    'n_distinct_items_log',
    'n_sessions_log',
    'avg_events_per_session',
    'avg_session_duration_sec',
    'cart_to_view_ratio',
    'purchase_to_cart_ratio',
    'max_funnel_stage',
    'recency_days',
    'active_span_days',
]

# K-Means: range of k values to try in the elbow method
KMEANS_K_MIN = 2
KMEANS_K_MAX = 10

# Final chosen K after reviewing the elbow + silhouette plots
# Elbow bends at K=5 AND silhouette has a local peak at K=5 → K=5 is the best choice
KMEANS_K_FINAL = 5

# DBSCAN tuning knobs
DBSCAN_EPS = 1.5        # The radius of each point's neighborhood
DBSCAN_MIN_SAMPLES = 10 # Minimum points to form a dense core region
