import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the .env file in the project root.
# load_dotenv() reads each KEY=VALUE line from .env and adds it to os.environ
# BEFORE any os.environ.get() calls below.
#
# override=False (the default) means: if a variable is already set in the
# real environment (e.g., in Docker or CI), the .env file does NOT overwrite it.
# This is the correct production behaviour — Docker sets the real credentials,
# .env is only for local development.
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=False)

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
SEGMENT_FUNNELS_FILE = PROCESSED_DATA_DIR / "segment_funnels.parquet"

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

# ── Phase 7: Database ─────────────────────────────────────────────────────────
# Database credentials are NEVER hardcoded here.
# They are read from environment variables at runtime.
# This is a fundamental security rule — credentials must not appear in code
# that could be committed to a public git repository.
#
# For local development, set these in a .env file (which is git-ignored).
# For production (Docker), set them in docker-compose.yml as env vars.
#
# Defaults below are only for local Docker development convenience.
DB_HOST     = os.environ.get("DB_HOST",     "localhost")
DB_PORT     = os.environ.get("DB_PORT",     "5432")
DB_NAME     = os.environ.get("DB_NAME",     "cja_db")
DB_USER     = os.environ.get("DB_USER",     "cja_user")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "cja_password")

# SQLAlchemy connection URL constructed from the env vars above
DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# For unit tests (no Postgres needed) — SQLite in-memory database
TEST_DATABASE_URL = "sqlite://"
