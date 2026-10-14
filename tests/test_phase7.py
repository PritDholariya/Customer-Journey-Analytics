"""
test_phase7.py — Unit Tests for Phase 7: Database Layer

IMPORTANT: These tests use an SQLite in-memory database ("sqlite://")
instead of PostgreSQL. This means:
  - No Docker or Postgres needed to run the tests
  - Tests run in milliseconds (no network)
  - Tests are fully isolated (each test gets a fresh database)

WHY SQLite FOR TESTING?
    SQLAlchemy is designed to be database-agnostic. The same Python code that
    creates tables and inserts rows works on both SQLite and PostgreSQL because
    SQLAlchemy generates the correct SQL dialect for each database engine.
    SQLite lives entirely in memory (no file, no server), making it perfect
    for fast, isolated unit tests.

    The production code will run against PostgreSQL. Tests validate the Python
    logic — schema creation, data types, row counts, correctness of queries —
    not Postgres-specific features.
"""

import pytest
import pandas as pd
from sqlalchemy import text

from cja.config import (
    USER_FEATURES_FILE,
    USER_SEGMENTS_FILE,
    SEGMENT_FUNNELS_FILE,
    TEST_DATABASE_URL,
)
from cja.database import (
    get_engine,
    get_metadata_and_tables,
    load_all_data,
    build_segment_profiles,
    verify_load,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────
# pytest fixtures are reusable setup functions. A test function declares a
# fixture as a parameter and pytest automatically calls the fixture and
# injects its return value into the test function.

@pytest.fixture
def in_memory_engine():
    """
    Creates a fresh SQLite in-memory engine for each test.
    The 'sqlite://' URL (two slashes, no path) creates the database entirely
    in RAM — it vanishes when the engine is garbage collected.
    """
    engine = get_engine(TEST_DATABASE_URL)
    metadata = get_metadata_and_tables(engine)
    metadata.create_all(engine)   # CREATE TABLE statements
    return engine


@pytest.fixture
def loaded_engine(in_memory_engine):
    """
    Builds on in_memory_engine by also loading the real parquet data.
    Tests that need to query actual data should use this fixture.
    """
    if not USER_FEATURES_FILE.exists():
        pytest.skip("Parquet files not found. Run Phases 1–5 first.")
    load_all_data(in_memory_engine)
    return in_memory_engine


# ── Schema Tests ────────────────────────────────────────────────────────────────

def test_schema_creates_all_tables(in_memory_engine):
    """
    Verifies that all 4 expected tables are created in the database.
    SQLite's sqlite_master table holds metadata about all database objects.
    """
    with in_memory_engine.connect() as conn:
        result = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
        tables = {row[0] for row in result}   # set of table names

    expected_tables = {"user_features", "user_segments", "segment_profiles", "segment_funnels"}
    assert expected_tables.issubset(tables), \
        f"Missing tables: {expected_tables - tables}"


def test_schema_idempotent(in_memory_engine):
    """
    Calling get_metadata_and_tables() multiple times must not crash.
    extend_existing=True in the Table definitions ensures this.
    """
    try:
        metadata2 = get_metadata_and_tables(in_memory_engine)
        metadata2.create_all(in_memory_engine)
    except Exception as e:
        pytest.fail(f"Second call to get_metadata_and_tables raised: {e}")


# ── Data Loading Tests ──────────────────────────────────────────────────────────

def test_row_counts_match_parquet(loaded_engine):
    """
    After loading, the row count in each database table must match the
    row count in its source parquet file. This is the fundamental data
    integrity check for any ETL pipeline.
    """
    features_parquet_rows = len(pd.read_parquet(USER_FEATURES_FILE))
    segments_parquet_rows = len(pd.read_parquet(USER_SEGMENTS_FILE))
    funnels_parquet_rows  = len(pd.read_parquet(SEGMENT_FUNNELS_FILE))

    with loaded_engine.connect() as conn:
        feat_rows = conn.execute(text("SELECT COUNT(*) FROM user_features")).scalar()
        seg_rows  = conn.execute(text("SELECT COUNT(*) FROM user_segments")).scalar()
        fun_rows  = conn.execute(text("SELECT COUNT(*) FROM segment_funnels")).scalar()

    assert feat_rows == features_parquet_rows, \
        f"user_features: DB={feat_rows} vs parquet={features_parquet_rows}"
    assert seg_rows == segments_parquet_rows, \
        f"user_segments: DB={seg_rows} vs parquet={segments_parquet_rows}"
    assert fun_rows == funnels_parquet_rows, \
        f"segment_funnels: DB={fun_rows} vs parquet={funnels_parquet_rows}"


def test_segment_profiles_row_count(loaded_engine):
    """
    segment_profiles should have exactly one row per unique K-Means cluster.
    """
    segments_df = pd.read_parquet(USER_SEGMENTS_FILE)
    expected_profiles = segments_df['kmeans_cluster'].nunique()

    with loaded_engine.connect() as conn:
        actual = conn.execute(text("SELECT COUNT(*) FROM segment_profiles")).scalar()

    assert actual == expected_profiles, \
        f"Expected {expected_profiles} segment profiles, got {actual}"


def test_no_null_user_ids(loaded_engine):
    """
    user_id is the primary key — there must be zero NULL user_ids
    in user_features and user_segments.
    """
    with loaded_engine.connect() as conn:
        null_feat = conn.execute(
            text("SELECT COUNT(*) FROM user_features WHERE user_id IS NULL")
        ).scalar()
        null_seg = conn.execute(
            text("SELECT COUNT(*) FROM user_segments WHERE user_id IS NULL")
        ).scalar()

    assert null_feat == 0, "Found NULL user_ids in user_features"
    assert null_seg  == 0, "Found NULL user_ids in user_segments"


def test_load_is_idempotent(loaded_engine):
    """
    Running load_all_data() twice must NOT double the row count.
    The if_exists='replace' setting ensures tables are truncated before reload.
    """
    load_all_data(loaded_engine, if_exists="replace")

    expected_rows = len(pd.read_parquet(USER_FEATURES_FILE))
    with loaded_engine.connect() as conn:
        actual = conn.execute(text("SELECT COUNT(*) FROM user_features")).scalar()

    assert actual == expected_rows, \
        f"After second load: expected {expected_rows}, got {actual} (possible double-insert)"


# ── Business Logic Tests ────────────────────────────────────────────────────────

def test_converters_segment_exists_in_db(loaded_engine):
    """
    The 'Converters' segment — our highest-value customer group —
    must be present in the segment_funnels table.
    """
    with loaded_engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM segment_funnels WHERE segment_name = 'Converters'")
        ).scalar()

    assert result == 1, "The 'Converters' segment is missing from segment_funnels table."


def test_funnels_pct_in_valid_range(loaded_engine):
    """
    All percentage values in segment_funnels must be between 0 and 100.
    If any value is outside this range, there is a calculation bug in Phase 5.
    """
    with loaded_engine.connect() as conn:
        bad_rows = conn.execute(text("""
            SELECT COUNT(*) FROM segment_funnels
            WHERE pct_reached_view < 0 OR pct_reached_view > 100
               OR pct_reached_cart  < 0 OR pct_reached_cart  > 100
               OR pct_converted     < 0 OR pct_converted     > 100
        """)).scalar()

    assert bad_rows == 0, f"Found {bad_rows} rows with out-of-range percentages."
