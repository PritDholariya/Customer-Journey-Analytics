"""
database.py — Phase 7: PostgreSQL Storage

PURPOSE:
    Defines the relational database schema using SQLAlchemy, then loads the
    processed parquet outputs from Phases 3, 4, and 5 into PostgreSQL tables.

WHY A DATABASE AT ALL?
    The API in Phase 8 needs to serve data to multiple clients simultaneously.
    A parquet file can only be read by one Python process at a time — it has no
    concept of concurrent queries, indexes, or filtering by ID.
    A relational database solves all three: it handles many simultaneous requests,
    lets you add indexes for fast lookups, and supports SQL filtering.

WHY SQLALCHEMY OVER RAW SQL?
    Raw SQL strings are:
      1. Vulnerable to SQL injection if user input is ever concatenated into them
      2. Not portable — PostgreSQL syntax differs from SQLite and MySQL
      3. Hard to refactor — column renames require hunting string literals

    SQLAlchemy uses Python objects to represent tables and queries. The library
    generates the correct SQL dialect for whatever database you connect to.
    When you parameterize values through SQLAlchemy, injection is impossible
    because values are sent separately from the SQL command.

SCHEMA DESIGN:
    ┌──────────────────┐      ┌──────────────────┐
    │  user_features   │      │  user_segments   │
    │  user_id (PK)    │─────▶│  user_id (FK)    │
    │  n_views         │      │  kmeans_cluster  │
    │  n_addtocart     │      │  segment_name    │
    │  ...             │      └──────────────────┘
    └──────────────────┘
                                       │  (segment_name → segment_profiles)
    ┌──────────────────┐      ┌──────────────────┐
    │ segment_funnels  │      │ segment_profiles │
    │ segment_name(PK) │      │ segment_name(PK) │
    │ pct_reached_view │      │ avg_n_views      │
    │ pct_converted    │      │ avg_cart_ratio   │
    │ ...              │      │ ...              │
    └──────────────────┘      └──────────────────┘
"""

import pandas as pd
import numpy as np
from sqlalchemy import (
    create_engine, text,
    Column, Integer, BigInteger, Float, String, Boolean,
    MetaData, Table,
)
from sqlalchemy.exc import OperationalError

from cja.config import (
    DATABASE_URL,
    USER_FEATURES_FILE,
    USER_SEGMENTS_FILE,
    SEGMENT_FUNNELS_FILE,
)


# ── Engine Factory ─────────────────────────────────────────────────────────────

def get_engine(database_url: str = DATABASE_URL):
    """
    Creates and returns a SQLAlchemy engine.

    The engine is the entry point for all database operations. It manages a
    connection pool — a set of reusable open connections — so the app doesn't
    pay the cost of opening a new TCP connection for every query.

    Parameters
    ----------
    database_url : str
        The full connection string. Defaults to the value in config.py which
        reads from environment variables. Tests can pass a different URL (e.g.,
        SQLite) to run without a real Postgres instance.
    """
    return create_engine(database_url, echo=False)


# ── Schema Definition ──────────────────────────────────────────────────────────

def get_metadata_and_tables(engine):
    """
    Defines all table schemas and returns the SQLAlchemy MetaData object.

    MetaData is a registry that holds the definitions of all tables. Think of
    it as the 'blueprint folder' for the whole database. Calling
    `metadata.create_all(engine)` uses these blueprints to CREATE TABLE
    statements if the tables don't already exist.

    We use `extend_existing=True` so calling this function multiple times
    (e.g., in tests) doesn't raise a 'table already defined' error.
    """
    metadata = MetaData()

    # ── Table 1: user_features ─────────────────────────────────────────────────
    # One row per user. Stores the behavioral feature vector computed in Phase 3.
    # This is the source of truth for what we know about each user.
    Table(
        "user_features",
        metadata,
        Column("user_id",                  BigInteger, primary_key=True),
        Column("total_events",             Integer),
        Column("n_views",                  Integer),
        Column("n_addtocart",              Integer),
        Column("n_transactions",           Integer),
        Column("n_distinct_items",         Integer),
        Column("n_sessions",               Integer),
        Column("avg_events_per_session",   Float),
        Column("avg_session_duration_sec", Float),
        Column("total_time_on_site_sec",   Float),
        Column("max_funnel_stage",         Integer),
        Column("reached_cart",             Integer),   # 0 or 1
        Column("converted",                Integer),   # 0 or 1
        Column("cart_to_view_ratio",       Float),
        Column("purchase_to_cart_ratio",   Float),
        Column("recency_days",             Float),
        Column("active_span_days",         Float),
        extend_existing=True,
    )

    # ── Table 2: user_segments ─────────────────────────────────────────────────
    # One row per user. Links each user to their discovered segment.
    # This is what the API will query when asked "what segment is user X in?"
    Table(
        "user_segments",
        metadata,
        Column("user_id",        BigInteger, primary_key=True),
        Column("kmeans_cluster", Integer),
        Column("dbscan_cluster", Integer),
        Column("segment_name",   String(100)),
        extend_existing=True,
    )

    # ── Table 3: segment_profiles ──────────────────────────────────────────────
    # One row per segment. Stores the mean behavioral features per segment —
    # this is the 'cluster profile table' from the clustering output.
    # Used by the API to return segment summaries without re-aggregating.
    Table(
        "segment_profiles",
        metadata,
        Column("segment_id",               Integer, primary_key=True),
        Column("segment_name",             String(100)),
        Column("n_users",                  Integer),
        Column("avg_n_views",              Float),
        Column("avg_n_addtocart",          Float),
        Column("avg_n_transactions",       Float),
        Column("avg_n_sessions",           Float),
        Column("avg_cart_to_view_ratio",   Float),
        Column("avg_purchase_to_cart_ratio", Float),
        Column("avg_max_funnel_stage",     Float),
        Column("avg_recency_days",         Float),
        Column("avg_active_span_days",     Float),
        extend_existing=True,
    )

    # ── Table 4: segment_funnels ───────────────────────────────────────────────
    # One row per segment. Stores the funnel conversion metrics from Phase 5.
    Table(
        "segment_funnels",
        metadata,
        Column("segment_id",               Integer, primary_key=True),
        Column("segment_name",             String(100)),
        Column("n_users",                  Integer),
        Column("pct_reached_view",         Float),
        Column("pct_reached_cart",         Float),
        Column("pct_converted",            Float),
        Column("dropoff_view_to_cart",     Float),
        Column("dropoff_cart_to_purchase", Float),
        Column("biggest_dropoff_stage",    String(50)),
        extend_existing=True,
    )

    return metadata


# ── Data Loading ───────────────────────────────────────────────────────────────

def build_segment_profiles(segments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the per-segment mean feature profile from the user_segments table.
    This is a fresh aggregation — not reading from a parquet file —
    so the profiles are always consistent with the segment assignments.
    """
    profile_cols = [
        'n_views', 'n_addtocart', 'n_transactions', 'n_sessions',
        'cart_to_view_ratio', 'purchase_to_cart_ratio',
        'max_funnel_stage', 'recency_days', 'active_span_days',
    ]

    profiles = segments_df.groupby(['kmeans_cluster', 'segment_name'])[profile_cols].mean()
    profiles = profiles.rename(columns={c: f"avg_{c}" for c in profile_cols})
    profiles['n_users'] = segments_df.groupby(['kmeans_cluster', 'segment_name']).size()
    profiles = profiles.reset_index().rename(columns={'kmeans_cluster': 'segment_id'})
    return profiles.round(4)


def load_all_data(engine, if_exists: str = "replace") -> None:
    """
    Loads all processed parquet files into the database tables.

    Parameters
    ----------
    engine : sqlalchemy.Engine
        The database engine to write to.
    if_exists : str
        What to do if a table already has data: 'replace' drops and recreates
        the table, 'append' adds rows. Default is 'replace' for idempotency —
        running the pipeline twice gives the same result.

    WHY 'replace' not 'append'?
        Idempotency means the same input always produces the same output,
        regardless of how many times you run it. If you 'append', running
        the pipeline twice doubles the rows. 'replace' ensures a fresh,
        consistent load every time.
    """
    # ── Load user_features ─────────────────────────────────────────────────────
    print("Loading user_features.parquet...")
    features_df = pd.read_parquet(USER_FEATURES_FILE)

    cols_to_store = [
        'user_id', 'total_events', 'n_views', 'n_addtocart', 'n_transactions',
        'n_distinct_items', 'n_sessions', 'avg_events_per_session',
        'avg_session_duration_sec', 'total_time_on_site_sec',
        'max_funnel_stage', 'reached_cart', 'converted',
        'cart_to_view_ratio', 'purchase_to_cart_ratio',
        'recency_days', 'active_span_days',
    ]
    features_df[cols_to_store].to_sql(
        "user_features", engine, if_exists=if_exists, index=False, chunksize=5000
    )
    print(f"  Loaded {len(features_df):,} rows into user_features.")

    # ── Load user_segments ─────────────────────────────────────────────────────
    print("Loading user_segments.parquet...")
    segments_df = pd.read_parquet(USER_SEGMENTS_FILE)

    segments_df[['user_id', 'kmeans_cluster', 'dbscan_cluster', 'segment_name']].to_sql(
        "user_segments", engine, if_exists=if_exists, index=False, chunksize=5000
    )
    print(f"  Loaded {len(segments_df):,} rows into user_segments.")

    # ── Build and load segment_profiles ───────────────────────────────────────
    print("Building segment profiles...")
    profiles_df = build_segment_profiles(segments_df)
    profiles_df.to_sql(
        "segment_profiles", engine, if_exists=if_exists, index=False
    )
    print(f"  Loaded {len(profiles_df):,} rows into segment_profiles.")

    # ── Load segment_funnels ───────────────────────────────────────────────────
    print("Loading segment_funnels.parquet...")
    funnels_df = pd.read_parquet(SEGMENT_FUNNELS_FILE)
    funnels_df.to_sql(
        "segment_funnels", engine, if_exists=if_exists, index=False
    )
    print(f"  Loaded {len(funnels_df):,} rows into segment_funnels.")


# ── Verification Query ─────────────────────────────────────────────────────────

def verify_load(engine) -> None:
    """
    Runs simple COUNT(*) queries on every table and prints row counts.
    This confirms that data was written correctly without reading it all back.
    """
    tables = ["user_features", "user_segments", "segment_profiles", "segment_funnels"]
    print("\nVerification — row counts:")
    print("-" * 35)
    with engine.connect() as conn:
        for table in tables:
            # text() wraps a raw SQL string safely in SQLAlchemy
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()   # .scalar() extracts the single integer value
            print(f"  {table:<25} {count:>8,} rows")


# ── Main Entry Point ───────────────────────────────────────────────────────────

def setup_database(database_url: str = DATABASE_URL) -> None:
    """
    Full pipeline: connect → create schema → load data → verify.
    """
    print(f"Connecting to database...")
    engine = get_engine(database_url)

    # Test the connection before attempting to load data
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("  Connection successful.")
    except OperationalError as e:
        print(f"\nERROR: Cannot connect to the database.")
        print(f"  Is PostgreSQL running? Did you set the DB_* environment variables?")
        print(f"  Details: {e}")
        return

    # Create all tables if they don't exist
    print("Creating schema (if not exists)...")
    metadata = get_metadata_and_tables(engine)
    metadata.create_all(engine)
    print("  Schema ready.")

    # Load data
    load_all_data(engine)

    # Verify
    verify_load(engine)

    print("\nDatabase setup complete!")


if __name__ == "__main__":
    setup_database()
