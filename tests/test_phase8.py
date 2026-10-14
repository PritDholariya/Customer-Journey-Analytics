"""
test_phase8.py — Unit Tests for Phase 8: FastAPI Endpoints

TESTING STRATEGY:
    We use FastAPI's TestClient — a wrapper around the 'requests' library
    that sends HTTP requests to the app in-process without starting a real
    server.

    KEY FIX — SQLite StaticPool:
    A plain `sqlite://` URL creates a NEW in-memory database for EVERY new
    connection. When the test creates tables and inserts data on connection A,
    then FastAPI's endpoint opens connection B, it sees an empty database.
    The fix is SQLAlchemy's StaticPool — it forces ALL operations through a
    SINGLE persistent connection, so the same in-memory DB is shared across
    table creation, data loading, and every endpoint query.
"""

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from cja.config import (
    USER_FEATURES_FILE,
    USER_SEGMENTS_FILE,
    SEGMENT_FUNNELS_FILE,
    TEST_DATABASE_URL,
)
from cja.database import get_metadata_and_tables, load_all_data
from cja.api import app, get_db_engine


# ── Test Setup ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_engine():
    """
    Creates one SQLite in-memory database for the entire test module.

    WHY StaticPool + check_same_thread=False?
    SQLite's default 'connect' pool creates a NEW in-memory database for each
    connection. StaticPool maintains EXACTLY ONE connection and reuses it for
    every request, guaranteeing the tables we create are visible to all
    subsequent queries — including those made from inside FastAPI endpoints.
    check_same_thread=False is required because FastAPI runs endpoint functions
    in a thread pool, not the main thread where the engine was created.
    """
    if not USER_SEGMENTS_FILE.exists():
        pytest.skip("Parquet files not found. Run Phases 1-5 first.")

    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata = get_metadata_and_tables(engine)
    metadata.create_all(engine)
    load_all_data(engine)
    return engine


@pytest.fixture(scope="module")
def client(test_engine):
    """
    Creates a FastAPI TestClient with the database dependency overridden
    to use the test SQLite engine instead of real PostgreSQL.
    """
    # Dependency override: replace the real DB engine with the test engine
    app.dependency_overrides[get_db_engine] = lambda: test_engine
    with TestClient(app) as c:
        yield c
    # Cleanup: remove the override after all tests in the module finish
    app.dependency_overrides.clear()


# ── Health Check ───────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    """The /health endpoint must return HTTP 200 when the DB is reachable."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── Segments Endpoint ──────────────────────────────────────────────────────────

def test_list_segments_returns_five(client):
    """GET /segments must return exactly 5 segment profiles (K=5 clustering)."""
    response = client.get("/segments")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5, f"Expected 5 segments, got {len(data)}"


def test_list_segments_response_shape(client):
    """Each segment profile must have all required fields with correct types."""
    response = client.get("/segments")
    segment = response.json()[0]

    required_fields = [
        "segment_id", "segment_name", "n_users",
        "avg_n_views", "avg_n_addtocart", "avg_cart_to_view_ratio",
        "avg_recency_days", "avg_active_span_days",
    ]
    for field in required_fields:
        assert field in segment, f"Missing field: {field}"


def test_get_converters_segment(client):
    """GET /segments/Converters must return the Converters profile."""
    response = client.get("/segments/Converters")
    assert response.status_code == 200
    data = response.json()
    assert data["segment_name"] == "Converters"
    # Converters have the highest purchase rate — their avg_n_transactions
    # must be greater than zero
    assert data["avg_n_transactions"] > 0


def test_get_unknown_segment_returns_404(client):
    """Requesting a non-existent segment must return HTTP 404."""
    response = client.get("/segments/NonExistentSegment")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ── Funnel Endpoints ───────────────────────────────────────────────────────────

def test_list_funnels_returns_five(client):
    """GET /funnels must return one funnel row per segment."""
    response = client.get("/funnels")
    assert response.status_code == 200
    assert len(response.json()) == 5


def test_cart_abandoners_funnel(client):
    """
    Cart Abandoners should have 100% cart rate and near-zero conversion.
    This test validates business logic through the API layer.
    """
    response = client.get("/segments/Cart%20Abandoners/funnel")
    assert response.status_code == 200
    data = response.json()

    assert data["pct_reached_cart"] == 100.0, \
        "Cart Abandoners must have 100% cart reach rate"
    assert data["pct_converted"] < 5.0, \
        "Cart Abandoners must have very low conversion rate"
    assert data["biggest_dropoff_stage"] == "cart_to_purchase"


def test_converters_funnel_100_percent(client):
    """Converters must have 100% conversion rate at every funnel stage."""
    response = client.get("/segments/Converters/funnel")
    assert response.status_code == 200
    data = response.json()
    assert data["pct_converted"] == 100.0


# ── User Lookup Endpoint ───────────────────────────────────────────────────────

def test_user_lookup_valid_user(client):
    """
    GET /users/{user_id}/segment must return a valid segment for a real user.
    We pick the first user_id from the parquet file as our test user.
    """
    segments_df = pd.read_parquet(USER_SEGMENTS_FILE)
    real_user_id = int(segments_df['user_id'].iloc[0])

    response = client.get(f"/users/{real_user_id}/segment")
    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == real_user_id
    assert "segment_name" in data
    assert data["segment_name"] in [
        "Casual Visitors", "Active Browsers", "Cart Abandoners",
        "Loyal Window Shoppers", "Converters"
    ]


def test_user_lookup_invalid_user_returns_404(client):
    """A user_id that doesn't exist in the DB must return 404."""
    response = client.get("/users/999999999/segment")
    assert response.status_code == 404


def test_user_id_must_be_integer(client):
    """Passing a non-integer user_id must return HTTP 422 Unprocessable Entity."""
    response = client.get("/users/not_a_number/segment")
    # FastAPI validates path parameter types automatically via Pydantic
    assert response.status_code == 422
