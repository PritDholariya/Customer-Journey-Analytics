"""
api.py — Phase 8: FastAPI REST API

PURPOSE:
    Exposes the database tables built in Phase 7 as HTTP endpoints.
    Any external system — a dashboard, a marketing tool, a mobile app —
    can query customer segments and funnel data over HTTP without needing
    direct database access or Python knowledge.

WHY FASTAPI?
    FastAPI is a modern Python web framework with three key advantages:
    1. AUTOMATIC DOCS: visiting /docs gives a live, interactive Swagger UI
       where you can test every endpoint in your browser.
    2. DATA VALIDATION: request and response data is defined using Pydantic
       models (Python classes with type annotations). FastAPI auto-validates
       all incoming and outgoing data and returns clear error messages.
    3. PERFORMANCE: Built on Starlette (async framework) + Uvicorn (ASGI server),
       FastAPI is one of the fastest Python web frameworks — comparable to Go
       and Node.js in benchmarks.

REQUEST / RESPONSE LIFECYCLE:
    Client                FastAPI
      │                      │
      │  GET /segments        │
      │──────────────────────▶│
      │                      │── validate request
      │                      │── query database (SQLAlchemy)
      │                      │── serialize result (Pydantic)
      │                      │── return JSON
      │◀──────────────────────│
      │  200 OK + JSON body   │

API ENDPOINTS:
    GET  /health                  → health check (is the API alive?)
    GET  /segments                → list all 5 segments with profiles
    GET  /segments/{segment_name} → one segment's full profile
    GET  /segments/{segment_name}/funnel → funnel metrics for one segment
    GET  /users/{user_id}/segment → which segment does this user belong to?
    GET  /funnels                 → all segments' funnel metrics side by side
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from cja.config import DATABASE_URL


# ── Pydantic Response Models ───────────────────────────────────────────────────
# These classes define the SHAPE of JSON responses.
# FastAPI reads the return type annotation on each endpoint function and uses
# it to (a) validate the data, (b) serialize to JSON, (c) generate API docs.
#
# Every field has a type annotation. Optional[float] means the field can be
# a float OR None. This matters for segments with zero cart activity —
# their ratio columns will be None/null.

class SegmentProfile(BaseModel):
    """One row from segment_profiles: the average behavioral fingerprint."""
    segment_id:                 int
    segment_name:               str
    n_users:                    int
    avg_n_views:                float
    avg_n_addtocart:            float
    avg_n_transactions:         float
    avg_n_sessions:             float
    avg_cart_to_view_ratio:     float
    avg_purchase_to_cart_ratio: float
    avg_max_funnel_stage:       float
    avg_recency_days:           float
    avg_active_span_days:       float


class SegmentFunnel(BaseModel):
    """One row from segment_funnels: where does each segment drop off?"""
    segment_id:                int
    segment_name:              str
    n_users:                   int
    pct_reached_view:          float
    pct_reached_cart:          float
    pct_converted:             float
    dropoff_view_to_cart:      float
    dropoff_cart_to_purchase:  float
    biggest_dropoff_stage:     str


class UserSegment(BaseModel):
    """Minimal response for the user lookup endpoint."""
    user_id:        int
    kmeans_cluster: int
    segment_name:   str


class HealthCheck(BaseModel):
    status:   str
    database: str


# ── Database Dependency ────────────────────────────────────────────────────────
# FastAPI's dependency injection system works like this:
#   - A dependency is any callable that returns a value
#   - Endpoints declare dependencies in their function signature using Depends()
#   - FastAPI calls the dependency, passes the result into the endpoint function
#
# This pattern keeps database connection logic OUT of endpoint functions.
# It also makes testing easy: in tests, you can override the dependency with a
# test database engine without changing any endpoint code.

_engine: Optional[Engine] = None


def get_db_engine() -> Engine:
    """
    Returns the shared SQLAlchemy engine.
    The engine is created ONCE when the API starts (via the lifespan handler)
    and reused for every request. Creating a new engine per request would
    be extremely slow — each engine creates its own connection pool.
    """
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Database engine not initialized. Check DB_* environment variables."
        )
    return _engine


# ── Application Lifespan ───────────────────────────────────────────────────────
# The lifespan context manager runs startup and shutdown code for the app.
# Code before `yield` runs on startup; code after `yield` runs on shutdown.
# This is the modern FastAPI pattern (replaces the older @app.on_event approach).

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the database connection pool on startup, close it on shutdown."""
    global _engine
    print("API starting up — connecting to database...")
    _engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10)
    print("  Database engine ready.")
    yield
    # Shutdown: dispose the connection pool cleanly
    if _engine:
        _engine.dispose()
        print("API shutting down — database connections closed.")


# ── FastAPI App Instance ───────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Journey Analytics API",
    description=(
        "REST API for querying customer segment profiles and funnel metrics. "
        "Built on the Retailrocket e-commerce dataset using K-Means clustering."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoint 1: Health Check ───────────────────────────────────────────────────

@app.get("/health", response_model=HealthCheck, tags=["System"])
def health_check(engine: Engine = Depends(get_db_engine)):
    """
    Confirms the API is alive and the database connection is working.
    Monitoring tools ping this endpoint every few seconds to detect outages.

    Returns 200 OK if healthy, 503 if the database is unreachable.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "unreachable"

    return HealthCheck(status="ok", database=db_status)


# ── Endpoint 2: List All Segments ─────────────────────────────────────────────

@app.get("/segments", response_model=list[SegmentProfile], tags=["Segments"])
def list_segments(engine: Engine = Depends(get_db_engine)):
    """
    Returns the behavioral profile of all 5 customer segments.

    Each segment profile shows average feature values for users in that group
    (e.g., average views, average cart-to-view ratio).
    Segments are ordered by segment_id.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM segment_profiles ORDER BY segment_id")
        ).mappings().all()
        # .mappings() converts each Row to a dict-like object so Pydantic can read it

    if not rows:
        raise HTTPException(status_code=404, detail="No segment profiles found.")

    return [SegmentProfile(**row) for row in rows]


# ── Endpoint 3: Single Segment Profile ────────────────────────────────────────

@app.get("/segments/{segment_name}", response_model=SegmentProfile, tags=["Segments"])
def get_segment(segment_name: str, engine: Engine = Depends(get_db_engine)):
    """
    Returns the profile of a single segment by name.

    Example: GET /segments/Cart%20Abandoners
    (Note: spaces in URLs must be percent-encoded as %20)
    """
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM segment_profiles WHERE segment_name = :name"),
            {"name": segment_name}     # ← parameterized — NOT string concatenation
        ).mappings().first()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Segment '{segment_name}' not found. "
                   f"Use GET /segments to see all available segment names."
        )

    return SegmentProfile(**row)


# ── Endpoint 4: Segment Funnel ────────────────────────────────────────────────

@app.get("/segments/{segment_name}/funnel", response_model=SegmentFunnel, tags=["Segments"])
def get_segment_funnel(segment_name: str, engine: Engine = Depends(get_db_engine)):
    """
    Returns funnel drop-off metrics for a single segment.

    Shows the percentage of users who reached each funnel stage (view, cart,
    purchase) and the drop-off rate between consecutive stages.

    Example: GET /segments/Cart%20Abandoners/funnel
    """
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM segment_funnels WHERE segment_name = :name"),
            {"name": segment_name}
        ).mappings().first()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Funnel data for segment '{segment_name}' not found."
        )

    return SegmentFunnel(**row)


# ── Endpoint 5: All Funnels ────────────────────────────────────────────────────

@app.get("/funnels", response_model=list[SegmentFunnel], tags=["Funnels"])
def list_funnels(engine: Engine = Depends(get_db_engine)):
    """
    Returns funnel conversion metrics for ALL segments side by side.

    Ordered by segment_id. Useful for a dashboard that compares conversion
    rates and drop-off stages across all customer segments simultaneously.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM segment_funnels ORDER BY segment_id")
        ).mappings().all()

    if not rows:
        raise HTTPException(status_code=404, detail="No funnel data found.")

    return [SegmentFunnel(**row) for row in rows]


# ── Endpoint 6: User Lookup ────────────────────────────────────────────────────

@app.get("/users/{user_id}/segment", response_model=UserSegment, tags=["Users"])
def get_user_segment(user_id: int, engine: Engine = Depends(get_db_engine)):
    """
    Returns which segment a specific user belongs to.

    This is the core real-time lookup endpoint. Given a user_id (e.g., from
    a login event), returns their segment name so the marketing system can
    personalise their experience.

    Example: GET /users/257597/segment
    """
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT user_id, kmeans_cluster, segment_name "
                 "FROM user_segments WHERE user_id = :uid"),
            {"uid": user_id}
        ).mappings().first()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found in the segmentation database."
        )

    return UserSegment(**row)
