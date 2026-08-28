import pytest
import pandas as pd
from cja.config import CLEAN_EVENTS_FILE, MIN_EVENTS_PER_USER, SESSION_INACTIVITY_GAP_MINUTES

def test_no_duplicates():
    if not CLEAN_EVENTS_FILE.exists():
        pytest.skip("Clean file not generated.")
    df = pd.read_parquet(CLEAN_EVENTS_FILE)
    assert not df.duplicated().any(), "Duplicate rows exist."

def test_min_events_per_user():
    if not CLEAN_EVENTS_FILE.exists():
        pytest.skip("Clean file not generated.")
    df = pd.read_parquet(CLEAN_EVENTS_FILE)
    events_per_user = df.groupby('user_id').size()
    assert (events_per_user >= MIN_EVENTS_PER_USER).all(), f"Found users with fewer than {MIN_EVENTS_PER_USER} events."

def test_session_gaps():
    if not CLEAN_EVENTS_FILE.exists():
        pytest.skip("Clean file not generated.")
    df = pd.read_parquet(CLEAN_EVENTS_FILE)
    
    # Check within a single session, all gaps are <= 30 mins
    df = df.sort_values(['user_id', 'event_timestamp'])
    df['time_gap'] = df.groupby('session_id')['event_timestamp'].diff()
    
    # Convert to minutes for easier assertion
    gap_minutes = df['time_gap'].dt.total_seconds() / 60.0
    
    # max gap within any session should be less than or equal to threshold
    if not gap_minutes.isna().all(): 
        assert gap_minutes.max() <= SESSION_INACTIVITY_GAP_MINUTES, "Found events in the same session separated by >30 mins."
