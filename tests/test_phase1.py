import pytest
import pandas as pd
from cja.config import CANONICAL_EVENTS_FILE

def test_canonical_schema_columns():
    """Verify that the generated parquet file has the expected canonical columns."""
    if not CANONICAL_EVENTS_FILE.exists():
        pytest.skip("Canonical file not generated yet. Run acquisition phase first.")
        
    df = pd.read_parquet(CANONICAL_EVENTS_FILE)
    expected_columns = {
        'user_id', 'event_timestamp', 'event_type', 'item_id', 
        'transaction_id', 'stage_index', 'funnel_stage'
    }
    assert expected_columns.issubset(set(df.columns)), "Missing canonical columns"

def test_funnel_stages():
    """Verify that funnel stages map correctly."""
    if not CANONICAL_EVENTS_FILE.exists():
        pytest.skip("Canonical file not generated yet. Run acquisition phase first.")
        
    df = pd.read_parquet(CANONICAL_EVENTS_FILE)
    
    # Transactions should have stage_index 2
    transactions = df[df['event_type'] == 'transaction']
    if not transactions.empty:
        assert (transactions['stage_index'] == 2).all()
        
def test_visitor_sampling():
    """Verify sampling limit."""
    if not CANONICAL_EVENTS_FILE.exists():
        pytest.skip("Canonical file not generated yet. Run acquisition phase first.")
        
    df = pd.read_parquet(CANONICAL_EVENTS_FILE)
    unique_users = df['user_id'].nunique()
    
    from cja.config import N_SAMPLE_VISITORS
    assert unique_users <= N_SAMPLE_VISITORS
