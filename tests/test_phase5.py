import pytest
import pandas as pd
from cja.config import SEGMENT_FUNNELS_FILE, USER_SEGMENTS_FILE


def test_funnel_file_exists():
    if not SEGMENT_FUNNELS_FILE.exists():
        pytest.skip("Funnel file not generated. Run Phase 5 first.")


def test_one_row_per_segment():
    """The funnel table must have exactly one row per unique segment."""
    if not SEGMENT_FUNNELS_FILE.exists():
        pytest.skip("Funnel file not generated.")

    funnel_df = pd.read_parquet(SEGMENT_FUNNELS_FILE)
    segments_df = pd.read_parquet(USER_SEGMENTS_FILE)

    expected_segments = segments_df['segment_name'].nunique()
    actual_rows = len(funnel_df)

    assert actual_rows == expected_segments, \
        f"Expected {expected_segments} funnel rows, got {actual_rows}."


def test_funnel_is_non_increasing():
    """
    Core funnel invariant: you cannot have more users at a later stage
    than at an earlier one. view% >= cart% >= purchase% must hold for every segment.
    """
    if not SEGMENT_FUNNELS_FILE.exists():
        pytest.skip("Funnel file not generated.")

    funnel_df = pd.read_parquet(SEGMENT_FUNNELS_FILE)

    for _, row in funnel_df.iterrows():
        assert row['pct_reached_view'] >= row['pct_reached_cart'], \
            f"Funnel violation in '{row['segment_name']}': cart% > view%"
        assert row['pct_reached_cart'] >= row['pct_converted'], \
            f"Funnel violation in '{row['segment_name']}': purchase% > cart%"


def test_percentages_in_valid_range():
    """All percentage values must be between 0 and 100."""
    if not SEGMENT_FUNNELS_FILE.exists():
        pytest.skip("Funnel file not generated.")

    funnel_df = pd.read_parquet(SEGMENT_FUNNELS_FILE)
    pct_cols = ['pct_reached_view', 'pct_reached_cart', 'pct_converted']

    for col in pct_cols:
        assert (funnel_df[col] >= 0).all() and (funnel_df[col] <= 100).all(), \
            f"Column '{col}' has values outside [0, 100]."


def test_converters_have_highest_conversion():
    """
    Business logic test: the 'Converters' segment must have a higher
    pct_converted than all other segments. This confirms our clustering
    separated buyers from non-buyers correctly.
    """
    if not SEGMENT_FUNNELS_FILE.exists():
        pytest.skip("Funnel file not generated.")

    funnel_df = pd.read_parquet(SEGMENT_FUNNELS_FILE)

    if 'Converters' not in funnel_df['segment_name'].values:
        pytest.skip("'Converters' segment not present.")

    converter_pct = funnel_df.loc[
        funnel_df['segment_name'] == 'Converters', 'pct_converted'
    ].iloc[0]

    other_max_pct = funnel_df.loc[
        funnel_df['segment_name'] != 'Converters', 'pct_converted'
    ].max()

    assert converter_pct > other_max_pct, \
        "Converters segment does not have the highest conversion rate!"


def test_dropoffs_sum_correctly():
    """
    Sanity check: the two dropoff values plus the final conversion rate
    must sum to 100% (they represent the full journey from view to end).
    dropoff_view_to_cart + dropoff_cart_to_purchase + pct_converted = 100
    """
    if not SEGMENT_FUNNELS_FILE.exists():
        pytest.skip("Funnel file not generated.")

    funnel_df = pd.read_parquet(SEGMENT_FUNNELS_FILE)

    for _, row in funnel_df.iterrows():
        total = (
            row['dropoff_view_to_cart']
            + row['dropoff_cart_to_purchase']
            + row['pct_converted']
        )
        assert abs(total - 100.0) < 0.01, \
            f"Dropoffs don't sum to 100 for segment '{row['segment_name']}': got {total}"
