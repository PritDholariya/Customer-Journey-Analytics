"""
funnel.py — Phase 5: Funnel & Drop-off Analysis

PURPOSE:
    Takes the segmented user table (one row per user with their cluster label)
    and computes, per segment, how many users reached each stage of the purchase
    funnel and how many dropped off between stages.

THE FUNNEL SHAPE:
    view → addtocart → transaction
      100%     ?%          ?%

    Every user in our dataset has at least one view (by construction — they got
    into the dataset by generating an event). So view = 100% for every segment.
    The interesting question is: what fraction of each segment makes it past view
    into addtocart, and then from addtocart into transaction?

OUTPUT (one row per segment):
    segment_name, segment_id,
    n_users,
    pct_reached_view,       ← always ~100%, confirms our logic
    pct_reached_cart,       ← share who added ≥1 item to cart
    pct_converted,          ← share who made ≥1 purchase
    dropoff_view_to_cart,   ← view% − cart%  (biggest gap for most segments)
    dropoff_cart_to_purchase,← cart% − purchase%  (biggest gap for Cart Abandoners)
    biggest_dropoff_stage   ← label of the worst stage for each segment
"""

import pandas as pd
import numpy as np
from cja.config import USER_SEGMENTS_FILE, SEGMENT_FUNNELS_FILE


def compute_funnel_for_segment(segment_df: pd.DataFrame) -> dict:
    """
    Given a DataFrame of users all belonging to ONE segment, compute
    funnel stage conversion rates and drop-offs.

    Parameters
    ----------
    segment_df : pd.DataFrame
        Subset of the user_segments table for a single cluster.

    Returns
    -------
    dict
        Funnel metrics for this segment (one row worth of data).
    """
    n_users = len(segment_df)

    # ── Stage Conversion Rates ────────────────────────────────────────────────
    # These columns were computed in Phase 3 (features.py):
    #   max_funnel_stage: 0 = only viewed, 1 = also added to cart, 2 = also bought
    #   reached_cart: 1 if max_funnel_stage >= 1, else 0
    #   converted: 1 if max_funnel_stage == 2, else 0

    # IMPORTANT: We measure "share of users who EVER reached this stage"
    # This is a user-level funnel, not an event-level funnel.
    # A user is counted once, regardless of how many times they repeated the stage.

    n_reached_view = n_users  # All users have at least one view by definition
    n_reached_cart = int(segment_df['reached_cart'].sum())
    n_converted = int(segment_df['converted'].sum())

    # Convert to percentages (0–100 scale for readability)
    pct_view = 100.0
    pct_cart = round((n_reached_cart / n_users) * 100, 2) if n_users > 0 else 0.0
    pct_purchase = round((n_converted / n_users) * 100, 2) if n_users > 0 else 0.0

    # ── Drop-off Rates ────────────────────────────────────────────────────────
    # Drop-off = percentage of users lost between consecutive stages.
    # E.g., if 40% reached cart and 5% converted, drop-off at cart→purchase = 35 points.
    # This is an ABSOLUTE drop in share, not a relative drop.

    dropoff_view_to_cart = round(pct_view - pct_cart, 2)
    dropoff_cart_to_purchase = round(pct_cart - pct_purchase, 2)

    # The "biggest drop" tells us WHERE to focus improvement efforts.
    # If view→cart drop dominates → product discovery / intent problem.
    # If cart→purchase drop dominates → checkout / trust / friction problem.
    biggest_dropoff_stage = (
        "view_to_cart"
        if dropoff_view_to_cart >= dropoff_cart_to_purchase
        else "cart_to_purchase"
    )

    return {
        'n_users': n_users,
        'pct_reached_view': pct_view,
        'pct_reached_cart': pct_cart,
        'pct_converted': pct_purchase,
        'dropoff_view_to_cart': dropoff_view_to_cart,
        'dropoff_cart_to_purchase': dropoff_cart_to_purchase,
        'biggest_dropoff_stage': biggest_dropoff_stage,
    }


def build_funnel_table(
    input_filepath=USER_SEGMENTS_FILE,
    output_filepath=SEGMENT_FUNNELS_FILE,
) -> pd.DataFrame:
    """
    Main orchestrator for Phase 5.

    Loads the segmented user table, groups by segment, computes funnel metrics
    for each segment, validates the results, and saves to parquet.

    Returns
    -------
    pd.DataFrame
        The segment_funnels table (one row per segment).
    """
    if not input_filepath.exists():
        raise FileNotFoundError(
            f"Segments file not found at {input_filepath}. Run Phase 4 first."
        )

    print(f"Loading segmented users from {input_filepath}...")
    df = pd.read_parquet(input_filepath)

    print(f"Computing funnels for {df['segment_name'].nunique()} segments...\n")

    # Group by segment and apply the funnel computation to each group
    results = []
    for cluster_id, group in df.groupby('kmeans_cluster'):
        segment_name = group['segment_name'].iloc[0]
        metrics = compute_funnel_for_segment(group)
        metrics['segment_id'] = int(cluster_id)
        metrics['segment_name'] = segment_name
        results.append(metrics)

    # Assemble into a clean, well-ordered DataFrame
    funnel_df = pd.DataFrame(results)
    col_order = [
        'segment_id', 'segment_name', 'n_users',
        'pct_reached_view', 'pct_reached_cart', 'pct_converted',
        'dropoff_view_to_cart', 'dropoff_cart_to_purchase',
        'biggest_dropoff_stage',
    ]
    funnel_df = funnel_df[col_order].sort_values('segment_id').reset_index(drop=True)

    # ── Validation ────────────────────────────────────────────────────────────
    # INVARIANT: The funnel must be non-increasing.
    # You can never have MORE users at a later stage than an earlier stage.
    # (You can't buy something without first viewing and carting it.)
    for _, row in funnel_df.iterrows():
        assert row['pct_reached_view'] >= row['pct_reached_cart'], \
            f"Funnel violation in {row['segment_name']}: cart% > view%"
        assert row['pct_reached_cart'] >= row['pct_converted'], \
            f"Funnel violation in {row['segment_name']}: purchase% > cart%"

    print("Funnel table (all segments):")
    print("=" * 80)
    print(funnel_df.to_string(index=False))
    print("\nVALIDATION PASSED: funnel is non-increasing for all segments.")

    print(f"\nSaving funnel table to {output_filepath}...")
    funnel_df.to_parquet(output_filepath, index=False)
    print("Funnel analysis complete!")

    return funnel_df


if __name__ == "__main__":
    build_funnel_table()
