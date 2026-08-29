import pandas as pd
import numpy as np
from cja.config import CLEAN_EVENTS_FILE, USER_FEATURES_FILE

def calc_event_counts(df):
    """Calculates total events and counts per funnel stage."""
    # Count total events per user
    counts = df.groupby('user_id').size().rename('total_events').to_frame()
    
    # Pivot to get count of each event type per user
    event_piv = df.pivot_table(index='user_id', columns='event_type', aggfunc='size', fill_value=0)
    
    # Ensure columns exist even if some event types are missing in a tiny sample
    for col in ['view', 'addtocart', 'transaction']:
        if col not in event_piv.columns:
            event_piv[col] = 0
            
    counts['n_views'] = event_piv['view']
    counts['n_addtocart'] = event_piv['addtocart']
    counts['n_transactions'] = event_piv['transaction']
    
    return counts

def calc_item_diversity(df):
    """Calculates how many unique items a user interacted with."""
    return df.groupby('user_id')['item_id'].nunique().rename('n_distinct_items').to_frame()

def calc_session_metrics(df):
    """Calculates metrics derived from sessions (count, duration, intensity)."""
    # Number of sessions per user
    n_sessions = df.groupby('user_id')['session_id'].nunique().rename('n_sessions')
    
    # Session durations: max_time - min_time per session
    session_times = df.groupby('session_id')['event_timestamp'].agg(['min', 'max'])
    session_times['duration_sec'] = (session_times['max'] - session_times['min']).dt.total_seconds()
    
    # Join user_id back to session_id to aggregate to user level
    session_user_map = df[['session_id', 'user_id']].drop_duplicates().set_index('session_id')
    session_times = session_times.join(session_user_map)
    
    # Total and average time per user
    user_time = session_times.groupby('user_id')['duration_sec'].agg(['sum', 'mean'])
    user_time = user_time.rename(columns={'sum': 'total_time_on_site_sec', 'mean': 'avg_session_duration_sec'})
    
    res = pd.concat([n_sessions, user_time], axis=1)
    
    # Calculate avg events per session
    events_per_user = df.groupby('user_id').size()
    res['avg_events_per_session'] = events_per_user / res['n_sessions']
    
    return res

def calc_funnel_metrics(df, counts_df):
    """Calculates depth of funnel and transition ratios."""
    # Max funnel stage reached (0=view, 1=cart, 2=transaction)
    max_stage = df.groupby('user_id')['stage_index'].max().rename('max_funnel_stage').to_frame()
    
    # Booleans for quick segmenting
    max_stage['reached_cart'] = (max_stage['max_funnel_stage'] >= 1).astype(int)
    max_stage['converted'] = (max_stage['max_funnel_stage'] == 2).astype(int)
    
    # Ratios: guard against division by zero using np.where
    # cart_to_view_ratio: Intent signal (how often browsing turns into carts)
    max_stage['cart_to_view_ratio'] = np.where(
        counts_df['n_views'] > 0, 
        counts_df['n_addtocart'] / counts_df['n_views'], 
        0.0
    )
    
    # purchase_to_cart_ratio: Closing signal (how often carts turn into purchases)
    max_stage['purchase_to_cart_ratio'] = np.where(
        counts_df['n_addtocart'] > 0,
        counts_df['n_transactions'] / counts_df['n_addtocart'],
        0.0
    )
    
    return max_stage

def calc_time_metrics(df):
    """Calculates recency and lifetime span metrics."""
    # Global max date in the entire dataset to compute recency against
    global_max_date = df['event_timestamp'].max()
    
    user_times = df.groupby('user_id')['event_timestamp'].agg(['min', 'max'])
    
    # Recency: Days from user's last event to the dataset's maximum date
    recency = (global_max_date - user_times['max']).dt.total_seconds() / (24 * 3600)
    
    # Active Span: Days between user's first and last event
    span = (user_times['max'] - user_times['min']).dt.total_seconds() / (24 * 3600)
    
    return pd.DataFrame({'recency_days': recency, 'active_span_days': span})

def build_feature_table(input_filepath=CLEAN_EVENTS_FILE, output_filepath=USER_FEATURES_FILE):
    """
    Main orchestrator function.
    Reads clean event stream, aggregates all behavioral metrics to the user level,
    applies log transformations to skewed columns, and saves the feature table.
    """
    if not input_filepath.exists():
        raise FileNotFoundError(f"Clean data not found at {input_filepath}. Run Phase 2 first.")
    
    print(f"Loading clean data from {input_filepath}...")
    df = pd.read_parquet(input_filepath)
    
    print("Extracting features...")
    # Compute feature groups
    counts = calc_event_counts(df)
    items = calc_item_diversity(df)
    sessions = calc_session_metrics(df)
    funnel = calc_funnel_metrics(df, counts)
    times = calc_time_metrics(df)
    
    # Merge all features on user_id (the index)
    features = pd.concat([counts, items, sessions, funnel, times], axis=1)
    
    # Fill any NaNs generated during aggregations with 0
    # e.g., if a user's total time on site is 0, the mean might be NaN
    features = features.fillna(0)
    
    # Apply log1p transform to highly right-skewed count features
    # log1p is mathematical shorthand for log(1 + x). The '+1' prevents taking log(0) which is undefined.
    skewed_cols = ['total_events', 'n_views', 'n_addtocart', 'n_transactions', 
                   'n_distinct_items', 'n_sessions', 'total_time_on_site_sec']
    
    for col in skewed_cols:
        features[f"{col}_log"] = np.log1p(features[col])
        # We keep both: original for interpretability, _log for the ML clustering algorithms.
        
    # Make user_id a regular column instead of the index
    features = features.reset_index()
    
    print(f"Saving {len(features)} user feature rows to {output_filepath}...")
    features.to_parquet(output_filepath, index=False)
    
    print("Feature engineering complete!")
    print("\nVerification Stats:")
    print("-" * 20)
    print(f"Total Users: {len(features)}")
    print(f"Total Features (excluding user_id): {len(features.columns) - 1}")
    print(f"Missing Values (NaNs): {features.isna().sum().sum()}")
    
    return features

if __name__ == "__main__":
    build_feature_table()
