import pandas as pd
import numpy as np
from cja.config import CANONICAL_EVENTS_FILE, CLEAN_EVENTS_FILE, MIN_EVENTS_PER_USER, SESSION_INACTIVITY_GAP_MINUTES

def clean_and_sessionize_data(input_filepath=CANONICAL_EVENTS_FILE, output_filepath=CLEAN_EVENTS_FILE):
    """
    Cleans the canonical event data (removes duplicates, filters low-activity users)
    and derives a session_id based on inactivity gaps.
    """
    if not input_filepath.exists():
        raise FileNotFoundError(f"Canonical data not found at {input_filepath}. Run Phase 1 first.")
    
    print(f"Loading canonical data from {input_filepath}...")
    df = pd.read_parquet(input_filepath)
    initial_rows = len(df)
    
    # 1. Drop exact duplicate rows
    # In clickstream data, exact duplicates are often tracking bugs (e.g., JS firing twice)
    df = df.drop_duplicates()
    print(f"Dropped {initial_rows - len(df)} exact duplicate rows.")
    
    # 2. Filter visitors with too few events
    # Users with only 1 event ("bouncers") often don't provide enough behavioral signal
    # for meaningful clustering. We drop them here to reduce noise.
    events_per_user = df.groupby('user_id').size()
    valid_users = events_per_user[events_per_user >= MIN_EVENTS_PER_USER].index
    
    df = df[df['user_id'].isin(valid_users)].copy()
    print(f"Dropped {len(events_per_user) - len(valid_users)} users with fewer than {MIN_EVENTS_PER_USER} events.")
    
    # 3. Sessionization
    # Real data often lacks session IDs. We derive them using the industry standard (Google Analytics):
    # A new session starts if the gap between consecutive events for a user > 30 minutes.
    print("Sessionizing data...")
    
    # Ensure data is sorted by user and time (crucial for time-delta calculations)
    df = df.sort_values(['user_id', 'event_timestamp'])
    
    # Calculate time difference to the previous event, per user
    df['time_since_last_event'] = df.groupby('user_id')['event_timestamp'].diff()
    
    # A new session starts if it's the user's first event (NaT) OR the gap > 30 minutes
    gap_threshold = pd.Timedelta(minutes=SESSION_INACTIVITY_GAP_MINUTES)
    is_new_session = df['time_since_last_event'].isna() | (df['time_since_last_event'] > gap_threshold)
    
    # Cumulative sum creates a unique session ID per user. 
    # It increments by 1 every time `is_new_session` is True.
    df['user_session_index'] = is_new_session.groupby(df['user_id']).cumsum()
    
    # Create a globally unique session ID by combining user_id and their session index
    df['session_id'] = df['user_id'].astype(str) + "_" + df['user_session_index'].astype(str)
    
    # Drop temporary calculation columns to keep the dataframe clean
    df = df.drop(columns=['time_since_last_event', 'user_session_index'])
    
    # 4. Save to processed parquet file
    print(f"Saving clean, sessionized data to {output_filepath}...")
    df.to_parquet(output_filepath, index=False)
    
    print("Data cleaning and sessionization complete!")
    print("\nVerification Stats:")
    print("-" * 20)
    print(f"Total rows: {len(df)}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Total sessions: {df['session_id'].nunique()}")
    print(f"Avg events per session: {len(df) / df['session_id'].nunique():.2f}")
    
    return df

if __name__ == "__main__":
    clean_and_sessionize_data()
