import pandas as pd
import numpy as np
from cja.config import RAW_EVENTS_FILE, CANONICAL_EVENTS_FILE, N_SAMPLE_VISITORS, RANDOM_SEED

def load_and_prepare_data(raw_filepath=RAW_EVENTS_FILE, output_filepath=CANONICAL_EVENTS_FILE):
    """
    Loads raw Retailrocket events, maps them to a canonical schema,
    applies visitor sampling, and saves to parquet.
    """
    if not raw_filepath.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_filepath}. "
            "Please download 'events.csv' from Kaggle and place it in the data/raw directory."
        )

    print(f"Loading raw data from {raw_filepath}...")
    df = pd.read_csv(raw_filepath)
    
    # 1. Map to Canonical Schema
    # Rename columns to standard terminology so downstream code is dataset-agnostic
    df = df.rename(columns={
        'visitorid': 'user_id',
        'event': 'event_type',
        'itemid': 'item_id',
        'transactionid': 'transaction_id'
    })
    
    # Convert Unix epoch ms to datetime
    df['event_timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop(columns=['timestamp'])
    
    # 2. Map event_type to funnel_stage and stage_index
    # The funnel stages are ordered: view -> addtocart -> transaction (0 -> 1 -> 2)
    funnel_mapping = {
        'view': 0,
        'addtocart': 1,
        'transaction': 2
    }
    df['stage_index'] = df['event_type'].map(funnel_mapping)
    df['funnel_stage'] = df['event_type'] # redundant but explicit per specs
    
    # 3. Apply visitor sampling
    # We sample by visitor (not by event) to preserve entire customer journeys.
    # Otherwise, we might get half a journey which corrupts our behavioral features.
    unique_visitors = df['user_id'].unique()
    
    if len(unique_visitors) > N_SAMPLE_VISITORS:
        print(f"Sampling {N_SAMPLE_VISITORS} visitors from {len(unique_visitors)} total visitors...")
        # Use a fixed random seed for reproducibility across different runs/machines
        rng = np.random.default_rng(seed=RANDOM_SEED)
        sampled_visitors = rng.choice(unique_visitors, size=N_SAMPLE_VISITORS, replace=False)
        
        # Filter the dataframe to only include events from sampled visitors
        df = df[df['user_id'].isin(sampled_visitors)]
    
    # Sort events chronologically per user to help downstream sessionization (Phase 2)
    df = df.sort_values(by=['user_id', 'event_timestamp']).reset_index(drop=True)
    
    # 4. Save to processed parquet file
    print(f"Saving canonical data to {output_filepath}...")
    # Using PyArrow engine to save as parquet. Parquet is heavily compressed and fast to read.
    df.to_parquet(output_filepath, index=False)
    
    print("Data acquisition complete!")
    print("\nVerification Stats:")
    print("-" * 20)
    print(df['event_type'].value_counts())
    print(f"Total rows: {len(df)}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Time span: {df['event_timestamp'].min()} to {df['event_timestamp'].max()}")
    
    return df

if __name__ == "__main__":
    load_and_prepare_data()
