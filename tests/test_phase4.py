import pytest
import pandas as pd
import numpy as np
from cja.config import USER_SEGMENTS_FILE, KMEANS_K_FINAL

def test_segments_file_exists():
    if not USER_SEGMENTS_FILE.exists():
        pytest.skip("Segments file not generated. Run clustering phase first.")

def test_one_row_per_user():
    if not USER_SEGMENTS_FILE.exists():
        pytest.skip("Segments file not generated.")
    df = pd.read_parquet(USER_SEGMENTS_FILE)
    # user_id must be unique — no user should appear in two clusters
    assert df['user_id'].nunique() == len(df), "Duplicate user_ids found in segments file."

def test_kmeans_cluster_count():
    if not USER_SEGMENTS_FILE.exists():
        pytest.skip("Segments file not generated.")
    df = pd.read_parquet(USER_SEGMENTS_FILE)
    # K-Means must produce exactly KMEANS_K_FINAL clusters (0 through K-1)
    unique_clusters = df['kmeans_cluster'].nunique()
    assert unique_clusters == KMEANS_K_FINAL, \
        f"Expected {KMEANS_K_FINAL} K-Means clusters, got {unique_clusters}."

def test_no_nans_in_cluster_labels():
    if not USER_SEGMENTS_FILE.exists():
        pytest.skip("Segments file not generated.")
    df = pd.read_parquet(USER_SEGMENTS_FILE)
    # K-Means should assign every user to a cluster — no NaNs allowed
    assert df['kmeans_cluster'].isna().sum() == 0, "Found NaN values in kmeans_cluster column."
