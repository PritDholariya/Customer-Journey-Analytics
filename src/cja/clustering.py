import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

from cja.config import (
    USER_FEATURES_FILE, USER_SEGMENTS_FILE, FIGURES_DIR,
    CLUSTERING_FEATURES, RANDOM_SEED,
    KMEANS_K_MIN, KMEANS_K_MAX, KMEANS_K_FINAL,
    DBSCAN_EPS, DBSCAN_MIN_SAMPLES,
)

# Human-readable names for each cluster.
# These were assigned by reading the mean feature values per cluster
# from the print_cluster_profiles() output and interpreting the behavior:
#
#  Cluster 0: 72% of users, ~2.5 views, 0 cart adds, 1.4 sessions → barely visited
#  Cluster 1: 12 views but cart_ratio=0.013 → lots of browsing, almost no intent
#  Cluster 2: cart_ratio=0.648 (very high!), max_funnel_stage=1 → adds cart, never buys
#  Cluster 3: 3.9 sessions, 49-day active span, no carts → loyal but non-converting
#  Cluster 4: purchase_to_cart_ratio=0.941, max_funnel_stage=2.0 → they buy
SEGMENT_NAMES = {
    0: "Casual Visitors",        # Drive-by, minimal engagement
    1: "Active Browsers",        # Browse a lot, very low purchase intent
    2: "Cart Abandoners",        # High intent (64% cart rate), but never purchase
    3: "Loyal Window Shoppers",  # Return often over weeks, but never convert
    4: "Converters",             # Small but valuable — 94% purchase rate
}



def load_and_scale_features(filepath=USER_FEATURES_FILE):
    """
    Loads the user feature table, selects only the ML-relevant columns,
    and scales them using StandardScaler.

    WHY SCALE?
    K-Means calculates distance between points. If `recency_days` ranges from
    0-150 and `cart_to_view_ratio` ranges from 0-1, then recency will dominate
    the distance calculation simply because its numbers are bigger — not because
    it is more important. StandardScaler brings all features to the same scale
    (mean=0, std=1) so every feature contributes equally.
    """
    print(f"Loading features from {filepath}...")
    df = pd.read_parquet(filepath)

    # Select only the features we want the ML model to see
    X = df[CLUSTERING_FEATURES].values  # Convert to numpy array for sklearn

    # Fit the scaler on the data and transform it
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Feature matrix shape: {X_scaled.shape} ({X_scaled.shape[0]} users, {X_scaled.shape[1]} features)")
    return df, X_scaled, scaler


def run_elbow_analysis(X_scaled):
    """
    Runs K-Means for k=2 to k=10 and records two metrics:
      1. Inertia (within-cluster sum of squared distances) — the 'elbow' method
      2. Silhouette Score — a measure of how well-separated the clusters are

    WHY TWO METRICS?
    Inertia always decreases as K increases (more clusters = smaller clusters = closer to centroid).
    That's why we look for the 'elbow' — the point where adding another cluster stops
    giving much benefit. Silhouette score is a second opinion that peaks at the 'best' K.
    Using both together gives you a more defensible answer.
    """
    print("\nRunning elbow analysis (K-Means for k=2 to k=10)...")
    inertias = []
    silhouette_scores = []
    k_values = range(KMEANS_K_MIN, KMEANS_K_MAX + 1)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init='auto')
        labels = km.fit_predict(X_scaled)

        inertias.append(km.inertia_)
        # Silhouette score needs at least 2 clusters; ranges from -1 (bad) to +1 (perfect)
        sil = silhouette_score(X_scaled, labels, sample_size=5000, random_state=RANDOM_SEED)
        silhouette_scores.append(sil)
        print(f"  k={k}: inertia={km.inertia_:,.0f}, silhouette={sil:.4f}")

    # Plot both metrics side-by-side and save to the figures directory
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(list(k_values), inertias, 'o-', color='steelblue')
    ax1.set_title("Elbow Method (Inertia vs K)")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia (lower is better)")
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(list(k_values), silhouette_scores, 'o-', color='coral')
    ax2.set_title("Silhouette Score vs K")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score (higher is better)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = FIGURES_DIR / "elbow_and_silhouette.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nElbow plot saved to {save_path}")
    print(f"REVIEW the plot and update KMEANS_K_FINAL in config.py if needed (current value: {KMEANS_K_FINAL})")

    return inertias, silhouette_scores


def run_kmeans(X_scaled, k=KMEANS_K_FINAL):
    """
    Fits the final K-Means model with the chosen K and evaluates it.

    HOW K-MEANS WORKS (in plain English):
    1. Place K random 'centroids' (imaginary center points) in the feature space.
    2. Assign every user to the nearest centroid (using Euclidean distance).
    3. Move each centroid to the mean of all points assigned to it.
    4. Repeat steps 2-3 until centroids stop moving.
    The result is K groups of users who are behaviorally similar to each other
    and different from users in other groups.
    """
    print(f"\nFitting final K-Means with k={k}...")
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init='auto')
    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    print(f"  Silhouette Score : {sil:.4f}  (higher is better, max=1.0)")
    print(f"  Davies-Bouldin   : {db:.4f}  (lower is better, min=0.0)")

    return labels, km


def run_dbscan(X_scaled):
    """
    Fits a DBSCAN model as a comparison to K-Means.

    HOW DBSCAN WORKS (in plain English):
    Unlike K-Means which forces every point into a cluster, DBSCAN discovers
    clusters based on density. It asks: 'Is this point surrounded by enough
    neighbors within a radius of eps?' Points with enough neighbors are 'core'
    points. Points reachable from a core point join its cluster. Points that
    are not reachable from ANY core point are labeled -1 (NOISE / outlier).

    WHY DBSCAN MATTERS:
    K-Means assumes clusters are spherical and equally sized.
    DBSCAN handles arbitrary shapes and explicitly flags outliers (bots,
    fraudulent users, extreme power-users) with the -1 label.
    """
    print(f"\nFitting DBSCAN (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    labels = db.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  DBSCAN found {n_clusters} clusters and {n_noise} outlier points (label=-1)")
    print(f"  Outlier rate: {n_noise / len(labels) * 100:.1f}%")

    return labels


def print_cluster_profiles(df, labels, k):
    """
    Prints the mean of key raw features per cluster so you can give each cluster
    a human-readable name. This is 'business validation' — if the numbers make
    sense behaviorally, the clustering is working.
    """
    print(f"\n{'='*60}")
    print("CLUSTER PROFILES (mean of key features per cluster)")
    print(f"{'='*60}")

    df = df.copy()
    df['cluster'] = labels

    profile_cols = [
        'cluster', 'n_views', 'n_addtocart', 'n_transactions',
        'n_sessions', 'cart_to_view_ratio', 'purchase_to_cart_ratio',
        'max_funnel_stage', 'recency_days', 'active_span_days'
    ]

    profile = df.groupby('cluster')[profile_cols[1:]].mean().round(3)
    profile['size'] = df.groupby('cluster').size()
    print(profile.to_string())
    print(f"\nNote: cluster -1 (if present) = DBSCAN noise/outliers")


def build_segments(input_filepath=USER_FEATURES_FILE, output_filepath=USER_SEGMENTS_FILE):
    """
    Main orchestrator for Phase 4.
    Loads features → scales → elbow analysis → fit K-Means → fit DBSCAN → save segments.
    """
    # Step 1: Load and scale
    df, X_scaled, scaler = load_and_scale_features(input_filepath)

    # Step 2: Elbow analysis to help pick K
    run_elbow_analysis(X_scaled)

    # Step 3: Fit final K-Means with chosen K
    kmeans_labels, km_model = run_kmeans(X_scaled, k=KMEANS_K_FINAL)

    # Step 4: Fit DBSCAN for comparison
    dbscan_labels = run_dbscan(X_scaled)

    # Step 5: Print human-readable profiles so you can name the clusters
    print_cluster_profiles(df, kmeans_labels, k=KMEANS_K_FINAL)

    # Step 6: Attach both sets of labels to the user dataframe and save
    df['kmeans_cluster'] = kmeans_labels
    df['dbscan_cluster'] = dbscan_labels

    # Map cluster numbers to human-readable segment names (defined at the top of this file)
    df['segment_name'] = df['kmeans_cluster'].map(SEGMENT_NAMES).fillna("Unknown")

    print(f"\nSaving user segments to {output_filepath}...")
    df.to_parquet(output_filepath, index=False)
    print("Clustering complete!")

    return df


if __name__ == "__main__":
    build_segments()
