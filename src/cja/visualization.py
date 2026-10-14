"""
visualization.py — Phase 6: Visualization & Storytelling

PURPOSE:
    Converts the raw numbers from Phases 3, 4, and 5 into charts
    that tell a clear visual story about customer behavior.

WHY VISUALIZATION MATTERS:
    Numbers in a parquet file are invisible to stakeholders. A well-designed
    chart communicates a finding in 5 seconds that a table would take 5 minutes
    to decode. Visualization is also how you validate clustering visually —
    if clusters don't separate on the PCA scatter, something is wrong.

CHARTS PRODUCED:
    1. PCA Scatter Plot         — 2D visual of cluster separation
    2. Segment Size Bar Chart   — relative size of each segment
    3. Funnel Drop-off Chart    — where each segment abandons the journey
    4. Feature Radar/Box Chart  — behavioral profile of each segment
    5. Conversion Rate Chart    — final purchase rate per segment

All figures saved as PNGs to data/processed/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cja.config import (
    USER_SEGMENTS_FILE,
    SEGMENT_FUNNELS_FILE,
    USER_FEATURES_FILE,
    FIGURES_DIR,
    CLUSTERING_FEATURES,
    RANDOM_SEED,
)

# ── Color palette: one distinct color per segment ──────────────────────────────
# We define colors here (not inside each chart) so every chart uses the
# same color for the same segment — a key principle of data storytelling.
SEGMENT_COLORS = {
    "Casual Visitors":       "#6c757d",   # muted grey — low engagement
    "Active Browsers":       "#4a90d9",   # cool blue — browsing
    "Cart Abandoners":       "#e8a838",   # amber — mid-funnel warning
    "Loyal Window Shoppers": "#9b59b6",   # purple — returning but stuck
    "Converters":            "#27ae60",   # green — success
}


def _get_color_list(segment_names):
    """Map a list of segment names to their colors. Fallback to grey if unknown."""
    return [SEGMENT_COLORS.get(name, "#aaaaaa") for name in segment_names]


# ── Chart 1: PCA Scatter Plot ─────────────────────────────────────────────────

def plot_pca_scatter(df: pd.DataFrame, save: bool = True) -> None:
    """
    Reduces the 13 clustering features to 2 dimensions using PCA and
    plots each user as a dot coloured by their segment.

    WHY PCA?
    We have 13 features — impossible to plot directly. PCA (Principal
    Component Analysis) finds the 2 directions in 13-dimensional space
    that capture the most variance (spread) in the data. We project every
    user onto those 2 directions, giving us 2 numbers we can plot on X and Y.

    IMPORTANT CAVEAT:
    PC1 and PC2 are *linear combinations* of all 13 original features —
    not any single feature. You cannot say "X-axis = recency". This is
    why we only use PCA for plotting, never for feature engineering.
    """
    print("Generating PCA scatter plot...")

    # Scale the features before PCA — PCA is variance-sensitive,
    # so unscaled features would let large-range features dominate the components
    X = df[CLUSTERING_FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)

    # Fit PCA: n_components=2 means we want exactly 2 output dimensions
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X_scaled)

    # Report how much variance is explained (higher = more informative plot)
    var_explained = pca.explained_variance_ratio_ * 100
    print(f"  PCA: PC1={var_explained[0]:.1f}%, PC2={var_explained[1]:.1f}% variance explained")

    # Build a plotting dataframe
    plot_df = pd.DataFrame({
        'PC1': coords[:, 0],
        'PC2': coords[:, 1],
        'segment': df['segment_name'].values,
    })

    fig, ax = plt.subplots(figsize=(11, 7))

    for segment, group in plot_df.groupby('segment'):
        color = SEGMENT_COLORS.get(segment, "#aaaaaa")
        ax.scatter(
            group['PC1'], group['PC2'],
            label=segment,
            color=color,
            alpha=0.35,      # transparency so we can see overlapping points
            s=15,            # point size
            linewidths=0,
        )

    ax.set_title(
        "Customer Segments — PCA Projection\n"
        f"(PC1={var_explained[0]:.1f}% + PC2={var_explained[1]:.1f}% = "
        f"{sum(var_explained):.1f}% of variance explained)",
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel("Principal Component 1 (PC1)", fontsize=11)
    ax.set_ylabel("Principal Component 2 (PC2)", fontsize=11)
    ax.legend(title="Segment", fontsize=9, title_fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "pca_scatter.png"
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close()


# ── Chart 2: Segment Size Bar Chart ──────────────────────────────────────────

def plot_segment_sizes(df: pd.DataFrame, save: bool = True) -> None:
    """
    Simple horizontal bar chart showing the number of users per segment.

    WHY THIS CHART:
    Cluster size is often invisible in scatter plots. This chart immediately
    communicates that Casual Visitors dominate (72%) while Converters are rare
    (2.5%) — a critical business context for interpreting every other chart.
    """
    print("Generating segment size chart...")

    size_df = (
        df.groupby('segment_name')
        .size()
        .reset_index(name='n_users')
        .sort_values('n_users', ascending=True)  # ascending so largest is at the top
    )
    total = size_df['n_users'].sum()
    size_df['pct'] = (size_df['n_users'] / total * 100).round(1)

    colors = _get_color_list(size_df['segment_name'].tolist())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(size_df['segment_name'], size_df['n_users'], color=colors, edgecolor='white')

    # Annotate each bar with count and percentage
    for bar, (_, row) in zip(bars, size_df.iterrows()):
        ax.text(
            bar.get_width() + 80,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['n_users']):,}  ({row['pct']}%)",
            va='center', fontsize=10,
        )

    ax.set_title("Users per Segment", fontsize=13, fontweight='bold')
    ax.set_xlabel("Number of Users", fontsize=11)
    ax.set_xlim(0, size_df['n_users'].max() * 1.25)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "segment_sizes.png"
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close()


# ── Chart 3: Per-Segment Funnel Drop-off ─────────────────────────────────────

def plot_funnel_dropoff(funnel_df: pd.DataFrame, save: bool = True) -> None:
    """
    Grouped bar chart showing view%, cart%, and conversion% side by side
    for each segment.

    WHY THIS CHART:
    This is the central business insight chart. It visually encodes WHERE
    each segment is stuck. Cart Abandoners' bar drops sharply at 'purchase'
    while staying at 100% for view and cart — a striking visual that makes
    the checkout problem instantly obvious.
    """
    print("Generating funnel drop-off chart...")

    segments = funnel_df['segment_name'].tolist()
    x = np.arange(len(segments))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 7))

    bars_view = ax.bar(x - width, funnel_df['pct_reached_view'],
                       width, label='Reached View', color='#3498db', alpha=0.85)
    bars_cart = ax.bar(x, funnel_df['pct_reached_cart'],
                       width, label='Reached Cart', color='#e67e22', alpha=0.85)
    bars_conv = ax.bar(x + width, funnel_df['pct_converted'],
                       width, label='Converted (Purchased)', color='#27ae60', alpha=0.85)

    # Add value labels on top of each bar
    for bars in [bars_view, bars_cart, bars_conv]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:   # only label if bar is visible
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 1,
                    f"{height:.1f}%",
                    ha='center', va='bottom', fontsize=8, fontweight='bold'
                )

    ax.set_title("Funnel Conversion by Segment", fontsize=13, fontweight='bold')
    ax.set_ylabel("% of Users Reaching Stage", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "funnel_dropoff.png"
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close()


# ── Chart 4: Feature Distribution by Segment (Box Plots) ─────────────────────

def plot_feature_distributions(df: pd.DataFrame, save: bool = True) -> None:
    """
    Box plots for 4 key behavioral features, split by segment.

    WHY BOX PLOTS:
    They show the median, spread (IQR), and outliers for each segment in one
    compact chart. This lets you see not just the mean (which we showed in
    the cluster profile table) but also how spread out each segment is —
    are they tight and homogeneous, or spread and varied?
    """
    print("Generating feature distribution box plots...")

    # Choose 4 features that tell the clearest story
    features_to_plot = {
        'n_views': 'Total Views',
        'n_addtocart': 'Total Add-to-Carts',
        'cart_to_view_ratio': 'Cart-to-View Ratio',
        'active_span_days': 'Active Span (Days)',
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    segment_order = list(SEGMENT_COLORS.keys())
    palette = {k: v for k, v in SEGMENT_COLORS.items() if k in df['segment_name'].unique()}

    for ax, (col, label) in zip(axes, features_to_plot.items()):
        # Use seaborn boxplot for easy grouping
        present_order = [s for s in segment_order if s in df['segment_name'].unique()]
        sns.boxplot(
            data=df,
            x='segment_name',
            y=col,
            hue='segment_name',
            order=present_order,
            palette=palette,
            legend=False,
            ax=ax,
            fliersize=2,
            linewidth=0.8,
        )
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks(range(len(present_order)))
        ax.set_xticklabels(present_order, rotation=20, ha='right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.suptitle("Feature Distributions by Segment", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "feature_distributions.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# ── Chart 5: Conversion Rate Bar Chart ───────────────────────────────────────

def plot_conversion_rates(funnel_df: pd.DataFrame, save: bool = True) -> None:
    """
    Simple vertical bar chart of final conversion rates per segment.
    This is the 'punchline' chart — the direct answer to the business question
    'which segment is actually buying?'
    """
    print("Generating conversion rate chart...")

    funnel_sorted = funnel_df.sort_values('pct_converted', ascending=False)
    colors = _get_color_list(funnel_sorted['segment_name'].tolist())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        funnel_sorted['segment_name'],
        funnel_sorted['pct_converted'],
        color=colors, edgecolor='white', width=0.55
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.1f}%",
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.set_title("Conversion Rate by Segment\n(% of users who made a purchase)",
                 fontsize=13, fontweight='bold')
    ax.set_ylabel("Conversion Rate (%)", fontsize=11)
    ax.set_ylim(0, funnel_sorted['pct_converted'].max() * 1.25)
    ax.set_xticklabels(funnel_sorted['segment_name'], rotation=15, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "conversion_rates.png"
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close()


# ── Orchestrator ──────────────────────────────────────────────────────────────

def build_all_figures(
    segments_filepath=USER_SEGMENTS_FILE,
    funnels_filepath=SEGMENT_FUNNELS_FILE,
):
    """
    Main entry point. Loads required data and generates all 5 charts.
    """
    if not segments_filepath.exists():
        raise FileNotFoundError(f"Run Phase 4 first: {segments_filepath}")
    if not funnels_filepath.exists():
        raise FileNotFoundError(f"Run Phase 5 first: {funnels_filepath}")

    print(f"Loading data...")
    df = pd.read_parquet(segments_filepath)
    funnel_df = pd.read_parquet(funnels_filepath)

    print(f"\nGenerating all figures into {FIGURES_DIR}...\n")
    plot_pca_scatter(df)
    plot_segment_sizes(df)
    plot_funnel_dropoff(funnel_df)
    plot_feature_distributions(df)
    plot_conversion_rates(funnel_df)

    print("\nAll figures generated successfully!")
    print(f"Open the folder: {FIGURES_DIR}")


if __name__ == "__main__":
    build_all_figures()
