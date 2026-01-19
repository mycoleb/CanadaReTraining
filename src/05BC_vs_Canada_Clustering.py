"""
Compare BC vs Rest of Canada using separate clustering analyses.
Shows if program outcome patterns differ between regions.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


def load_and_filter_data(region: str = "BC") -> pd.DataFrame:
    """Load data and filter for specific region."""
    csv_path = DATA_DIR / "37100115.csv"
    
    print(f"\nLoading data for {region}...")
    
    chunks = []
    chunk_size = 100000
    
    # Columns we need
    needed_cols = ['REF_DATE', 'GEO', 'VALUE', 'Educational qualification', 
                   'Field of study', 'Characteristics after graduation', 'Gender']
    
    chunk_num = 0
    for chunk in pd.read_csv(csv_path, encoding='utf-8-sig', chunksize=chunk_size, 
                             usecols=needed_cols, low_memory=False):
        chunk_num += 1
        
        # Filter by region
        if region == "BC":
            filtered = chunk[chunk['GEO'].str.contains('British Columbia', case=False, na=False)].copy()
        else:  # Rest of Canada
            filtered = chunk[
                ~chunk['GEO'].str.contains('British Columbia', case=False, na=False) &
                ~chunk['GEO'].str.contains('^Canada$', case=False, na=False, regex=True)
            ].copy()
        
        if len(filtered) > 0:
            chunks.append(filtered)
            print(f"  Chunk {chunk_num}: {len(filtered):,} rows", end='\r')
        
        if chunk_num >= 200:
            break
    
    print(f"\n  Combining chunks...")
    df = pd.concat(chunks, ignore_index=True)
    print(f"  ✓ Loaded: {len(df):,} rows")
    
    return df


def cluster_region_data(df: pd.DataFrame, region_name: str, k: int = 3) -> tuple:
    """Cluster the data and return results."""
    
    print(f"\n{'='*80}")
    print(f"CLUSTERING: {region_name}")
    print(f"{'='*80}\n")
    
    # Filter to quality outcomes
    outcome_col = 'Characteristics after graduation'
    
    quality_keywords = ['income', 'earnings', 'wage', 'employment']
    count_keywords = ['number of', 'count']
    
    outcome_types = df[outcome_col].unique()
    quality_outcomes = []
    
    for outcome_type in outcome_types:
        outcome_str = str(outcome_type).lower()
        if any(kw in outcome_str for kw in count_keywords):
            continue
        if any(kw in outcome_str for kw in quality_keywords):
            quality_outcomes.append(outcome_type)
    
    print(f"Quality outcome metrics: {len(quality_outcomes)}")
    
    df_quality = df[df[outcome_col].isin(quality_outcomes)].copy()
    
    # Use most recent year
    recent_year = df_quality['REF_DATE'].max()
    print(f"Using year: {recent_year}")
    df_recent = df_quality[df_quality['REF_DATE'] == recent_year].copy()
    print(f"Records: {len(df_recent):,}")
    
    # Create program identifiers
    group_cols = ['Educational qualification', 'Field of study', 'Gender']
    df_recent['program_key'] = df_recent[group_cols].astype(str).agg(' | '.join, axis=1)
    
    # Pivot to wide format
    print("Pivoting data...")
    pivot = df_recent.pivot_table(
        index='program_key',
        columns=outcome_col,
        values='VALUE',
        aggfunc='mean'
    )
    
    print(f"Pivot shape: {pivot.shape[0]} programs × {pivot.shape[1]} outcomes")
    
    # Clean data
    min_cols = max(2, int(pivot.shape[1] * 0.5))
    pivot_clean = pivot.dropna(thresh=min_cols).copy()
    
    for col in pivot_clean.columns:
        pivot_clean[col].fillna(pivot_clean[col].median(), inplace=True)
    
    print(f"After cleaning: {pivot_clean.shape[0]} programs")
    
    if pivot_clean.shape[0] < 10:
        print("ERROR: Too few programs")
        return None, None, None, None
    
    # Standardize
    num_cols = list(pivot_clean.columns)
    X_raw = pivot_clean[num_cols].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X_raw)
    
    # Cluster
    print(f"Clustering with k={k}...")
    if X.shape[0] > 10000:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000, n_init=10)
    else:
        km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    
    labels = km.fit_predict(X)
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    
    return Z, labels, pivot_clean, km


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("BC vs REST OF CANADA - CLUSTERING COMPARISON")
    print("="*80)
    
    # Load data for both regions
    bc_data = load_and_filter_data("BC")
    other_data = load_and_filter_data("Rest of Canada")
    
    # Cluster both
    k = 3  # Same k for fair comparison
    
    bc_Z, bc_labels, bc_pivot, bc_km = cluster_region_data(bc_data, "British Columbia", k=k)
    other_Z, other_labels, other_pivot, other_km = cluster_region_data(other_data, "Rest of Canada", k=k)
    
    if bc_Z is None or other_Z is None:
        print("ERROR: Clustering failed")
        return 1
    
    # Create comparison visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    # Side-by-side PCA plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # BC plot
    scatter1 = ax1.scatter(bc_Z[:, 0], bc_Z[:, 1], c=bc_labels, cmap='tab10', alpha=0.6, s=20)
    ax1.set_title(f'British Columbia\n{len(bc_Z):,} programs, k={k} clusters', fontsize=12, fontweight='bold')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Cluster')
    
    # Rest of Canada plot
    scatter2 = ax2.scatter(other_Z[:, 0], other_Z[:, 1], c=other_labels, cmap='tab10', alpha=0.6, s=20)
    ax2.set_title(f'Rest of Canada\n{len(other_Z):,} programs, k={k} clusters', fontsize=12, fontweight='bold')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Cluster')
    
    plt.suptitle('Program Outcome Clusters: BC vs Rest of Canada', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bc_vs_canada_clusters_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Side-by-side cluster comparison")
    
    # Cluster profiles comparison
    bc_prof = (
        pd.DataFrame(StandardScaler().fit_transform(bc_pivot), columns=bc_pivot.columns)
        .assign(cluster=bc_labels)
        .groupby('cluster')
        .mean()
    )
    
    other_prof = (
        pd.DataFrame(StandardScaler().fit_transform(other_pivot), columns=other_pivot.columns)
        .assign(cluster=other_labels)
        .groupby('cluster')
        .mean()
    )
    
    # Plot cluster profiles
    for cl in range(k):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        if cl in bc_prof.index:
            bc_prof.loc[cl].sort_values().plot(kind='barh', ax=ax1, color='#1f77b4', alpha=0.7)
            ax1.set_title(f'BC - Cluster {cl}', fontweight='bold')
            ax1.set_xlabel('Mean (z-score)')
            ax1.grid(axis='x', alpha=0.3)
        
        if cl in other_prof.index:
            other_prof.loc[cl].sort_values().plot(kind='barh', ax=ax2, color='#ff7f0e', alpha=0.7)
            ax2.set_title(f'Rest of Canada - Cluster {cl}', fontweight='bold')
            ax2.set_xlabel('Mean (z-score)')
            ax2.grid(axis='x', alpha=0.3)
        
        plt.suptitle(f'Cluster {cl} Profile Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"cluster_{cl}_bc_vs_canada_profile.png", dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"  ✓ Cluster profile comparisons (k={k})")
    
    # Cluster size comparison
    bc_sizes = pd.Series(bc_labels).value_counts().sort_index()
    other_sizes = pd.Series(other_labels).value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(k)
    width = 0.35
    
    ax.bar(x - width/2, bc_sizes.values, width, label='BC', alpha=0.7, color='#1f77b4')
    ax.bar(x + width/2, other_sizes.values, width, label='Rest of Canada', alpha=0.7, color='#ff7f0e')
    
    ax.set_xlabel('Cluster', fontweight='bold')
    ax.set_ylabel('Number of Programs', fontweight='bold')
    ax.set_title('Cluster Sizes: BC vs Rest of Canada', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Cluster {i}' for i in range(k)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bc_vs_canada_cluster_sizes.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Cluster size comparison")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    print("CLUSTER SIZES:")
    print("-" * 40)
    print(f"{'Cluster':<10} {'BC':>12} {'Rest':>12}")
    print("-" * 40)
    for i in range(k):
        bc_count = bc_sizes.get(i, 0)
        other_count = other_sizes.get(i, 0)
        print(f"Cluster {i:<3} {bc_count:>12,} {other_count:>12,}")
    
    print(f"\n{'='*80}")
    print("✓ ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("\nOutput files:")
    print(f"  • bc_vs_canada_clusters_comparison.png")
    print(f"  • cluster_X_bc_vs_canada_profile.png (for each cluster)")
    print(f"  • bc_vs_canada_cluster_sizes.png")
    print(f"\n{'='*80}\n")
    
    print("INTERPRETATION:")
    print("  - Compare cluster positions: Do programs cluster similarly in both regions?")
    print("  - Check cluster profiles: Do similar clusters have same characteristics?")
    print("  - Look at sizes: Are program distributions similar?")
    print("  - Different patterns suggest regional variations in program outcomes")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())