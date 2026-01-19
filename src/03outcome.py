from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


def pick_k_by_silhouette_fast(X: np.ndarray, k_min: int = 2, k_max: int = 8, 
                               sample_size: int = 5000) -> int:
    """
    Fast k selection using sampling for large datasets.
    """
    n_samples = X.shape[0]
    
    if n_samples > 10000:
        print(f"  Large dataset ({n_samples} rows) - sampling {sample_size} rows for k selection")
        indices = np.random.choice(n_samples, size=min(sample_size, n_samples), replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    best_k, best_s = None, -1
    for k in range(k_min, k_max + 1):
        if n_samples > 10000:
            km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000, n_init=3)
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
        
        labels = km.fit_predict(X_sample)
        s = silhouette_score(X_sample, labels, sample_size=min(2000, len(X_sample)))
        print(f"  k={k}: silhouette={s:.3f}")
        
        if s > best_s:
            best_k, best_s = k, s
    
    return int(best_k)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load the StatsCan table 37-10-0115-01 data
    csv_path = DATA_DIR / "37100115.csv"
    
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        print("\nExpected file: data/37100115.csv")
        return 1
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    print(f"Loaded data: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # 2) Normalize column names
    df.columns = [re.sub(r"\s+", " ", c.strip()) for c in df.columns]

    # 3) Identify the outcome dimension
    # Look for a column that contains different types of statistics/outcomes
    metadata_cols = {
        'REF_DATE', 'GEO', 'DGUID', 'VALUE', 'STATUS', 'SYMBOL', 
        'TERMINATED', 'DECIMALS', 'UOM', 'UOM_ID', 'SCALAR_FACTOR', 
        'SCALAR_ID', 'VECTOR', 'COORDINATE'
    }
    
    dim_cols = [c for c in df.columns if c not in metadata_cols]
    
    print(f"\nDimension columns: {dim_cols}")
    
    # Find the column that contains outcome types
    outcome_col = None
    for col in dim_cols:
        unique_vals = df[col].dropna().unique()
        sample_vals = [str(v).lower() for v in unique_vals[:20]]
        
        # Look for income/employment/outcome keywords
        if any(keyword in ' '.join(sample_vals) 
               for keyword in ['income', 'employment', 'rate', 'wage', 'earnings', 'median']):
            outcome_col = col
            print(f"\nFound outcome column: '{outcome_col}'")
            print(f"Unique outcome types: {len(unique_vals)}")
            print(f"\nSample outcome types:")
            for i, val in enumerate(unique_vals[:15], 1):
                print(f"  {i}. {val}")
            break
    
    if not outcome_col:
        print("\nERROR: Could not find outcome column!")
        print("Available dimension columns:")
        for col in dim_cols:
            print(f"  - {col}: {df[col].nunique()} unique values")
        return 1
    
    # 4) CRITICAL FIX: Filter to ONLY outcome quality metrics (not counts)
    print(f"\n{'='*80}")
    print("FILTERING TO OUTCOME QUALITY METRICS ONLY")
    print(f"{'='*80}")
    
    # Identify which rows are actual outcomes vs. counts
    outcome_types = df[outcome_col].unique()
    
    # Keywords that indicate QUALITY outcomes (not just counts)
    quality_keywords = [
        'income', 'earnings', 'wage', 'salary',  # Income metrics
        'employment rate', 'unemployment',        # Employment metrics
        'median', 'average', 'mean',              # Aggregate measures
        'percentage', 'proportion', 'rate'        # Rates/percentages
    ]
    
    # Keywords that indicate COUNTS (not quality)
    count_keywords = [
        'number of graduates',
        'number of students', 
        'count',
        'total number'
    ]
    
    # Classify each outcome type
    quality_outcomes = []
    count_outcomes = []
    
    for outcome_type in outcome_types:
        outcome_str = str(outcome_type).lower()
        
        # Check if it's a count
        if any(kw in outcome_str for kw in count_keywords):
            count_outcomes.append(outcome_type)
        # Check if it's a quality metric
        elif any(kw in outcome_str for kw in quality_keywords):
            quality_outcomes.append(outcome_type)
        else:
            # If unclear, be conservative and exclude
            count_outcomes.append(outcome_type)
    
    print(f"\nQUALITY outcome metrics ({len(quality_outcomes)}):")
    for outcome in quality_outcomes:
        print(f"  ✓ {outcome}")
    
    print(f"\nCOUNT metrics ({len(count_outcomes)}) - EXCLUDING these:")
    for outcome in count_outcomes[:10]:
        print(f"  ✗ {outcome}")
    if len(count_outcomes) > 10:
        print(f"  ... and {len(count_outcomes) - 10} more")
    
    if not quality_outcomes:
        print("\nERROR: No quality outcome metrics found!")
        return 1
    
    # Filter dataframe to only quality outcomes
    print(f"\nFiltering data...")
    print(f"Before: {len(df):,} rows")
    
    df_quality = df[df[outcome_col].isin(quality_outcomes)].copy()
    
    print(f"After:  {len(df_quality):,} rows (kept {100*len(df_quality)/len(df):.1f}%)")
    
    if len(df_quality) < 100:
        print("\nERROR: Too few rows after filtering!")
        return 1
    
    # 5) Apply additional filters to reduce size
    print(f"\n{'='*80}")
    print("APPLYING ADDITIONAL FILTERS")
    print(f"{'='*80}")
    
    work = df_quality.copy()
    
    # Filter to recent years
    if 'REF_DATE' in work.columns:
        years = sorted(work['REF_DATE'].dropna().unique())
        recent_years = years[-3:] if len(years) > 3 else years
        work = work[work['REF_DATE'].isin(recent_years)]
        print(f"Filtered to years: {recent_years} ({len(work):,} rows)")
    
    # Filter to Canada-level only
    if 'GEO' in work.columns:
        canada_geos = [g for g in work['GEO'].unique() if 'canada' in str(g).lower() and len(str(g)) < 20]
        if canada_geos:
            work = work[work['GEO'].isin(canada_geos)]
            print(f"Filtered to Canada geography ({len(work):,} rows)")
    
    # Drop rows with missing values
    work = work.dropna(subset=['VALUE'])
    print(f"After dropping NaN: {len(work):,} rows")
    
    # 6) Create program identifiers and pivot
    group_dims = [c for c in dim_cols if c != outcome_col]
    group_dims = ['REF_DATE', 'GEO'] + group_dims
    
    print(f"\nGrouping by: {group_dims}")
    work['program_key'] = work[group_dims].astype(str).agg(' | '.join, axis=1)
    
    print(f"\nPivoting data...")
    pivot = work.pivot_table(
        index='program_key',
        columns=outcome_col,
        values='VALUE',
        aggfunc='mean'
    )
    
    print(f"Pivoted: {pivot.shape[0]:,} programs × {pivot.shape[1]} outcome metrics")
    print(f"\nOutcome metrics in pivot:")
    for i, col in enumerate(pivot.columns, 1):
        print(f"  {i}. {col}")
    
    # 7) Clean the pivoted data
    min_cols_required = max(2, int(pivot.shape[1] * 0.5))
    pivot_clean = pivot.dropna(thresh=min_cols_required).copy()
    
    print(f"\nAfter removing sparse rows: {pivot_clean.shape[0]:,} programs")
    
    if pivot_clean.shape[0] < 10:
        print("\nERROR: Too few valid rows after cleaning.")
        return 1
    
    # Sample if still too large
    MAX_ROWS = 20000
    if pivot_clean.shape[0] > MAX_ROWS:
        print(f"Sampling {MAX_ROWS:,} programs for analysis...")
        pivot_clean = pivot_clean.sample(n=MAX_ROWS, random_state=42)
    
    # Fill remaining NaN with median
    for col in pivot_clean.columns:
        pivot_clean[col].fillna(pivot_clean[col].median(), inplace=True)
    
    # 8) Feature matrix
    print(f"\nPreparing feature matrix...")
    num_cols = list(pivot_clean.columns)
    X_raw = pivot_clean[num_cols].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X_raw)
    
    print(f"Final matrix: {X.shape[0]:,} programs × {X.shape[1]} outcomes")
    
    # Show some statistics
    print(f"\nOutcome statistics (raw values):")
    print(pivot_clean.describe().to_string())

    # 9) Choose k + cluster
    print(f"\n{'='*80}")
    print("CLUSTERING PROGRAMS BY OUTCOME QUALITY")
    print(f"{'='*80}")
    print("\nSelecting optimal k via silhouette score...")
    
    k = pick_k_by_silhouette_fast(X, k_min=2, k_max=8, sample_size=5000)
    print(f"\nBest k={k}")
    
    if X.shape[0] > 10000:
        print(f"Using MiniBatchKMeans for large dataset...")
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000, n_init=10)
    else:
        km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    
    print("Fitting final model...")
    cluster_labels = km.fit_predict(X)

    # 10) PCA for visualization
    print("Creating visualizations...")
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=cluster_labels, cmap="tab10", alpha=0.6, s=20)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"Program Outcome Quality Clusters (k={k})\n{X.shape[0]:,} programs - Quality metrics only")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.tight_layout()
    p1 = OUT_DIR / "outcome_quality_clusters_pca.png"
    plt.savefig(p1, dpi=200)
    plt.close()

    # 11) Cluster profiles
    prof = (
        pd.DataFrame(X, columns=num_cols)
        .assign(cluster=cluster_labels)
        .groupby("cluster")[num_cols]
        .mean()
        .sort_index()
    )

    # Also get raw value profiles
    prof_raw = (
        pivot_clean
        .assign(cluster=cluster_labels)
        .groupby("cluster")[num_cols]
        .mean()
        .sort_index()
    )

    for cl in prof.index:
        # Plot A: Standardized (z-score)
        plt.figure(figsize=(12, 6))
        prof.loc[cl].sort_values().plot(kind="barh")
        plt.title(f"Cluster {cl} - Standardized Outcome Profile (z-scores)")
        plt.xlabel("Mean (z-score)")
        plt.ylabel("Outcome metric")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"outcome_cluster_{cl}_profile_standardized.png", dpi=200)
        plt.close()
        
        # Plot B: Raw values
        plt.figure(figsize=(12, 6))
        prof_raw.loc[cl].sort_values().plot(kind="barh")
        plt.title(f"Cluster {cl} - Raw Outcome Profile (actual values)")
        plt.xlabel("Mean value")
        plt.ylabel("Outcome metric")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"outcome_cluster_{cl}_profile_raw.png", dpi=200)
        plt.close()

    # 12) Cluster exemplars with raw values
    centers = km.cluster_centers_
    closest = {}
    for cl in range(k):
        d = np.linalg.norm(X - centers[cl], axis=1)
        idx = np.argsort(d)[:10]
        exemplar_keys = pivot_clean.index[idx].tolist()
        closest[cl] = pd.DataFrame({
            'cluster': cl,
            'program_key': exemplar_keys,
            'distance_to_center': d[idx]
        })
        # Add raw outcome values
        for col_name in num_cols:
            closest[cl][col_name] = pivot_clean.iloc[idx][col_name].values

    exemplars = pd.concat(list(closest.values()), ignore_index=True)
    exemplars.to_csv(OUT_DIR / "outcome_quality_cluster_exemplars.csv", index=False)

    # Summary statistics
    summary = pd.DataFrame({
        'cluster': range(k),
        'count': [sum(cluster_labels == i) for i in range(k)],
        'percentage': [100 * sum(cluster_labels == i) / len(cluster_labels) for i in range(k)]
    })
    
    # Add mean raw values per cluster
    for col in num_cols:
        summary[f'mean_{col}'] = prof_raw[col].values
    
    summary.to_csv(OUT_DIR / "outcome_quality_cluster_summary.csv", index=False)

    print(f"\n{'='*80}")
    print("SUCCESS! Output files created:")
    print(f"{'='*80}")
    print(f"  - {p1}")
    print(f"  - {OUT_DIR}/outcome_cluster_*_profile_standardized.png")
    print(f"  - {OUT_DIR}/outcome_cluster_*_profile_raw.png")
    print(f"  - {OUT_DIR}/outcome_quality_cluster_exemplars.csv")
    print(f"  - {OUT_DIR}/outcome_quality_cluster_summary.csv")
    
    print(f"\n{'='*80}")
    print("CLUSTER SUMMARY:")
    print(f"{'='*80}")
    print(summary.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("INTERPRETATION:")
    print(f"{'='*80}")
    print("""
Now your clusters represent programs grouped by ACTUAL OUTCOME QUALITY:
- High income clusters = Fields with strong earnings
- High employment rate clusters = Fields with good job prospects  
- Low outcome clusters = Fields with weaker labour market outcomes

Next steps:
1. Look at outcome_quality_cluster_exemplars.csv to see which fields are in each cluster
2. Look at the _raw.png plots to see actual dollar amounts and percentages
3. Run the analyze_clusters script to parse the program names
    """)
    print(f"{'='*80}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

