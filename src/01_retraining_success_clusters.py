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
    
    For datasets > 10k rows, we sample to speed up silhouette calculation.
    """
    n_samples = X.shape[0]
    
    # If dataset is large, sample for silhouette calculation
    if n_samples > 10000:
        print(f"  Large dataset ({n_samples} rows) - sampling {sample_size} rows for k selection")
        indices = np.random.choice(n_samples, size=min(sample_size, n_samples), replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    best_k, best_s = None, -1
    for k in range(k_min, k_max + 1):
        # Use MiniBatchKMeans for speed on large datasets
        if n_samples > 10000:
            km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000, n_init=3)
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
        
        labels = km.fit_predict(X_sample)
        
        # Silhouette on sample only
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
        print("This should be the CSV extracted from the StatsCan download ZIP.")
        print("\nDownload from:")
        print("https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3710011501")
        return 1
    
    print(f"Loading data from {csv_path}...")
    print("(This may take a minute for large files...)")
    
    # Read with UTF-8 BOM handling (StatsCan files often have BOM)
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    print(f"Loaded data: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # 2) Normalize column names
    df.columns = [re.sub(r"\s+", " ", c.strip()) for c in df.columns]

    # StatsCan tables have a standard structure
    if "VALUE" not in df.columns:
        print("\nERROR: Expected 'VALUE' column not found!")
        print("Available columns:", list(df.columns))
        return 1
    
    # Identify dimension columns (exclude standard StatsCan metadata)
    metadata_cols = {
        'REF_DATE', 'GEO', 'DGUID', 'VALUE', 'STATUS', 'SYMBOL', 
        'TERMINATED', 'DECIMALS', 'UOM', 'UOM_ID', 'SCALAR_FACTOR', 
        'SCALAR_ID', 'VECTOR', 'COORDINATE'
    }
    
    dim_cols = [c for c in df.columns if c not in metadata_cols]
    
    print(f"\nFound {len(dim_cols)} dimension columns")
    
    # 3) Find the outcome dimension
    outcome_dim = None
    for col in dim_cols:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) > 3 and any(
            keyword in str(val).lower() 
            for val in unique_vals[:10] 
            for keyword in ['rate', 'employment', 'earnings', 'wage', 'income', 'outcome']
        ):
            outcome_dim = col
            print(f"Identified outcome dimension: {outcome_dim}")
            print(f"  Unique outcomes: {len(unique_vals)}")
            break
    
    if not outcome_dim:
        print("\nWARNING: Could not auto-identify outcome dimension.")
        outcome_dim = dim_cols[-1] if dim_cols else None
    
    if not outcome_dim:
        print("\nERROR: No suitable dimension for pivoting found!")
        return 1
    
    # 4) Reduce data size BEFORE pivoting (filter to most recent year and Canada-level)
    print("\nApplying filters to reduce data size...")
    
    work = df.copy()
    
    # Filter 1: Keep only most recent year(s)
    if 'REF_DATE' in work.columns:
        # Get unique years and keep last 3 years
        years = sorted(work['REF_DATE'].dropna().unique())
        recent_years = years[-3:] if len(years) > 3 else years
        work = work[work['REF_DATE'].isin(recent_years)]
        print(f"  Filtered to years: {recent_years} ({len(work):,} rows)")
    
    # Filter 2: Keep only Canada-level data (not provinces) for speed
    if 'GEO' in work.columns:
        canada_geos = work['GEO'].unique()
        # Keep only "Canada" geography
        canada_only = [g for g in canada_geos if 'canada' in str(g).lower() and len(str(g)) < 20]
        if canada_only:
            work = work[work['GEO'].isin(canada_only)]
            print(f"  Filtered to Canada-level geography ({len(work):,} rows)")
    
    # Filter 3: Drop rows with missing values
    work = work.dropna(subset=['VALUE'])
    print(f"  After dropping NaN: {len(work):,} rows")
    
    # 5) Group dimensions for pivoting
    group_dims = [c for c in dim_cols if c != outcome_dim]
    group_dims = ['REF_DATE', 'GEO'] + group_dims
    
    print(f"\nCreating program identifiers...")
    work['program_key'] = work[group_dims].astype(str).agg(' | '.join, axis=1)
    
    # 6) Pivot to wide format
    print(f"Pivoting data (this may take a moment)...")
    pivot = work.pivot_table(
        index='program_key',
        columns=outcome_dim,
        values='VALUE',
        aggfunc='mean'
    )
    
    print(f"Pivoted data: {pivot.shape[0]:,} programs × {pivot.shape[1]} outcomes")
    
    # 7) Clean the pivoted data
    # Keep rows with at least 50% of outcomes available
    min_cols_required = max(2, int(pivot.shape[1] * 0.5))
    pivot_clean = pivot.dropna(thresh=min_cols_required).copy()
    
    print(f"After removing sparse rows: {pivot_clean.shape[0]:,} programs")
    
    if pivot_clean.shape[0] < 10:
        print("\nERROR: Too few valid rows after cleaning.")
        return 1
    
    # If still too many rows, sample for analysis
    MAX_ROWS = 20000
    if pivot_clean.shape[0] > MAX_ROWS:
        print(f"\nDataset still large ({pivot_clean.shape[0]:,} rows)")
        print(f"Sampling {MAX_ROWS:,} random programs for analysis...")
        pivot_clean = pivot_clean.sample(n=MAX_ROWS, random_state=42)
    
    # Fill remaining NaN with median
    for col in pivot_clean.columns:
        pivot_clean[col].fillna(pivot_clean[col].median(), inplace=True)
    
    # 8) Feature matrix
    print(f"\nPreparing feature matrix...")
    num_cols = list(pivot_clean.columns)
    X_raw = pivot_clean[num_cols].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X_raw)
    
    print(f"Final matrix size: {X.shape[0]:,} rows × {X.shape[1]} features")

    # 9) Choose k + cluster (with fast method)
    print("\nSelecting optimal k via silhouette score (fast method)...")
    k = pick_k_by_silhouette_fast(X, k_min=2, k_max=8, sample_size=5000)
    print(f"\nBest k={k}")
    
    # Use MiniBatchKMeans for final clustering if dataset is large
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

    # Plot A: PCA scatter
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=cluster_labels, cmap="tab10", alpha=0.6, s=20)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"Graduate Outcome Clusters (k={k}) - PCA view\n{X.shape[0]:,} programs")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.tight_layout()
    p1 = OUT_DIR / "retraining_clusters_pca.png"
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

    # Plot B: bar profile per cluster
    for cl in prof.index:
        plt.figure(figsize=(12, 6))
        prof.loc[cl].sort_values().plot(kind="barh")
        plt.title(f"Cluster {cl} - Standardized Outcome Profile")
        plt.xlabel("Mean (z-score)")
        plt.ylabel("Outcome metric")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"retraining_cluster_{cl}_profile.png", dpi=200)
        plt.close()

    # 12) Cluster exemplars
    centers = km.cluster_centers_
    closest = {}
    for cl in range(k):
        d = np.linalg.norm(X - centers[cl], axis=1)
        idx = np.argsort(d)[:10]  # Get top 10 instead of 5
        exemplar_keys = pivot_clean.index[idx].tolist()
        closest[cl] = pd.DataFrame({
            'cluster': cl,
            'program_key': exemplar_keys,
            'distance_to_center': d[idx]
        })
        # Add the actual outcome values
        for i, col_name in enumerate(num_cols):
            closest[cl][col_name] = pivot_clean.iloc[idx][col_name].values

    exemplars = pd.concat(list(closest.values()), ignore_index=True)
    exemplars.to_csv(OUT_DIR / "retraining_cluster_exemplars.csv", index=False)

    # Summary statistics
    summary = pd.DataFrame({
        'cluster': range(k),
        'count': [sum(cluster_labels == i) for i in range(k)],
        'percentage': [100 * sum(cluster_labels == i) / len(cluster_labels) for i in range(k)]
    })
    summary.to_csv(OUT_DIR / "retraining_cluster_summary.csv", index=False)

    print(f"\n{'='*70}")
    print("SUCCESS! Output files created:")
    print(f"  - {p1}")
    print(f"  - {OUT_DIR}/retraining_cluster_*_profile.png")
    print(f"  - {OUT_DIR}/retraining_cluster_exemplars.csv")
    print(f"  - {OUT_DIR}/retraining_cluster_summary.csv")
    print(f"\n{'='*70}")
    print("\nCluster Summary:")
    print(summary.to_string(index=False))
    print(f"\n{'='*70}")
    print("\nHow to interpret clusters:")
    print("1. Open retraining_cluster_exemplars.csv")
    print("2. Look at the top programs for each cluster")
    print("3. Look at the profile plots to see outcome patterns")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())