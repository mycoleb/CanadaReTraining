"""
Compare British Columbia graduate outcomes vs Rest of Canada.
Optimized for large files - reads data in chunks.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


def load_graduates_data_optimized() -> pd.DataFrame:
    """Load the graduates outcomes data efficiently using chunking and filtering."""
    csv_path = DATA_DIR / "37100115.csv"
    
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        return None
    
    print(f"Loading large file from {csv_path}...")
    print("This may take a few minutes for a 1.5GB file...")
    
    # Read in chunks and filter as we go
    chunks = []
    chunk_size = 100000
    
    print("\nReading data in chunks:")
    
    # First, quickly scan to get column names
    first_chunk = pd.read_csv(csv_path, encoding='utf-8-sig', nrows=1)
    columns = first_chunk.columns.tolist()
    
    # Identify which columns we actually need
    needed_cols = ['REF_DATE', 'GEO', 'VALUE']
    
    # Find other important columns
    for col in columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['outcome', 'statistics', 'characteristic', 
                                                     'field', 'qualification', 'gender']):
            needed_cols.append(col)
    
    print(f"  Keeping {len(needed_cols)} columns out of {len(columns)}")
    print(f"  Columns: {needed_cols[:5]}...")
    
    # Read file in chunks, keeping only needed columns
    chunk_num = 0
    for chunk in pd.read_csv(csv_path, encoding='utf-8-sig', chunksize=chunk_size, 
                             usecols=needed_cols, low_memory=False):
        chunk_num += 1
        
        # Filter to only BC and other provinces (not Canada totals)
        if 'GEO' in chunk.columns:
            chunk_filtered = chunk[
                (chunk['GEO'].str.contains('British Columbia', case=False, na=False)) |
                (~chunk['GEO'].str.contains('Canada', case=False, na=False))
            ].copy()
            
            if len(chunk_filtered) > 0:
                chunks.append(chunk_filtered)
                print(f"  Chunk {chunk_num}: Kept {len(chunk_filtered):,} rows", end='\r')
        
        # Limit total chunks to prevent memory issues
        if chunk_num >= 200:  # Stop after ~20M rows processed
            print(f"\n  Reached chunk limit, stopping...")
            break
    
    if not chunks:
        print("\nERROR: No data found")
        return None
    
    print(f"\n\nCombining {len(chunks)} chunks...")
    df = pd.concat(chunks, ignore_index=True)
    print(f"✓ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    return df


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("BC vs REST OF CANADA - GRADUATE OUTCOMES COMPARISON")
    print("="*80 + "\n")
    
    # Load data
    df = load_graduates_data_optimized()
    if df is None:
        return 1
    
    # Normalize column names
    df.columns = [re.sub(r"\s+", " ", c.strip()) for c in df.columns]
    
    print(f"\n{'='*80}")
    print("FILTERING DATA")
    print(f"{'='*80}\n")
    
    # Show available geographies
    print("Available geographies:")
    geo_counts = df['GEO'].value_counts()
    for geo, count in list(geo_counts.items())[:15]:
        print(f"  - {geo}: {count:,} records")
    
    # Separate BC and other provinces
    bc_data = df[df['GEO'].str.contains('British Columbia', case=False, na=False)].copy()
    other_provinces = df[~df['GEO'].str.contains('British Columbia', case=False, na=False)].copy()
    
    print(f"\nSplit:")
    print(f"  BC: {len(bc_data):,} records")
    print(f"  Other provinces: {len(other_provinces):,} records")
    
    if len(bc_data) == 0 or len(other_provinces) == 0:
        print("\nERROR: Not enough data")
        return 1
    
    # Find outcome column
    metadata_cols = {'REF_DATE', 'GEO', 'DGUID', 'VALUE', 'STATUS', 'SYMBOL', 
                     'TERMINATED', 'DECIMALS', 'UOM', 'UOM_ID', 'SCALAR_FACTOR', 
                     'SCALAR_ID', 'VECTOR', 'COORDINATE'}
    
    dim_cols = [c for c in df.columns if c not in metadata_cols]
    
    outcome_col = None
    for col in dim_cols:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) > 3:
            sample = [str(v).lower() for v in unique_vals[:20]]
            if any(kw in ' '.join(sample) for kw in ['income', 'employment', 'wage', 'median']):
                outcome_col = col
                print(f"\nOutcome column: '{outcome_col}'")
                break
    
    if not outcome_col:
        outcome_col = dim_cols[-1] if dim_cols else None
        print(f"\nUsing column: '{outcome_col}'")
    
    # Filter to quality outcomes
    print(f"\nFiltering to income/employment metrics...")
    
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
    
    print(f"  Found {len(quality_outcomes)} quality metrics")
    
    if not quality_outcomes:
        print("ERROR: No quality metrics found")
        return 1
    
    # Filter data
    bc_quality = bc_data[bc_data[outcome_col].isin(quality_outcomes)].copy()
    other_quality = other_provinces[other_provinces[outcome_col].isin(quality_outcomes)].copy()
    
    print(f"\nAfter filtering:")
    print(f"  BC: {len(bc_quality):,}")
    print(f"  Other: {len(other_quality):,}")
    
    # Use most recent year only
    if 'REF_DATE' in df.columns:
        recent = df['REF_DATE'].max()
        print(f"\nUsing most recent year: {recent}")
        bc_quality = bc_quality[bc_quality['REF_DATE'] == recent]
        other_quality = other_quality[other_quality['REF_DATE'] == recent]
        print(f"  BC (recent): {len(bc_quality):,}")
        print(f"  Other (recent): {len(other_quality):,}")
    
    # Compare outcomes
    print(f"\n{'='*80}")
    print("COMPARING OUTCOMES")
    print(f"{'='*80}\n")
    
    bc_summary = bc_quality.groupby(outcome_col)['VALUE'].agg(['mean', 'median', 'count']).reset_index()
    other_summary = other_quality.groupby(outcome_col)['VALUE'].agg(['mean', 'median', 'count']).reset_index()
    
    comparison = pd.merge(bc_summary, other_summary, on=outcome_col, suffixes=('_bc', '_other'))
    comparison['mean_diff'] = comparison['mean_bc'] - comparison['mean_other']
    comparison['pct_diff'] = 100 * (comparison['mean_bc'] / comparison['mean_other'] - 1)
    comparison['abs_diff'] = comparison['mean_diff'].abs()
    comparison = comparison.sort_values('abs_diff', ascending=False)
    
    print("TOP DIFFERENCES:")
    print("-" * 80)
    print(f"{'Outcome':<45} {'BC':>12} {'Other':>12} {'Diff %':>10}")
    print("-" * 80)
    
    for _, row in comparison.head(15).iterrows():
        name = str(row[outcome_col])[:43]
        print(f"{name:<45} {row['mean_bc']:>12.0f} {row['mean_other']:>12.0f} {row['pct_diff']:>9.1f}%")
    
    comparison.to_csv(OUT_DIR / "bc_vs_canada_comparison.csv", index=False)
    print(f"\n✓ Saved: bc_vs_canada_comparison.csv")
    
    # Visualizations
    print(f"\n{'='*80}")
    print("GENERATING CHARTS")
    print(f"{'='*80}\n")
    
    plot_data = comparison.head(10)
    
    # Chart 1: Side-by-side comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(plot_data))
    
    ax.barh(y_pos, plot_data['mean_bc'].values, alpha=0.7, label='BC', color='#1f77b4')
    ax.barh(y_pos, plot_data['mean_other'].values, alpha=0.7, label='Rest of Canada', color='#ff7f0e')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(x)[:35] for x in plot_data[outcome_col].values], fontsize=9)
    ax.set_xlabel('Mean Value')
    ax.set_title('BC vs Rest of Canada - Top Differences', fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bc_vs_canada_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Comparison chart")
    
    # Chart 2: Percentage differences
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['green' if x > 0 else 'red' for x in plot_data['pct_diff'].values]
    ax.barh(y_pos, plot_data['pct_diff'].values, color=colors, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(x)[:35] for x in plot_data[outcome_col].values], fontsize=9)
    ax.set_xlabel('Percent Difference (%)')
    ax.set_title('BC vs Rest of Canada\n(Green = BC Higher, Red = BC Lower)', fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bc_vs_canada_pct_diff.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Percentage difference chart")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    better_bc = (comparison['mean_diff'] > 0).sum()
    better_other = (comparison['mean_diff'] < 0).sum()
    
    print(f"Outcomes analyzed: {len(comparison)}")
    print(f"BC higher: {better_bc}")
    print(f"Other provinces higher: {better_other}")
    
    if len(comparison) > 0:
        avg = comparison['mean_diff'].mean()
        print(f"\nAverage difference: {avg:.1f}")
        print(f"BC tends to be {'HIGHER' if avg > 0 else 'LOWER'} overall")
    
    print(f"\n{'='*80}")
    print("✓ COMPLETE!")
    print(f"{'='*80}")
    print("\nFiles created:")
    print(f"  • bc_vs_canada_comparison.csv")
    print(f"  • bc_vs_canada_comparison.png")
    print(f"  • bc_vs_canada_pct_diff.png")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())