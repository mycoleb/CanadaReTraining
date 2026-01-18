"""
Explore the StatsCan data files to understand their structure.
Run this first to see what columns and data you have.
"""
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def explore_metadata():
    """Explore the metadata CSV to understand the data dictionary."""
    meta_path = DATA_DIR / "37100115_MetaData.csv"
    
    if not meta_path.exists():
        print(f"Metadata file not found: {meta_path}")
        return
    
    print("=" * 70)
    print("METADATA FILE EXPLORATION")
    print("=" * 70)
    
    meta = pd.read_csv(meta_path, encoding='utf-8-sig')
    print(f"\nMetadata shape: {meta.shape[0]} rows, {meta.shape[1]} columns")
    print(f"\nColumns in metadata:")
    for i, col in enumerate(meta.columns, 1):
        print(f"  {i}. {col}")
    
    print("\nFirst few rows:")
    print(meta.head(10).to_string())


def explore_data():
    """Explore the main data CSV."""
    data_path = DATA_DIR / "37100115.csv"
    
    if not data_path.exists():
        print(f"\nMain data file not found: {data_path}")
        print("\nExpected location: data/37100115.csv")
        return
    
    print("\n" + "=" * 70)
    print("MAIN DATA FILE EXPLORATION")
    print("=" * 70)
    
    # Read first 1000 rows to get a sense of structure
    df = pd.read_csv(data_path, encoding='utf-8-sig', nrows=1000)
    
    print(f"\nData shape (first 1000 rows): {df.shape[0]} rows, {df.shape[1]} columns")
    
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        print(f"  {i:2d}. {col:30s} | dtype: {str(dtype):10s} | unique: {n_unique}")
    
    print("\n" + "=" * 70)
    print("SAMPLE DATA (first 5 rows):")
    print("=" * 70)
    print(df.head().to_string())
    
    # Identify dimension columns
    metadata_cols = {
        'REF_DATE', 'GEO', 'DGUID', 'VALUE', 'STATUS', 'SYMBOL', 
        'TERMINATED', 'DECIMALS', 'UOM', 'UOM_ID', 'SCALAR_FACTOR', 
        'SCALAR_ID', 'VECTOR', 'COORDINATE'
    }
    
    dim_cols = [c for c in df.columns if c not in metadata_cols]
    
    print("\n" + "=" * 70)
    print("DIMENSION COLUMNS (these define the data structure):")
    print("=" * 70)
    for col in dim_cols:
        unique_vals = df[col].dropna().unique()
        print(f"\n{col}:")
        print(f"  Unique values: {len(unique_vals)}")
        print(f"  Sample values: {list(unique_vals[:5])}")
    
    print("\n" + "=" * 70)
    print("VALUE COLUMN STATISTICS:")
    print("=" * 70)
    if 'VALUE' in df.columns:
        print(df['VALUE'].describe())
        print(f"\nNon-null values: {df['VALUE'].notna().sum()}")
        print(f"Null values: {df['VALUE'].isna().sum()}")
    
    print("\n" + "=" * 70)
    print("GEOGRAPHY BREAKDOWN:")
    print("=" * 70)
    if 'GEO' in df.columns:
        print(df['GEO'].value_counts().head(15))
    
    print("\n" + "=" * 70)
    print("TIME PERIOD BREAKDOWN:")
    print("=" * 70)
    if 'REF_DATE' in df.columns:
        print(f"Earliest: {df['REF_DATE'].min()}")
        print(f"Latest: {df['REF_DATE'].max()}")
        print(f"\nSample dates: {df['REF_DATE'].unique()[:10]}")


def main():
    print("\nSTATCAN DATA EXPLORER")
    print("=" * 70)
    
    explore_metadata()
    explore_data()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Review the dimension columns above")
    print("2. Identify which dimension represents outcome types")
    print("   (e.g., 'Employment rate', 'Median earnings', etc.)")
    print("3. Run the updated analysis script:")
    print("   python src/01_retraining_success_clusters_UPDATED.py")
    print("=" * 70)


if __name__ == "__main__":
    main()