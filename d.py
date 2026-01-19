"""
Diagnostic script to inspect StatsCan CSV files and identify issues.
"""
from pathlib import Path
import pandas as pd


DATA_DIR = Path("data")


def inspect_csv(filepath: Path):
    """Inspect a CSV file and report its contents."""
    print(f"\n{'='*80}")
    print(f"INSPECTING: {filepath.name}")
    print(f"{'='*80}")
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return
    
    # Check file size
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"\nFile size: {size_mb:.2f} MB")
    
    if size_mb < 0.1:
        print(f"⚠️  WARNING: File is very small ({size_mb:.2f} MB)")
        print("   This might be an error page or corrupted download")
    
    # Check first few lines
    print(f"\nFirst 10 lines of file:")
    print("-" * 80)
    try:
        with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                print(f"{i+1:3d}: {line.rstrip()[:100]}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    # Try to load as CSV
    print(f"\n" + "-" * 80)
    print("Attempting to load as CSV...")
    
    strategies = [
        {'encoding': 'utf-8-sig', 'on_bad_lines': 'skip'},
        {'encoding': 'latin-1', 'on_bad_lines': 'skip'},
    ]
    
    for i, strategy in enumerate(strategies, 1):
        try:
            print(f"\nStrategy {i}: {strategy}")
            df = pd.read_csv(filepath, **strategy, low_memory=False, nrows=100)
            
            print(f"✓ Loaded successfully!")
            print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"\n  Columns:")
            for j, col in enumerate(df.columns, 1):
                print(f"    {j:2d}. {col}")
            
            print(f"\n  First 3 rows:")
            print(df.head(3).to_string())
            
            print(f"\n  Data types:")
            print(df.dtypes)
            
            return df
            
        except Exception as e:
            print(f"  ❌ Failed: {type(e).__name__}: {e}")
            continue
    
    print(f"\n❌ All loading strategies failed!")


def main():
    print("="*80)
    print("STATCAN CSV FILE DIAGNOSTIC TOOL")
    print("="*80)
    
    # Files to check
    files_to_check = [
        "1410043201-eng.csv",
        "1410028701-eng.csv",
        "37100115.csv",
        "37100115-eng.csv",
    ]
    
    for filename in files_to_check:
        filepath = DATA_DIR / filename
        if filepath.exists():
            df = inspect_csv(filepath)
        else:
            print(f"\n⏭️  Skipping {filename} (not found)")
    
    print(f"\n{'='*80}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("1. Check if file sizes seem reasonable (should be 10-200 MB)")
    print("2. If files are tiny (<1 MB), they're probably error pages - re-download")
    print("3. If files won't load, they may be corrupted - re-download")
    print("4. Make sure you extracted CSVs from the ZIP files")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()