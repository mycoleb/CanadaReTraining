from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


def load_table_from_manual_download(table_number: str, data_dir: Path) -> pd.DataFrame:
    """
    Load a manually downloaded StatsCan table.
    Tries multiple filename patterns and encoding strategies.
    """
    # Possible filenames
    possible_names = [
        f"{table_number}.csv",
        f"{table_number}-eng.csv",
        f"{table_number}_eng.csv",
    ]
    
    filepath = None
    for name in possible_names:
        test_path = data_dir / name
        if test_path.exists():
            filepath = test_path
            break
    
    if filepath is None:
        formatted = f"{table_number[:2]}-{table_number[2:4]}-{table_number[4:8]}-{table_number[8:]}"
        print(f"\n{'='*80}")
        print(f"ERROR: Could not find table {table_number}")
        print(f"{'='*80}")
        print(f"\nLooked for files:")
        for name in possible_names:
            print(f"  - {data_dir / name}")
        
        print(f"\nMANUAL DOWNLOAD INSTRUCTIONS:")
        print(f"1. Go to: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid={formatted}")
        print(f"2. Click 'Download options' → 'Download entire table' → CSV")
        print(f"3. Extract the ZIP file")
        print(f"4. Save the main CSV file as: {data_dir / f'{table_number}-eng.csv'}")
        print(f"5. Run this script again")
        print(f"{'='*80}\n")
        
        raise FileNotFoundError(f"Table {table_number} not found")
    
    print(f"✓ Found: {filepath}")
    
    # Try multiple parsing strategies
    parsing_strategies = [
        # Strategy 1: Standard UTF-8 with BOM
        {'encoding': 'utf-8-sig', 'on_bad_lines': 'skip'},
        # Strategy 2: Latin-1 encoding
        {'encoding': 'latin-1', 'on_bad_lines': 'skip'},
        # Strategy 3: UTF-8 without BOM
        {'encoding': 'utf-8', 'on_bad_lines': 'skip'},
        # Strategy 4: With quoting
        {'encoding': 'utf-8-sig', 'on_bad_lines': 'skip', 'quoting': 3},
    ]
    
    last_error = None
    for i, strategy in enumerate(parsing_strategies, 1):
        try:
            print(f"  Trying parsing strategy {i}...")
            df = pd.read_csv(filepath, **strategy, low_memory=False)
            
            # Verify it looks like a StatsCan table
            if df.shape[0] > 0 and df.shape[1] > 5:
                print(f"  ✓ Successfully loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
                return df
            else:
                print(f"  ✗ File loaded but appears invalid (too few rows/columns)")
        except Exception as e:
            last_error = e
            print(f"  ✗ Strategy {i} failed: {type(e).__name__}")
            continue
    
    # All strategies failed
    print(f"\n{'='*80}")
    print(f"ERROR: Could not parse {filepath}")
    print(f"{'='*80}")
    print(f"Last error: {last_error}")
    print("\nThe CSV file may be corrupted or in an unexpected format.")
    print("Try:")
    print("1. Re-download the file from StatsCan")
    print("2. Make sure you extracted the CSV from the ZIP (not using the ZIP directly)")
    print("3. Open the CSV in a text editor to check if it looks normal")
    print(f"{'='*80}\n")
    
    raise ValueError(f"Could not parse {filepath}")


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found. Columns: {list(df.columns)[:25]}...")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("JOB MARKET TIGHTNESS PREDICTOR")
    print("="*80)
    print("\nThis script requires two datasets:")
    print("1. Table 14-10-0432-01: Job vacancy statistics")
    print("2. Table 14-10-0287-01: Labour force survey")
    print("\nLooking for manually downloaded files...")
    print("="*80 + "\n")

    # Load vacancy data
    try:
        print("[1/2] Loading job vacancy data (Table 14-10-0432-01)...")
        vac = load_table_from_manual_download("1410043201", DATA_DIR)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nFailed to load vacancy data: {e}")
        return 1

    # Load labour force data
    try:
        print("\n[2/2] Loading labour force survey data (Table 14-10-0287-01)...")
        lfs = load_table_from_manual_download("1410028701", DATA_DIR)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nFailed to load labour force data: {e}")
        return 1

    print("\n" + "="*80)
    print("DATA LOADED SUCCESSFULLY - PROCESSING...")
    print("="*80 + "\n")

    # Clean column names
    for df in (vac, lfs):
        df.columns = [c.strip() for c in df.columns]

    # Find required columns
    try:
        date_col_v = _find_col(vac, ["REF_DATE", "Ref_Date", "ref_date"])
        geo_col_v = _find_col(vac, ["GEO", "Geo", "geo"])
        val_col_v = _find_col(vac, ["VALUE", "Value", "value"])

        date_col_l = _find_col(lfs, ["REF_DATE", "Ref_Date", "ref_date"])
        geo_col_l = _find_col(lfs, ["GEO", "Geo", "geo"])
        val_col_l = _find_col(lfs, ["VALUE", "Value", "value"])
    except KeyError as e:
        print(f"ERROR: {e}")
        return 1

    # Identify the vacancy rate series rows
    stat_col_v = [c for c in vac.columns if c.lower() in ("statistics", "statistic", "statistiques")]
    if not stat_col_v:
        print("ERROR: Could not find 'Statistics' column in vacancy table")
        print("Available columns:", vac.columns.tolist())
        return 1
    stat_col_v = stat_col_v[0]

    print(f"Filtering vacancy data for 'job vacancy rate'...")
    vac_rate = vac[vac[stat_col_v].astype(str).str.contains("job vacancy rate", case=False, na=False)].copy()
    
    if len(vac_rate) == 0:
        print(f"ERROR: No 'job vacancy rate' found in {stat_col_v} column")
        print(f"Available values in column '{stat_col_v}':")
        for val in vac[stat_col_v].unique()[:15]:
            print(f"  - {val}")
        return 1
    
    vac_rate = vac_rate[[date_col_v, geo_col_v, val_col_v]].rename(
        columns={date_col_v: "date", geo_col_v: "geo", val_col_v: "vacancy_rate"}
    )
    print(f"  ✓ Found {len(vac_rate):,} vacancy rate records")

    # Identify unemployment rate in LFS cube
    stat_col_l = [c for c in lfs.columns if c.lower() in ("statistics", "statistic", "statistiques")]
    if not stat_col_l:
        print("ERROR: Could not find 'Statistics' column in LFS table")
        print("Available columns:", lfs.columns.tolist())
        return 1
    stat_col_l = stat_col_l[0]

    print(f"Filtering LFS data for 'unemployment rate'...")
    unemp = lfs[lfs[stat_col_l].astype(str).str.contains("unemployment rate", case=False, na=False)].copy()
    
    if len(unemp) == 0:
        print(f"ERROR: No 'unemployment rate' found in {stat_col_l} column")
        print(f"Available values in column '{stat_col_l}':")
        for val in lfs[stat_col_l].unique()[:15]:
            print(f"  - {val}")
        return 1
    
    unemp = unemp[[date_col_l, geo_col_l, val_col_l]].rename(
        columns={date_col_l: "date", geo_col_l: "geo", val_col_l: "unemp_rate"}
    )
    print(f"  ✓ Found {len(unemp):,} unemployment rate records")

    # Parse dates
    vac_rate["date"] = pd.to_datetime(vac_rate["date"].astype(str), errors="coerce")
    unemp["date"] = pd.to_datetime(unemp["date"].astype(str), errors="coerce")

    print(f"\nMerging datasets...")
    df = pd.merge(vac_rate, unemp, on=["date", "geo"], how="inner").dropna()
    df = df.sort_values(["geo", "date"]).reset_index(drop=True)
    print(f"  ✓ Merged data: {len(df):,} records")

    if len(df) < 100:
        print("ERROR: Too few records after merging. Check that both datasets cover the same time periods and geographies.")
        return 1

    # Calculate tightness
    print(f"\nCalculating labour market tightness metric...")
    df["tightness"] = df["vacancy_rate"] / df["unemp_rate"].replace(0, np.nan)

    # Create binary label: tight vs loose
    df["tight_label"] = (
        df.groupby("geo")["tightness"]
          .transform(lambda s: s > s.quantile(0.70))
          .astype(int)
    )

    print(f"  Tight markets: {df['tight_label'].sum():,} records ({100*df['tight_label'].mean():.1f}%)")
    print(f"  Loose markets: {(~df['tight_label'].astype(bool)).sum():,} records ({100*(1-df['tight_label'].mean()):.1f}%)")

    # Create lagged features
    print(f"\nCreating lagged features...")
    for lag in [1, 2, 3]:
        df[f"vac_lag{lag}"] = df.groupby("geo")["vacancy_rate"].shift(lag)
        df[f"unemp_lag{lag}"] = df.groupby("geo")["unemp_rate"].shift(lag)
        df[f"tight_lag{lag}"] = df.groupby("geo")["tightness"].shift(lag)

    feat_cols = ["vacancy_rate", "unemp_rate", "tightness",
                 "vac_lag1", "vac_lag2", "vac_lag3",
                 "unemp_lag1", "unemp_lag2", "unemp_lag3",
                 "tight_lag1", "tight_lag2", "tight_lag3"]

    model_df = df.dropna(subset=feat_cols + ["tight_label"]).copy()
    print(f"  ✓ Final dataset for modeling: {len(model_df):,} records")

    # Prepare for time-series cross-validation
    model_df = model_df.sort_values("date")
    X = model_df[feat_cols].to_numpy(dtype=float)
    y = model_df["tight_label"].to_numpy(dtype=int)

    print(f"\n{'='*80}")
    print("TRAINING MODEL")
    print(f"{'='*80}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)

    # Time-series cross-validation
    print(f"\nRunning time-series cross-validation (5 splits)...")
    tscv = TimeSeriesSplit(n_splits=5)
    probs_all = np.full_like(y, fill_value=np.nan, dtype=float)

    for i, (train_idx, test_idx) in enumerate(tscv.split(Xs), 1):
        print(f"  Fold {i}/5: Training on {len(train_idx):,} samples, testing on {len(test_idx):,} samples")
        clf.fit(Xs[train_idx], y[train_idx])
        probs_all[test_idx] = clf.predict_proba(Xs[test_idx])[:, 1]

    # Evaluate
    ap = average_precision_score(y[~np.isnan(probs_all)], probs_all[~np.isnan(probs_all)])
    print(f"\n✓ Average Precision Score: {ap:.3f}")

    # Final model on all data for visualization
    print(f"\nTraining final model on all data...")
    clf.fit(Xs, y)

    # Generate plots
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")

    # Plot A: ROC curve
    print("  Creating ROC curve...")
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(clf, Xs, y)
    plt.title("Labour Market Tightness – ROC Curve")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tightness_predictor_roc.png", dpi=200)
    plt.close()

    # Plot B: Confusion matrix
    print("  Creating confusion matrix...")
    yhat = (clf.predict_proba(Xs)[:, 1] >= 0.5).astype(int)
    plt.figure(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y, yhat)
    plt.title("Labour Market Tightness – Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tightness_predictor_confusion.png", dpi=200)
    plt.close()

    # Plot C: Time series of predicted tightness
    print("  Creating time series visualization...")
    model_df["p_tight"] = clf.predict_proba(Xs)[:, 1]
    agg = model_df.groupby("date")["p_tight"].mean().reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(agg["date"], agg["p_tight"], linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
    plt.title("Predicted Labour Market Tightness Over Time\n(Canada – Average Across Provinces)")
    plt.xlabel("Date")
    plt.ylabel("Probability of Tight Market")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tightness_predictor_canada_timeseries.png", dpi=200)
    plt.close()

    print(f"\n{'='*80}")
    print("SUCCESS! Output files created:")
    print(f"{'='*80}")
    print(f"  ✓ {OUT_DIR / 'tightness_predictor_roc.png'}")
    print(f"  ✓ {OUT_DIR / 'tightness_predictor_confusion.png'}")
    print(f"  ✓ {OUT_DIR / 'tightness_predictor_canada_timeseries.png'}")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())