from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from statcan_wds import load_statcan_cube_full_table


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


# StatsCan cubes (strip hyphens from table numbers)
CUBE_VACANCY = "1410043201"   # 14-10-0432-01
CUBE_LFS = "1410028701"       # 14-10-0287-01


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found. Columns: {list(df.columns)[:25]}...")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vac = load_statcan_cube_full_table(CUBE_VACANCY, DATA_DIR)
    lfs = load_statcan_cube_full_table(CUBE_LFS, DATA_DIR)

    # These full-table CSVs typically have columns like:
    # REF_DATE, GEO, DGUID, UOM, UOM_ID, SCALAR_FACTOR, SCALAR_ID, VECTOR, COORDINATE, VALUE, STATUS, SYMBOL, TERMINATED, DECIMALS
    # Plus sometimes extra dimensions.
    for df in (vac, lfs):
        df.columns = [c.strip() for c in df.columns]

    date_col_v = _find_col(vac, ["REF_DATE", "Ref_Date", "ref_date"])
    geo_col_v = _find_col(vac, ["GEO", "Geo", "geo"])
    val_col_v = _find_col(vac, ["VALUE", "Value", "value"])

    date_col_l = _find_col(lfs, ["REF_DATE", "Ref_Date", "ref_date"])
    geo_col_l = _find_col(lfs, ["GEO", "Geo", "geo"])
    val_col_l = _find_col(lfs, ["VALUE", "Value", "value"])

    # Identify the vacancy rate series rows (the table includes multiple "Statistics")
    stat_col_v = [c for c in vac.columns if c.lower() in ("statistics", "statistic", "statistiques")]
    if not stat_col_v:
        raise SystemExit("Could not find a 'Statistics' column in vacancy table. Inspect the downloaded CSV.")
    stat_col_v = stat_col_v[0]

    vac_rate = vac[vac[stat_col_v].astype(str).str.contains("job vacancy rate", case=False, na=False)].copy()
    vac_rate = vac_rate[[date_col_v, geo_col_v, val_col_v]].rename(
        columns={date_col_v: "date", geo_col_v: "geo", val_col_v: "vacancy_rate"}
    )

    # Identify unemployment rate in LFS cube
    stat_col_l = [c for c in lfs.columns if c.lower() in ("statistics", "statistic", "statistiques")]
    if not stat_col_l:
        raise SystemExit("Could not find a 'Statistics' column in LFS table. Inspect the downloaded CSV.")
    stat_col_l = stat_col_l[0]

    unemp = lfs[lfs[stat_col_l].astype(str).str.contains("unemployment rate", case=False, na=False)].copy()
    unemp = unemp[[date_col_l, geo_col_l, val_col_l]].rename(
        columns={date_col_l: "date", geo_col_l: "geo", val_col_l: "unemp_rate"}
    )

    # Parse dates (StatsCan REF_DATE is often "YYYY-MM")
    vac_rate["date"] = pd.to_datetime(vac_rate["date"].astype(str), errors="coerce")
    unemp["date"] = pd.to_datetime(unemp["date"].astype(str), errors="coerce")

    df = pd.merge(vac_rate, unemp, on=["date", "geo"], how="inner").dropna()
    df = df.sort_values(["geo", "date"]).reset_index(drop=True)

    # Tightness definition (derived label):
    # tightness = vacancy_rate / unemp_rate (simple proxy)
    df["tightness"] = df["vacancy_rate"] / df["unemp_rate"].replace(0, np.nan)

    # Label: "tight" if tightness is above the 70th percentile within each province/territory
    df["tight_label"] = (
        df.groupby("geo")["tightness"]
          .transform(lambda s: s > s.quantile(0.70))
          .astype(int)
    )

    # Features: current + lags
    for lag in [1, 2, 3]:
        df[f"vac_lag{lag}"] = df.groupby("geo")["vacancy_rate"].shift(lag)
        df[f"unemp_lag{lag}"] = df.groupby("geo")["unemp_rate"].shift(lag)
        df[f"tight_lag{lag}"] = df.groupby("geo")["tightness"].shift(lag)

    feat_cols = ["vacancy_rate", "unemp_rate", "tightness",
                 "vac_lag1", "vac_lag2", "vac_lag3",
                 "unemp_lag1", "unemp_lag2", "unemp_lag3",
                 "tight_lag1", "tight_lag2", "tight_lag3"]

    model_df = df.dropna(subset=feat_cols + ["tight_label"]).copy()

    X = model_df[feat_cols].to_numpy(dtype=float)
    y = model_df["tight_label"].to_numpy(dtype=int)

    # Time-series split (keeps chronology)
    # We'll split on the overall time ordering (still mixed geos, but strictly time-respecting).
    model_df = model_df.sort_values("date")
    X = model_df[feat_cols].to_numpy(dtype=float)
    y = model_df["tight_label"].to_numpy(dtype=int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)

    tscv = TimeSeriesSplit(n_splits=5)
    probs_all = np.full_like(y, fill_value=np.nan, dtype=float)

    for train_idx, test_idx in tscv.split(Xs):
        clf.fit(Xs[train_idx], y[train_idx])
        probs_all[test_idx] = clf.predict_proba(Xs[test_idx])[:, 1]

    # Evaluate
    ap = average_precision_score(y[~np.isnan(probs_all)], probs_all[~np.isnan(probs_all)])
    print(f"Average precision (PR-AUC proxy): {ap:.3f}")

    # Plot A: ROC (using final fit on all data for a clean display)
    clf.fit(Xs, y)
    plt.figure()
    RocCurveDisplay.from_estimator(clf, Xs, y)
    plt.title("Tight vs Loose – ROC (Logistic Regression)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tightness_predictor_roc.png", dpi=200)

    # Plot B: Confusion matrix at 0.5
    yhat = (clf.predict_proba(Xs)[:, 1] >= 0.5).astype(int)
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y, yhat)
    plt.title("Tight vs Loose – Confusion Matrix (threshold=0.5)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tightness_predictor_confusion.png", dpi=200)

    # Plot C: Canada-level monthly average predicted tightness probability
    model_df["p_tight"] = clf.predict_proba(Xs)[:, 1]
    agg = model_df.groupby("date")["p_tight"].mean().reset_index()

    plt.figure()
    plt.plot(agg["date"], agg["p_tight"])
    plt.title("Average Predicted Labour Market Tightness (Canada – mean over provinces)")
    plt.xlabel("Date")
    plt.ylabel("P(tight)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tightness_predictor_canada_timeseries.png", dpi=200)

    print("Wrote:")
    print(f"- {OUT_DIR / 'tightness_predictor_roc.png'}")
    print(f"- {OUT_DIR / 'tightness_predictor_confusion.png'}")
    print(f"- {OUT_DIR / 'tightness_predictor_canada_timeseries.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
