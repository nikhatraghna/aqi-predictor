"""Detect feature drift: reference (older) vs recent window, using KS + PSI together.

A feature is DRIFT only when KS is significant AND PSI magnitude is large — this avoids
KS's tendency to over-flag on large reference samples. Monitors the production model's
features (from feature_config.json).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

FEATURES_PATH = "data/processed/islamabad_features.parquet"
CONTRACT_PATH = Path("models/best_model/feature_config.json")
REPORT_DIR    = Path("reports/drift")
REPORT_PATH   = REPORT_DIR / "data_drift_report.json"

RECENT_WINDOW = 72
KS_THRESHOLD  = 0.05    # KS p-value below this = statistically significant
PSI_WARN      = 0.10    # PSI moderate shift
PSI_DRIFT     = 0.20    # PSI significant shift


def get_features_to_check(df: pd.DataFrame) -> list:
    """Use the production model's feature list; fall back to numeric cols."""
    if CONTRACT_PATH.exists():
        with open(CONTRACT_PATH) as f:
            feats = json.load(f).get("features", [])
        feats = [c for c in feats if c in df.columns]
        if feats:
            return feats
    exclude = {"pm25", "datetime"}
    return [c for c in df.select_dtypes("number").columns if c not in exclude]


def compute_psi(expected, actual, buckets: int = 10) -> float:
    """Population Stability Index using reference-quantile bins."""
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]
    if len(expected) < 2 or len(actual) < 2:
        return 0.0

    edges = np.unique(np.percentile(expected, np.linspace(0, 100, buckets + 1)))
    if len(edges) < 3:           # not enough distinct values to bin
        return 0.0

    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual,   bins=edges)
    eps    = 1e-6
    e_perc = np.clip(e_counts / e_counts.sum(), eps, None)
    a_perc = np.clip(a_counts / a_counts.sum(), eps, None)
    psi = np.sum((a_perc - e_perc) * np.log(a_perc / e_perc))
    return float(round(psi, 4))


def load_windows():
    """Split the feature data into reference (older) and recent windows."""
    print("\n[INFO] Loading feature dataset...")
    df = pd.read_parquet(FEATURES_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    if len(df) <= RECENT_WINDOW + 10:
        raise ValueError("Not enough data to split reference vs recent windows.")

    recent_df    = df.tail(RECENT_WINDOW).copy()
    reference_df = df.iloc[:-RECENT_WINDOW].copy()
    print(f"[INFO] Reference rows : {len(reference_df)}")
    print(f"[INFO] Recent rows    : {len(recent_df)}")
    return reference_df, recent_df


def classify(ks_p: float, psi: float) -> str:
    """DRIFT only when KS significant AND PSI large; KS-only or moderate PSI → WARNING."""
    ks_sig = ks_p < KS_THRESHOLD
    if ks_sig and psi >= PSI_DRIFT:
        return "🔴 DRIFT"
    if ks_sig or psi >= PSI_WARN:
        return "🟡 WARNING"
    return "🟢 NORMAL"


def check_feature_drift(reference_df, recent_df, features):
    """KS + PSI per feature: reference vs recent distribution."""
    print("\n=================================================")
    print(" DATA DRIFT REPORT (KS + PSI) ")
    print("=================================================")
    print(f"  {'feature':<22} {'KS p':>9} {'PSI':>7}  status")
    print("  " + "-" * 50)

    results = []
    for feat in features:
        ref = reference_df[feat].dropna()
        rec = recent_df[feat].dropna()
        if len(ref) < 2 or len(rec) < 2:
            continue
        _, ks_p = ks_2samp(ref, rec)
        psi     = compute_psi(ref, rec)
        status  = classify(ks_p, psi)
        results.append({
            "feature":  feat,
            "ks_pvalue": round(float(ks_p), 6),
            "psi":       psi,
            "status":    status,
        })
        print(f"  {feat:<22} {ks_p:>9.4f} {psi:>7.4f}  {status}")
    return results


def save_report(results: list):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[SUCCESS] Report saved → {REPORT_PATH}")


def main():
    print("\n=================================================")
    print(" AQI DATA DRIFT MONITOR ")
    print("=================================================")

    reference_df, recent_df = load_windows()
    features = get_features_to_check(reference_df)
    print(f"[INFO] Monitoring {len(features)} features: {features}")

    results = check_feature_drift(reference_df, recent_df, features)
    n_drift = sum(1 for r in results if "DRIFT" in r["status"])
    n_warn  = sum(1 for r in results if "WARNING" in r["status"])

    save_report(results)
    print(f"\n[INFO] DRIFT: {n_drift}  |  WARNING: {n_warn}  |  "
          f"NORMAL: {len(results) - n_drift - n_warn}  (of {len(results)})")
    print("=================================================")
    print(" DRIFT ANALYSIS COMPLETE ")
    print("=================================================")


if __name__ == "__main__":
    main()
