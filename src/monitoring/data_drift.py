"""Detect data drift between training and inference feature distributions."""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

FEATURE_PATH   = "data/processed/islamabad_features.parquet"
FORECAST_PATH  = "data/processed/forecast_3days.parquet"
REPORT_PATH    = "reports/drift/data_drift_report.json"

FEATURES_TO_CHECK = [
    "temperature", "humidity", "pressure", "wind_speed",
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_24",
    "pm25_roll_mean_3", "pm25_roll_mean_24",
]

KS_THRESHOLD  = 0.05   # p-value below this = drift
PSI_WARNING   = 0.1    # PSI warning threshold
PSI_DRIFT     = 0.2    # PSI drift threshold


def compute_psi(expected: np.ndarray, actual: np.ndarray,
               buckets: int = 10) -> float:
    """Compute Population Stability Index between two distributions.

    PSI < 0.1  : No drift
    PSI < 0.2  : Warning
    PSI >= 0.2 : Drift detected

    Args:
        expected: Training distribution array.
        actual:   Inference distribution array.
        buckets:  Number of bins.

    Returns:
        PSI score as float.
    """
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        buckets + 1,
    )
    expected_counts = np.histogram(expected, breakpoints)[0] + 1e-6
    actual_counts   = np.histogram(actual,   breakpoints)[0] + 1e-6
    expected_pct    = expected_counts / expected_counts.sum()
    actual_pct      = actual_counts   / actual_counts.sum()
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(round(psi, 4))


def check_drift(train_df: pd.DataFrame,
               inference_df: pd.DataFrame) -> list:
    """Run KS test and PSI for each feature.

    Args:
        train_df:     Training feature DataFrame.
        inference_df: Inference feature DataFrame.

    Returns:
        List of drift result dicts per feature.
    """
    results = []
    for col in FEATURES_TO_CHECK:
        if col not in train_df.columns or col not in inference_df.columns:
            continue
        train_vals     = train_df[col].dropna().values
        inference_vals = inference_df[col].dropna().values
        if len(inference_vals) < 2:
            continue

        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(train_vals, inference_vals)

        # PSI
        psi = compute_psi(train_vals, inference_vals)

        # Mean and std shift
        mean_shift = abs(train_vals.mean() - inference_vals.mean())
        std_shift  = abs(train_vals.std()  - inference_vals.std())

        # Determine status
        if ks_pvalue < KS_THRESHOLD or psi >= PSI_DRIFT:
            status = "🔴 DRIFT DETECTED"
        elif psi >= PSI_WARNING:
            status = "🟡 WARNING"
        else:
            status = "🟢 Normal"

        results.append({
            "feature":    col,
            "ks_stat":    round(float(ks_stat),   4),
            "ks_pvalue":  round(float(ks_pvalue), 4),
            "psi":        psi,
            "mean_shift": round(float(mean_shift), 4),
            "std_shift":  round(float(std_shift),  4),
            "status":     status,
        })
    return results


def print_report(results: list) -> None:
    """Print a formatted drift report table.

    Args:
        results: List of drift result dicts.
    """
    print("\n" + "=" * 72)
    print("  DATA DRIFT REPORT")
    print("=" * 72)
    print(f"  {'Feature':<25} {'KS p-val':>9} {'PSI':>7} {'Mean Δ':>9}  Status")
    print("-" * 72)
    for r in results:
        print(
            f"  {r['feature']:<25} "
            f"{r['ks_pvalue']:>9.4f} "
            f"{r['psi']:>7.4f} "
            f"{r['mean_shift']:>9.4f}  "
            f"{r['status']}"
        )
    print("=" * 72)
    drifted  = [r for r in results if "DRIFT"   in r["status"]]
    warnings = [r for r in results if "WARNING" in r["status"]]
    print(f"  Drift detected : {len(drifted)} feature(s)")
    print(f"  Warnings       : {len(warnings)} feature(s)")
    print(f"  Normal         : {len(results) - len(drifted) - len(warnings)} feature(s)")
    print("=" * 72)


def save_report(results: list) -> None:
    """Save drift report to JSON.

    Args:
        results: List of drift result dicts.
    """
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[SUCCESS] Report saved → {REPORT_PATH}")


def run_drift_check() -> list:
    """Load data, run drift checks, print and save report.

    Returns:
        List of drift result dicts.
    """
    print("[INFO] Loading training features...")
    train_df = pd.read_parquet(FEATURE_PATH)

    # Use last 72 rows as inference batch proxy
    # In production this would be live incoming data
    print("[INFO] Using last 72 rows as inference batch...")
    inference_df = train_df.tail(72).copy()
    train_df     = train_df.iloc[:-72].copy()

    print(f"[INFO] Train rows     : {len(train_df)}")
    print(f"[INFO] Inference rows : {len(inference_df)}")

    print("[INFO] Running drift checks...")
    results = check_drift(train_df, inference_df)

    print_report(results)
    save_report(results)
    return results


if __name__ == "__main__":
    results = run_drift_check()
