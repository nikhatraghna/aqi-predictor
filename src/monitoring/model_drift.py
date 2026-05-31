"""Model drift monitoring — recent performance vs the model's baseline (contract-driven)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.inference.predict import predict
from src.inference.load_model import load_feature_config

FEATURES_PATH = "data/processed/islamabad_features.parquet"
REPORT_DIR    = Path("reports/drift")
REPORT_PATH   = REPORT_DIR / "model_drift_report.json"
TARGET        = "pm25"
RECENT_WINDOW = 72

# Drift thresholds = degradation vs baseline RMSE
WARN_PCT  = 0.20   # +20% worse → WARNING
DRIFT_PCT = 0.40   # +40% worse → DRIFT


def load_recent_batch() -> pd.DataFrame:
    print("[INFO] Loading engineered feature dataset...")
    df = pd.read_parquet(FEATURES_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    recent = df.tail(RECENT_WINDOW).copy()
    print(f"[INFO] Recent batch: {recent.shape}")
    return recent


def get_baseline_rmse() -> float:
    """Baseline = the model's held-out test RMSE, recorded in the contract."""
    test = load_feature_config().get("test_metrics") or {}
    rmse = test.get("rmse")
    if rmse is None:
        raise ValueError("No baseline test RMSE in feature_config.json. Re-run select_best_model.py.")
    return float(rmse)


def evaluate_model(df: pd.DataFrame) -> dict:
    y_true = df[TARGET].values
    y_pred = predict(df)   # contract-driven: selects features + scales if required
    return {
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
    }


def detect_drift(current_rmse: float, baseline_rmse: float):
    degradation = (current_rmse - baseline_rmse) / baseline_rmse
    if degradation >= DRIFT_PCT:
        status = "🔴 DRIFT"
    elif degradation >= WARN_PCT:
        status = "🟡 WARNING"
    else:
        status = "🟢 NORMAL"
    return status, round(degradation * 100, 2)


def save_report(report: dict):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n[SUCCESS] Report saved → {REPORT_PATH}")


def main():
    print("\n================================================")
    print(" AQI MODEL DRIFT MONITOR ")
    print("================================================\n")

    cfg           = load_feature_config()
    df            = load_recent_batch()
    metrics       = evaluate_model(df)
    baseline_rmse = get_baseline_rmse()
    status, degradation_pct = detect_drift(metrics["rmse"], baseline_rmse)

    print("\n================================================")
    print(" MODEL PERFORMANCE ")
    print("================================================")
    print(f"Model          : {cfg['model_name']}")
    print(f"Baseline RMSE  : {baseline_rmse}")
    print(f"Current  RMSE  : {metrics['rmse']}")
    print(f"Current  MAE   : {metrics['mae']}")
    print(f"Current  R²    : {metrics['r2']}")
    print(f"RMSE degraded  : {degradation_pct}%")
    print(f"\n STATUS : {status}")
    print("================================================")

    report = {
        "model":                cfg["model_name"],
        "baseline_rmse":        baseline_rmse,
        "current_metrics":      metrics,
        "rmse_degradation_pct": degradation_pct,
        "status":               status,
    }
    save_report(report)
    print("\n[SUCCESS] Model drift monitoring complete.")


if __name__ == "__main__":
    main()
