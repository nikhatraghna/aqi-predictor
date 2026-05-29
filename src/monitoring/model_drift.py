"""
Production AQI model drift monitoring.

This monitor performs TWO types of monitoring:

1. REAL-TIME DRIFT
   - Feature/data drift
   - Prediction distribution drift
   - PSI + KS tests

2. DELAYED PERFORMANCE EVALUATION
   - RMSE / MAE / RÂ² degradation
   - Requires actual PM2.5 labels

Additional capabilities:
  - Rolling parquet history
  - Hopsworks Feature Store logging
  - Consecutive drift tracking
  - Safe auto-retraining
  - Schema validation
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.inference.load_model import (
    load_best_model,
    load_scaler,
)

# =========================================================
# CONFIG
# =========================================================

FEATURES_PATH = "data/processed/islamabad_features.parquet"

REPORT_DIR = Path("reports/drift")

REPORT_JSON = REPORT_DIR / "model_drift_report.json"

HISTORY_PARQUET = (
    REPORT_DIR / "model_drift_history.parquet"
)

STATE_FILE = REPORT_DIR / "drift_state.json"

TARGET_COLUMN = "pm25"

FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "pm25_lag_1",
    "pm25_lag_3",
    "pm25_lag_6",
    "pm25_lag_24",
    "pm25_roll_mean_3",
    "pm25_roll_mean_24",
    "hour",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
]

# Performance thresholds
PERF_WARNING_PCT = 0.20
PERF_DRIFT_PCT = 0.40

# Distribution thresholds
KS_THRESHOLD = 0.05

PSI_WARNING = 0.10
PSI_DRIFT = 0.20

# Retraining safety
RETRAIN_AFTER_N_DRIFTS = 3

# Hopsworks
HOPSWORKS_FG_NAME = "model_drift_monitoring"
HOPSWORKS_FG_VERSION = 1


# =========================================================
# REGISTRY METRICS
# =========================================================

def get_best_model_metrics():
    """
    Load baseline metrics from saved registry JSON.
    """

    path = Path("models/best_model_metrics.json")

    if not path.exists():
        raise FileNotFoundError(
            "best_model_metrics.json not found"
        )

    with open(path, "r") as f:
        return json.load(f)


# =========================================================
# LOAD DATA
# =========================================================

def load_recent_batch(
    batch_size: int = 72,
):
    """
    Split dataset into:
      - train history
      - recent inference batch
    """

    print("\n[INFO] Loading engineered dataset...")

    df = pd.read_parquet(FEATURES_PATH)

    df = df.sort_values("datetime")

    inference_df = df.tail(batch_size).copy()

    train_df = df.iloc[:-batch_size].copy()

    print(f"[INFO] Train rows     : {len(train_df)}")
    print(f"[INFO] Inference rows : {len(inference_df)}")

    return train_df, inference_df


# =========================================================
# SCHEMA VALIDATION
# =========================================================

def validate_schema(df: pd.DataFrame):
    """
    Ensure inference batch matches training schema.
    """

    print("\n[INFO] Validating schema...")

    missing = []

    for col in FEATURE_COLUMNS:

        if col not in df.columns:
            missing.append(col)

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    # Null check
    null_counts = df[FEATURE_COLUMNS].isnull().sum()

    bad_nulls = null_counts[null_counts > 0]

    if len(bad_nulls) > 0:

        raise ValueError(
            f"Null values detected:\n{bad_nulls}"
        )

    # Infinite check
    inf_mask = np.isinf(
        df[FEATURE_COLUMNS]
    ).sum()

    if inf_mask.sum() > 0:
        raise ValueError(
            "Infinite values detected."
        )

    print("[SUCCESS] Schema validation passed.")


# =========================================================
# PERFORMANCE EVALUATION
# =========================================================

def evaluate_performance_drift(
    inference_df: pd.DataFrame,
):
    """
    DELAYED PERFORMANCE EVALUATION.

    IMPORTANT:
    This requires ACTUAL observed PM2.5 values.

    In real production systems this runs AFTER
    ground truth labels become available.
    """

    print("\n[INFO] Running performance evaluation...")

    baseline = get_best_model_metrics()

    model = load_best_model()

    scaler = load_scaler()

    X = inference_df[FEATURE_COLUMNS].copy()

    y_true = inference_df[TARGET_COLUMN].values

    X = scaler.transform(X)

    y_pred = model.predict(X)

    current_mae = float(
        mean_absolute_error(y_true, y_pred)
    )

    current_rmse = float(
        np.sqrt(mean_squared_error(y_true, y_pred))
    )

    current_r2 = float(
        r2_score(y_true, y_pred)
    )

    rmse_deg = (
        current_rmse - baseline["rmse"]
    ) / baseline["rmse"]

    mae_deg = (
        current_mae - baseline["mae"]
    ) / baseline["mae"]

    if rmse_deg >= PERF_DRIFT_PCT:
        status = "DRIFT"

    elif rmse_deg >= PERF_WARNING_PCT:
        status = "WARNING"

    else:
        status = "NORMAL"

    print(f"[INFO] RMSE degradation : {rmse_deg*100:.2f}%")
    print(f"[INFO] Status           : {status}")

    return {
        "current": {
            "mae": round(current_mae, 4),
            "rmse": round(current_rmse, 4),
            "r2": round(current_r2, 4),
        },
        "baseline": baseline,
        "rmse_degradation_pct": round(
            rmse_deg * 100,
            2,
        ),
        "mae_degradation_pct": round(
            mae_deg * 100,
            2,
        ),
        "status": status,
        "y_pred": y_pred,
    }


# =========================================================
# PSI
# =========================================================

def compute_psi(
    expected,
    actual,
    buckets=10,
):
    """
    Population Stability Index.
    """

    eps = 1e-6

    lo = min(expected.min(), actual.min())
    hi = max(expected.max(), actual.max())

    breakpoints = np.linspace(
        lo,
        hi,
        buckets + 1,
    )

    exp_counts = (
        np.histogram(expected, breakpoints)[0]
        + eps
    )

    act_counts = (
        np.histogram(actual, breakpoints)[0]
        + eps
    )

    exp_pct = exp_counts / exp_counts.sum()

    act_pct = act_counts / act_counts.sum()

    psi = np.sum(
        (act_pct - exp_pct)
        * np.log(act_pct / exp_pct)
    )

    return round(float(psi), 4)


# =========================================================
# DISTRIBUTION DRIFT
# =========================================================

def check_distribution_drift(
    train_df,
    inference_df,
):
    """
    REAL-TIME DRIFT CHECK.

    Uses:
      - KS test
      - PSI
      - Prediction statistics

    Does NOT require true labels.
    """

    print("\n[INFO] Running distribution drift...")

    model = load_best_model()

    scaler = load_scaler()

    # Training predictions
    X_train = train_df[
        FEATURE_COLUMNS
    ].tail(72)

    X_train = scaler.transform(X_train)

    train_preds = model.predict(X_train)

    # Inference predictions
    X_inf = inference_df[
        FEATURE_COLUMNS
    ].copy()

    X_inf = scaler.transform(X_inf)

    inf_preds = model.predict(X_inf)

    # KS test
    ks_stat, ks_pvalue = stats.ks_2samp(
        train_preds,
        inf_preds,
    )

    # PSI
    psi = compute_psi(
        train_preds,
        inf_preds,
    )

    # Prediction mean shift
    pred_mean_shift = abs(
        train_preds.mean()
        - inf_preds.mean()
    )

    # Status
    if (
        ks_pvalue < KS_THRESHOLD
        or psi >= PSI_DRIFT
    ):
        status = "DRIFT"

    elif psi >= PSI_WARNING:
        status = "WARNING"

    else:
        status = "NORMAL"

    print(f"[INFO] KS p-value  : {ks_pvalue:.4f}")
    print(f"[INFO] PSI         : {psi:.4f}")
    print(f"[INFO] Status      : {status}")

    return {
        "ks_stat": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_pvalue), 4),
        "psi": psi,
        "pred_mean_shift": round(
            float(pred_mean_shift),
            4,
        ),
        "status": status,
    }


# =========================================================
# AGGREGATE
# =========================================================

def aggregate_status(
    perf_status,
    dist_status,
):
    """
    Worst severity wins.
    """

    ranking = {
        "NORMAL": 0,
        "WARNING": 1,
        "DRIFT": 2,
    }

    return max(
        perf_status,
        dist_status,
        key=lambda x: ranking[x],
    )


# =========================================================
# DRIFT STATE
# =========================================================

def load_drift_state():
    """
    Load rolling drift counter.
    """

    if not STATE_FILE.exists():

        return {
            "consecutive_drift_runs": 0
        }

    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_drift_state(state):
    """
    Save drift counter.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)


# =========================================================
# REPORT
# =========================================================

def build_report(
    perf,
    dist,
    overall,
):
    """
    Build final report dictionary.
    """

    emoji = {
        "NORMAL": "🟢",
        "WARNING": "🟡",
        "DRIFT": "🔴",
    }

    return {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "overall_status": overall,

        "emoji": emoji[overall],

        "performance": {
            "current": perf["current"],
            "baseline": perf["baseline"],
            "rmse_degradation_pct":
                perf["rmse_degradation_pct"],
            "mae_degradation_pct":
                perf["mae_degradation_pct"],
            "status": perf["status"],
        },

        "distribution": dist,
    }


# =========================================================
# PRINT
# =========================================================

def print_report(report):
    """
    Pretty terminal report.
    """

    print("\n" + "=" * 60)

    print(
        f" MODEL DRIFT REPORT "
        f"{report['emoji']} "
        f"{report['overall_status']}"
    )

    print("=" * 60)

    perf = report["performance"]

    print("\nPerformance:")
    print(
        f"RMSE degradation : "
        f"{perf['rmse_degradation_pct']}%"
    )

    print(
        f"MAE degradation  : "
        f"{perf['mae_degradation_pct']}%"
    )

    print(
        f"Performance status : "
        f"{perf['status']}"
    )

    dist = report["distribution"]

    print("\nDistribution:")
    print(f"KS p-value : {dist['ks_pvalue']}")
    print(f"PSI        : {dist['psi']}")
    print(f"Status     : {dist['status']}")

    print("\n" + "=" * 60)


# =========================================================
# SAVE LOCAL
# =========================================================

def save_report_local(report):
    """
    Save JSON + append parquet history.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Latest JSON
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=4)

    print(
        f"\n[SUCCESS] Report saved â†’ "
        f"{REPORT_JSON}"
    )

    # Rolling history
    row = pd.DataFrame([{
        "timestamp": report["timestamp"],
        "overall_status":
            report["overall_status"],

        "rmse_degradation_pct":
            report["performance"][
                "rmse_degradation_pct"
            ],

        "psi":
            report["distribution"]["psi"],

        "ks_pvalue":
            report["distribution"]["ks_pvalue"],

        "perf_status":
            report["performance"]["status"],

        "dist_status":
            report["distribution"]["status"],
    }])

    if HISTORY_PARQUET.exists():

        history = pd.read_parquet(
            HISTORY_PARQUET
        )

        history = pd.concat(
            [history, row],
            ignore_index=True,
        )

    else:
        history = row

    history.to_parquet(
        HISTORY_PARQUET,
        index=False,
    )

    print(
        f"[SUCCESS] History updated â†’ "
        f"{HISTORY_PARQUET}"
    )


# =========================================================
# HOPSWORKS SAVE
# =========================================================

def save_report_hopsworks(report):
    """
    Upload report row to Hopsworks.
    """

    try:

        import os
        import hopsworks

        from dotenv import load_dotenv

        load_dotenv()

        api_key = os.getenv(
            "HOPSWORKS_API_KEY"
        )

        project_name = os.getenv(
            "HOPSWORKS_PROJECT"
        )

        if not api_key or not project_name:

            print(
                "[WARNING] Missing Hopsworks credentials."
            )

            return

        print(
            "\n[INFO] Uploading to Hopsworks..."
        )

        project = hopsworks.login(
            api_key_value=api_key,
            project=project_name,
        )

        fs = project.get_feature_store()

        # Updated row creation to match expected schema
        row = pd.DataFrame([{
            "timestamp": report["timestamp"],
            "overall_status": report["overall_status"],
            "rmse_current": report["performance"]["current"]["rmse"],
            "rmse_baseline": report["performance"]["baseline"]["rmse"],
            "rmse_degradation_pct": report["performance"]["rmse_degradation_pct"],
            "mae_current": report["performance"]["current"]["mae"],
            "r2_current": report["performance"]["current"]["r2"],
            "ks_stat": report["distribution"]["ks_stat"],
            "ks_pvalue": report["distribution"]["ks_pvalue"],
            "psi": report["distribution"]["psi"],
            "pred_mean_shift": report["distribution"]["pred_mean_shift"],
            "perf_status": report["performance"]["status"],
            "dist_status": report["distribution"]["status"],
        }])

        fg = fs.get_or_create_feature_group(
            name=HOPSWORKS_FG_NAME,
            version=HOPSWORKS_FG_VERSION,
            primary_key=["timestamp"],
            description="AQI model drift monitoring",
            online_enabled=False,
        )

        fg.insert(row)

        print(
            "[SUCCESS] Uploaded to Hopsworks."
        )

    except Exception as exc:

        print(
            f"[WARNING] Upload failed: {exc}"
        )


# =========================================================
# SAFE RETRAINING
# =========================================================

def trigger_retraining(report):
    """
    Retrain ONLY after N consecutive DRIFT runs.
    """

    state = load_drift_state()

    if report["overall_status"] == "DRIFT":

        state["consecutive_drift_runs"] += 1

    else:

        state["consecutive_drift_runs"] = 0

    save_drift_state(state)

    print(
        f"\n[INFO] Consecutive DRIFT runs : "
        f"{state['consecutive_drift_runs']}"
    )

    if (
        state["consecutive_drift_runs"]
        < RETRAIN_AFTER_N_DRIFTS
    ):

        print(
            "[INFO] Retraining threshold not reached."
        )

        return

    print(
        "\n[WARNING] Triggering retraining pipeline..."
    )

    try:

        subprocess.run(
            [
                "python",
                "-m",
                "src.training.train_and_register_pipeline",
            ],
            check=True,
        )

        print(
            "[SUCCESS] Retraining completed."
        )

        state["consecutive_drift_runs"] = 0

        save_drift_state(state)

    except subprocess.CalledProcessError as exc:

        print(
            f"[ERROR] Retraining failed: {exc}"
        )


# =========================================================
# MAIN
# =========================================================

def run_model_drift_monitoring(
    auto_retrain=True,
    upload_to_hopsworks=True,
):
    """
    Full monitoring pipeline.
    """

    print("\n" + "=" * 60)
    print(" AQI MODEL DRIFT MONITOR ")
    print("=" * 60)

    # Load
    train_df, inference_df = load_recent_batch()

    # Validate schema
    validate_schema(inference_df)

    # Performance evaluation
    perf = evaluate_performance_drift(
        inference_df
    )

    # Distribution drift
    dist = check_distribution_drift(
        train_df,
        inference_df,
    )

    # Aggregate
    overall = aggregate_status(
        perf["status"],
        dist["status"],
    )

    # Build report
    report = build_report(
        perf,
        dist,
        overall,
    )

    # Print
    print_report(report)

    # Save local
    save_report_local(report)

    # Upload
    if upload_to_hopsworks:
        save_report_hopsworks(report)

    # Safe retraining
    if auto_retrain:
        trigger_retraining(report)

    print(
        "\n[SUCCESS] Monitoring complete."
    )

    return report


if __name__ == "__main__":

    run_model_drift_monitoring(
        auto_retrain=True,
        upload_to_hopsworks=True,
    )