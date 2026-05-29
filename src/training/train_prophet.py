# src/training/train_prophet_advanced.py

import os
import json
import joblib
import pandas as pd
from pathlib import Path

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─────────────────────────────────────
# SAVE METRICS
# ─────────────────────────────────────

def save_metrics(name, metrics_dict):
    path = Path("models/metrics")
    path.mkdir(parents=True, exist_ok=True)

    with open(path / f"{name}.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"[SUCCESS] Metrics saved → {path / f'{name}.json'}")


# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────

FEATURE_PATH = "data/processed/islamabad_features.parquet"
MODEL_DIR = "models/saved_models"

TEST_SIZE = 7 * 24  # 7 days


# ─────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────

print("\n[INFO] Loading data...")
df = pd.read_parquet(FEATURE_PATH)

# ─────────────────────────────────────
# PREPARE PROPHET FORMAT
# ─────────────────────────────────────

print("\n[INFO] Preparing Prophet dataset...")

prophet_df = pd.DataFrame({
    "ds": pd.to_datetime(df["datetime"]).dt.tz_localize(None),
    "y": df["pm25"],
})

# ─────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────

train_df = prophet_df[:-TEST_SIZE]
test_df = prophet_df[-TEST_SIZE:]

print(f"[INFO] Train shape: {train_df.shape}")
print(f"[INFO] Test shape : {test_df.shape}")

# ─────────────────────────────────────
# MODEL (TUNED SETTINGS)
# ─────────────────────────────────────

print("\n[INFO] Training Prophet model...")

model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False,
    changepoint_prior_scale=0.1,   # controls flexibility
    seasonality_prior_scale=5.0    # controls seasonality strength
)

model.fit(train_df)

print("[SUCCESS] Prophet trained.")

# ─────────────────────────────────────
# FORECAST
# ─────────────────────────────────────

future = model.make_future_dataframe(
    periods=TEST_SIZE,
    freq="H",
)

forecast = model.predict(future)

predictions = forecast["yhat"].tail(TEST_SIZE).values
actual = test_df["y"].values

# ─────────────────────────────────────
# TRAIN PERFORMANCE (approx)
# ─────────────────────────────────────

train_preds = forecast["yhat"].iloc[:len(train_df)].values
train_actual = train_df["y"].values

train_r2 = r2_score(train_actual, train_preds)

# ─────────────────────────────────────
# TEST PERFORMANCE
# ─────────────────────────────────────

test_mae = mean_absolute_error(actual, predictions)
test_rmse = mean_squared_error(actual, predictions) ** 0.5
test_r2 = r2_score(actual, predictions)

# ─────────────────────────────────────
# RESULTS
# ─────────────────────────────────────

print("\n==============================")
print(" PROPHET FINAL RESULTS")
print("==============================")

print("\n--- TRAIN ---")
print(f"R²   : {train_r2:.4f}")

print("\n--- TEST ---")
print(f"MAE  : {test_mae:.2f}")
print(f"RMSE : {test_rmse:.2f}")
print(f"R²   : {test_r2:.4f}")

gap = train_r2 - test_r2
print(f"\n[INFO] Overfitting gap: {gap:.4f}")

if gap > 0.1:
    print("[WARNING] Possible overfitting detected!")
else:
    print("[INFO] No major overfitting detected.")

# ─────────────────────────────────────
# SAVE METRICS
# ─────────────────────────────────────

metrics = {
    "train": {
        "r2": train_r2
    },
    "test": {
        "mae": test_mae,
        "rmse": test_rmse,
        "r2": test_r2
    }
}

save_metrics("prophet_advanced", metrics)

# ─────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = f"{MODEL_DIR}/prophet_model.pkl"
joblib.dump(model, MODEL_PATH)

print(f"[SUCCESS] Model saved → {MODEL_PATH}")

# ─────────────────────────────────────
# PREVIEW
# ─────────────────────────────────────

preview = pd.DataFrame({
    "Actual": actual,
    "Predicted": predictions
})

print("\nPrediction Preview:\n")
print(preview.head(10))

print("\n[SUCCESS] Advanced Prophet pipeline complete.")
