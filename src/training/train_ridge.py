# src/training/train_ridge_advanced.py

import json
from pathlib import Path
import os
import joblib
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


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

MODEL_PATH = f"{MODEL_DIR}/ridge_model.pkl"
SCALER_PATH = f"{MODEL_DIR}/ridge_scaler.pkl"

TEST_SIZE = 7 * 24  # 7 days instead of 3


# ─────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────

print("\n[INFO] Loading data...")
df = pd.read_parquet(FEATURE_PATH)

# ─────────────────────────────────────
# FEATURES
# ─────────────────────────────────────

FEATURE_COLUMNS = [
    "temperature", "humidity", "pressure", "wind_speed",
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_24",
    "pm25_roll_mean_3", "pm25_roll_mean_24",
    "hour", "day_of_week", "month",
    "hour_sin", "hour_cos",
]

TARGET_COLUMN = "pm25"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# ─────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────

X_train = X[:-TEST_SIZE]
X_test = X[-TEST_SIZE:]

y_train = y[:-TEST_SIZE]
y_test = y[-TEST_SIZE:]

print(f"[INFO] Train shape: {X_train.shape}")
print(f"[INFO] Test shape : {X_test.shape}")

# ─────────────────────────────────────
# SCALING
# ─────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─────────────────────────────────────
# TIME SERIES CROSS-VALIDATION + TUNING
# ─────────────────────────────────────

print("\n[INFO] Running TimeSeries CV + GridSearch...")

tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    "alpha": [0.1, 1.0, 10.0, 50.0, 100.0]
}

ridge = Ridge()

grid = GridSearchCV(
    ridge,
    param_grid,
    cv=tscv,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

print(f"[SUCCESS] Best alpha: {grid.best_params_['alpha']}")

# ─────────────────────────────────────
# TRAIN PERFORMANCE (OVERFITTING CHECK)
# ─────────────────────────────────────

train_preds = best_model.predict(X_train_scaled)

train_mae = mean_absolute_error(y_train, train_preds)
train_rmse = mean_squared_error(y_train, train_preds) ** 0.5
train_r2 = r2_score(y_train, train_preds)

# ─────────────────────────────────────
# TEST PERFORMANCE
# ─────────────────────────────────────

test_preds = best_model.predict(X_test_scaled)

test_mae = mean_absolute_error(y_test, test_preds)
test_rmse = mean_squared_error(y_test, test_preds) ** 0.5
test_r2 = r2_score(y_test, test_preds)

# ─────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────

print("\n==============================")
print(" FINAL MODEL RESULTS")
print("==============================")

print("\n--- TRAIN ---")
print(f"MAE  : {train_mae:.2f}")
print(f"RMSE : {train_rmse:.2f}")
print(f"R²   : {train_r2:.4f}")

print("\n--- TEST ---")
print(f"MAE  : {test_mae:.2f}")
print(f"RMSE : {test_rmse:.2f}")
print(f"R²   : {test_r2:.4f}")

# Overfitting check
gap = train_r2 - test_r2
print(f"\n[INFO] Overfitting gap (R² diff): {gap:.4f}")

if gap > 0.1:
    print("[WARNING] Possible overfitting detected!")
else:
    print("[SUCCESS] Model generalizes well.")

# ─────────────────────────────────────
# SAVE METRICS
# ─────────────────────────────────────

metrics = {
    "train": {
        "mae": train_mae,
        "rmse": train_rmse,
        "r2": train_r2
    },
    "test": {
        "mae": test_mae,
        "rmse": test_rmse,
        "r2": test_r2
    },
    "best_alpha": grid.best_params_["alpha"]
}

save_metrics("ridge_advanced", metrics)

# ─────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\n[SUCCESS] Model saved → {MODEL_PATH}")
print(f"[SUCCESS] Scaler saved → {SCALER_PATH}")

# ─────────────────────────────────────
# PREVIEW
# ─────────────────────────────────────

results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": test_preds
})

print("\nPrediction Preview:\n")
print(results_df.head(10))

print("\n[SUCCESS] Advanced training pipeline complete.")
