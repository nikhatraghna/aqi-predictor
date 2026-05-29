# src/training/train_random_forest_advanced.py

import os
import json
import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
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

TARGET_COLUMN = "pm25"
TEST_SIZE = 7 * 24  # 7 days


# ─────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────

print("\n[INFO] Loading data...")
df = pd.read_parquet(FEATURE_PATH)

# ─────────────────────────────────────
# FEATURES
# ─────────────────────────────────────

DROP_COLUMNS = ["datetime", "pm25"]

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET_COLUMN]

print(f"[INFO] Features shape: {X.shape}")

# ─────────────────────────────────────
# TRAIN / TEST SPLIT (TIME-BASED)
# ─────────────────────────────────────

X_train = X[:-TEST_SIZE]
X_test = X[-TEST_SIZE:]

y_train = y[:-TEST_SIZE]
y_test = y[-TEST_SIZE:]

print(f"[INFO] Train shape: {X_train.shape}")
print(f"[INFO] Test shape : {X_test.shape}")

# ─────────────────────────────────────
# TIME SERIES CV + GRID SEARCH
# ─────────────────────────────────────

print("\n[INFO] Running TimeSeries CV + GridSearch...")

tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [8, 12, 16],
    "min_samples_split": [2, 5],
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=tscv,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print(f"[SUCCESS] Best params: {grid.best_params_}")

# ─────────────────────────────────────
# TRAIN PERFORMANCE
# ─────────────────────────────────────

train_preds = best_model.predict(X_train)

train_mae = mean_absolute_error(y_train, train_preds)
train_rmse = mean_squared_error(y_train, train_preds) ** 0.5
train_r2 = r2_score(y_train, train_preds)

# ─────────────────────────────────────
# TEST PERFORMANCE
# ─────────────────────────────────────

test_preds = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_preds)
test_rmse = mean_squared_error(y_test, test_preds) ** 0.5
test_r2 = r2_score(y_test, test_preds)

# ─────────────────────────────────────
# RESULTS
# ─────────────────────────────────────

print("\n==============================")
print(" RANDOM FOREST FINAL RESULTS")
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
print(f"\n[INFO] Overfitting gap: {gap:.4f}")

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
    "best_params": grid.best_params_
}

save_metrics("random_forest_advanced", metrics)

# ─────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = f"{MODEL_DIR}/random_forest_model.pkl"
joblib.dump(best_model, MODEL_PATH)

print(f"[SUCCESS] Model saved → {MODEL_PATH}")

# ─────────────────────────────────────
# FEATURE IMPORTANCE (BIG ADVANTAGE)
# ─────────────────────────────────────

importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop Features:\n")
print(importances.head(10))

# ─────────────────────────────────────
# PREVIEW
# ─────────────────────────────────────

preview = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": test_preds
})

print("\nPrediction Preview:\n")
print(preview.head(10))

print("\n[SUCCESS] Advanced Random Forest pipeline complete.")
