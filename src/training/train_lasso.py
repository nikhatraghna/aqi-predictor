# src/training/train_lasso_advanced.py

import os
import json
import joblib
import pandas as pd
from pathlib import Path

from sklearn.linear_model import Lasso
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

TARGET_COLUMN = "pm25"
TEST_SIZE = 7 * 24


# ─────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────

from src.config.features import FEATURE_COLUMNS, TARGET_COLUMN

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]
# ─────────────────────────────────────
# SPLIT
# ─────────────────────────────────────

X_train = X[:-TEST_SIZE]
X_test = X[-TEST_SIZE:]

y_train = y[:-TEST_SIZE]
y_test = y[-TEST_SIZE:]

# ─────────────────────────────────────
# SCALING (important for Lasso)
# ─────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─────────────────────────────────────
# CV + GRID SEARCH
# ─────────────────────────────────────

tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1.0]
}

lasso = Lasso(max_iter=10000)

grid = GridSearchCV(
    lasso,
    param_grid,
    cv=tscv,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

print(f"[SUCCESS] Best alpha: {grid.best_params_['alpha']}")

# ─────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────

train_preds = best_model.predict(X_train_scaled)
test_preds = best_model.predict(X_test_scaled)

train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)

print("\n--- TRAIN R²:", train_r2)
print("--- TEST  R²:", test_r2)

print("\nOverfitting gap:", train_r2 - test_r2)

# ─────────────────────────────────────
# FEATURE SELECTION RESULT
# ─────────────────────────────────────

coef = pd.Series(best_model.coef_, index=X.columns)

selected = coef[coef != 0]

print("\nSelected Features:\n")
print(selected.sort_values(ascending=False))

# ─────────────────────────────────────
# SAVE
# ─────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(best_model, f"{MODEL_DIR}/lasso_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/lasso_scaler.pkl")

print("\n[SUCCESS] Lasso training complete.")
