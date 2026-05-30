"""Automatic feature selection for forecasting models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression

FEATURE_PATH = "data/processed/islamabad_features.parquet"
OUTPUT_PATH = "models/selected_features.json"
TARGET = "pm25"

# Features excluded from candidate predictors
DROP_COLS = [
    "datetime",
    "pm25",  # target

    # Current pollutant measurements
    # Removed for forecasting use case
    "pm10",
    "co",
    "no2",
    "so2",
    "o3",
    "aod",
    "dust",

    # Optional short-window features
    "pm25_roll_mean_3",
    "pm25_roll_std_3",
]


def load_features():
    """Load candidate features and target."""
    df = pd.read_parquet(FEATURE_PATH)

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[TARGET]

    X = X.select_dtypes(include=[np.number])

    mask = y.notna()
    return X.loc[mask], y.loc[mask]


def select_for_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 12,
) -> list:
    """Select Ridge features using f_regression."""

    print(f"\n[Ridge] Selecting top {k} features via f_regression...")

    selector = SelectKBest(
        score_func=f_regression,
        k=min(k, X.shape[1]),
    )

    selector.fit(X, y)

    selected = X.columns[selector.get_support()].tolist()

    # Always keep critical forecasting features
    must_have = [
        "pm25_lag_1",
        "pm25_lag_24",
        "pm25_roll_mean_24",
    ]

    for feature in must_have:
        if feature in X.columns and feature not in selected:
            selected.append(feature)

    print(f"[Ridge] Selected {len(selected)} features:")
    print(selected)

    return selected


def select_for_tree_models(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 15,
) -> list:
    """Select tree-model features using RF importance."""

    print(f"\n[Tree] Selecting top {k} features via RF importance...")

    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X, y)

    importances = pd.Series(
        rf.feature_importances_,
        index=X.columns,
    )

    selected = (
        importances
        .nlargest(min(k, len(importances)))
        .index
        .tolist()
    )

    print(f"[Tree] Selected {len(selected)} features:")
    print(selected)

    return selected


def select_for_prophet() -> list:
    """Prophet does not require engineered features."""
    return []


def run_feature_selection(
    ridge_k: int = 12,
    tree_k: int = 15,
):
    """Standalone feature-selection utility."""

    print("=" * 60)
    print("AUTOMATIC FEATURE SELECTION")
    print("=" * 60)

    X, y = load_features()

    print(
        f"\n[INFO] Dataset: {X.shape[0]} rows, "
        f"{X.shape[1]} candidate features"
    )

    selected = {
        "ridge": select_for_ridge(X, y, ridge_k),
        "random_forest": select_for_tree_models(X, y, tree_k),
        "xgboost": select_for_tree_models(X, y, tree_k),
        "prophet": [],
    }

    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(selected, f, indent=4)

    print(f"\n[SUCCESS] Saved → {OUTPUT_PATH}")

    return selected


def load_selected_features(model_name: str) -> list:
    """Load saved feature list."""

    path = Path(OUTPUT_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"{OUTPUT_PATH} not found."
        )

    with open(path) as f:
        selected = json.load(f)

    return selected[model_name]


if __name__ == "__main__":
    run_feature_selection()
