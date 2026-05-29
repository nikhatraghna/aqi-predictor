"""Run AQI predictions using the best production model."""

import pandas as pd
from src.inference.load_model import (
    load_best_model,
    load_scaler,
    get_best_model_name,
)

RIDGE_FEATURE_COLS = [
    "temperature", "humidity", "pressure", "wind_speed",
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_24",
    "pm25_roll_mean_3", "pm25_roll_mean_24",
    "hour", "day_of_week", "month", "hour_sin", "hour_cos",
]

DROP_COLS = ["datetime", "pm25"]


def predict(X: pd.DataFrame):
    model_name = get_best_model_name()
    model      = load_best_model()
    scaler     = load_scaler()  # None unless Ridge

    X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])

    if model_name == "ridge":
        missing = [c for c in RIDGE_FEATURE_COLS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns for Ridge: {missing}")
        X = X[RIDGE_FEATURE_COLS]
        if scaler is not None:
            X = scaler.transform(X)
        else:
            raise RuntimeError("Ridge selected but scaler is missing.")

    # XGBoost / RandomForest / Prophet — no scaling, use all feature cols
    predictions = model.predict(X)
    print(f"[INFO] Model     : {model_name}")
    print(f"[INFO] Scaled    : {model_name == 'ridge'}")
    print(f"[INFO] Predictions shape: {predictions.shape}")
    return predictions


if __name__ == "__main__":
    df    = pd.read_parquet("data/processed/islamabad_features.parquet")
    preds = predict(df.tail(24).copy())
    print("\nSample Predictions (last 24 hours):")
    for i, p in enumerate(preds):
        print(f"  Hour {i+1:>2}: PM2.5 = {p:.2f}")
    print("\n[SUCCESS] predict.py working.")
