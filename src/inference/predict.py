import pandas as pd
from src.inference.load_model import load_best_model, load_scaler
from src.models.model_registry import get_best_model_name

# Exact columns Ridge was trained on — must match train_ridge.py
RIDGE_FEATURE_COLS = [
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

DROP_COLS = ["datetime", "pm25"]


def predict(X: pd.DataFrame):
    """Run prediction on a feature DataFrame.

    Selects only the columns the model was trained on.
    Applies scaler only when best model is Ridge.

    Args:
        X: Feature DataFrame (any columns — will be filtered).

    Returns:
        NumPy array of predicted PM2.5 values.
    """
    model_name = get_best_model_name()
    model      = load_best_model()
    scaler     = load_scaler()

    # Drop target and datetime if present
    X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])

    # For Ridge: select exact training columns
    if model_name == "ridge":
        missing = [c for c in RIDGE_FEATURE_COLS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns for Ridge: {missing}")
        X = X[RIDGE_FEATURE_COLS]
        if scaler is not None:
            X = scaler.transform(X)

    predictions = model.predict(X)
    print(f"[INFO] Predictions shape: {predictions.shape}")
    return predictions


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/islamabad_features.parquet")
    preds = predict(df.tail(24).copy())
    print("\nSample Predictions (last 24 hours):")
    for i, p in enumerate(preds):
        print(f"  Hour {i+1:>2}: PM2.5 = {p:.2f}")
    print("\n[SUCCESS] predict.py working.")
