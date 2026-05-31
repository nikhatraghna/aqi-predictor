"""Run AQI predictions using the promoted production model + its inference contract."""

import pandas as pd

from src.inference.load_model import (
    load_best_model,
    load_scaler,
    load_feature_config,
)


def predict(X: pd.DataFrame):
    """Predict PM2.5 using the promoted model, honoring its feature_config.json contract."""
    config           = load_feature_config()
    features         = config["features"]
    requires_scaling = config.get("requires_scaling", False)
    model_name       = config["model_name"]

    model  = load_best_model()
    scaler = load_scaler()   # None unless the contract requires it

    # 1. Validate every trained feature is present in the input
    missing = [c for c in features if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required features for '{model_name}': {missing}")

    # 2. Subset + order columns EXACTLY as the model was trained (the bug-killer)
    X = X[features]

    # 3. Scale only if the contract says so (Ridge/Lasso); trees skip this
    if requires_scaling:
        if scaler is None:
            raise RuntimeError(f"'{model_name}' requires scaling but no scaler was loaded.")
        X = scaler.transform(X)

    predictions = model.predict(X)
    print(f"[INFO] Model: {model_name} | scaled: {requires_scaling} | "
          f"features: {len(features)} | predictions: {predictions.shape}")
    return predictions


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/islamabad_features.parquet")
    preds = predict(df.tail(24).copy())
    print("\nSample Predictions (last 24 hours):")
    for i, p in enumerate(preds):
        print(f"  Hour {i+1:>2}: PM2.5 = {p:.2f}")
    print("\n[SUCCESS] predict.py working.")
