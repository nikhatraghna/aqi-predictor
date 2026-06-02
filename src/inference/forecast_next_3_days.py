"""Backtest: model predictions vs ACTUAL PM2.5 over the most recent 72 observed hours.

A model-validation (hindcast) view, distinct from the forward forecast in forecast_future.py.
"""
import pandas as pd
from src.inference.predict import predict

FEATURE_PATH = "data/processed/islamabad_features.parquet"
OUT_PATH     = "data/processed/hindcast_3days.parquet"


def main():
    df = pd.read_parquet(FEATURE_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    recent = df.tail(72).copy()
    preds = predict(recent)
    out = pd.DataFrame({
        "datetime":       pd.to_datetime(recent["datetime"].values),
        "actual_pm25":    recent["pm25"].round(2).values,
        "predicted_pm25": [round(float(p), 2) for p in preds],
    })
    out.to_parquet(OUT_PATH, index=False)
    mae = (out["actual_pm25"] - out["predicted_pm25"]).abs().mean()
    print(f"[SUCCESS] Backtest → {OUT_PATH}  ({len(out)} hours, MAE={mae:.2f} µg/m³)")


if __name__ == "__main__":
    main()
