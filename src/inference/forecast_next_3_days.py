"""Generate a 72-hour (3-day) AQI forecast using the promoted production model."""

import pandas as pd
from src.inference.predict import predict

FEATURE_PATH = "data/processed/islamabad_features.parquet"

AQI_BREAKPOINTS = [
    (0,   12,   "Good",                  "🟢"),
    (12,  35,   "Moderate",              "🟡"),
    (35,  55,   "Unhealthy (Sensitive)", "🟠"),
    (55,  150,  "Unhealthy",             "🔴"),
    (150, 250,  "Very Unhealthy",        "🟣"),
    (250, 9999, "Hazardous",             "⛔"),
]


def pm25_to_category(pm25: float) -> tuple:
    """Map a PM2.5 value to its AQI category label and emoji."""
    for lo, hi, label, emoji in AQI_BREAKPOINTS:
        if lo <= pm25 < hi:
            return label, emoji
    return "Hazardous", "⛔"


def forecast_next_3_days() -> pd.DataFrame:
    """Predict PM2.5 for the most recent 72 feature rows.

    NOTE: uses the last 72 HISTORICAL rows (hindcast), not live future data.
    True forward forecasting is handled by hourly_feature_pipeline.py.
    """
    print("[INFO] Loading feature dataset...")
    df = pd.read_parquet(FEATURE_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    forecast_df = df.tail(72).copy()
    timestamps  = forecast_df["datetime"].values

    print("[INFO] Running 72-hour forecast...")
    predictions = predict(forecast_df)   # predict() selects its own feature subset

    results = []
    for i, (ts, pm25) in enumerate(zip(timestamps, predictions)):
        category, emoji = pm25_to_category(pm25)
        results.append({
            "hour":           i + 1,
            "datetime":       pd.Timestamp(ts),
            "predicted_pm25": round(float(pm25), 2),
            "category":       category,
            "status":         emoji,
        })
    return pd.DataFrame(results)


def print_forecast(forecast: pd.DataFrame) -> None:
    """Print a formatted 3-day forecast table."""
    print("\n" + "=" * 60)
    print("  3-DAY AQI FORECAST — ISLAMABAD")
    print("=" * 60)
    print(f"{'Hour':<6} {'Datetime':<20} {'PM2.5':>8}  Status")
    print("-" * 60)
    for day in range(3):
        day_rows = forecast[day * 24:(day + 1) * 24]
        print(f"\n📅 Day {day + 1}")
        for _, row in day_rows.iterrows():
            dt_str = row["datetime"].strftime("%Y-%m-%d %H:%M")
            print(f"{int(row['hour']):<6} {dt_str:<20} "
                  f"{row['predicted_pm25']:>8.2f}  {row['status']} {row['category']}")
    print("\n" + "=" * 60)
    avg = forecast["predicted_pm25"].mean()
    cat, emoji = pm25_to_category(avg)
    print(f"72h Avg PM2.5: {avg:.2f}  {emoji} {cat}")
    print("=" * 60)


if __name__ == "__main__":
    forecast = forecast_next_3_days()
    print_forecast(forecast)
    forecast.to_parquet("data/processed/forecast_3days.parquet", index=False)
    print("\n[SUCCESS] Forecast saved → data/processed/forecast_3days.parquet")
