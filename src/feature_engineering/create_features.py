"""Feature engineering pipeline for PM2.5 prediction."""

import os
import sys

sys.path.append(os.path.abspath("../.."))
import pandas as pd
import numpy as np

from src.data_pipeline.preprocess import preprocess_data
from src.data_pipeline.validate_data import validate_data

WEATHER_PATH = "data/raw/islamabad_historical_weather.parquet"
AQI_PATH     = "data/raw/islamabad_historical_air_quality.parquet"
OUTPUT_PATH  = "data/processed/islamabad_features.parquet"


def load_data() -> tuple:
    """Load raw weather and AQI parquet files.

    Returns:
        Tuple of (weather_df, aqi_df).
    """
    print("\n[INFO] Loading datasets...")
    weather_df = pd.read_parquet(WEATHER_PATH)
    aqi_df     = pd.read_parquet(AQI_PATH)
    print(f"[INFO] Weather shape : {weather_df.shape}")
    print(f"[INFO] AQI shape     : {aqi_df.shape}")
    return weather_df, aqi_df


def merge_datasets(weather_df: pd.DataFrame, aqi_df: pd.DataFrame) -> pd.DataFrame:
    """Merge AQI and weather on datetime.

    Args:
        weather_df: Weather DataFrame.
        aqi_df:     AQI DataFrame.

    Returns:
        Merged DataFrame sorted by datetime.
    """
    print("\n[INFO] Merging datasets...")
    df = pd.merge(aqi_df, weather_df, on="datetime", how="inner")
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"[INFO] Merged shape  : {df.shape}")
    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features for pm25.

    Uses .shift(n) so each lag only contains PAST values.
    lag_1 = pm25 one hour ago — safe at prediction time.

    Args:
        df: Input DataFrame sorted by datetime.

    Returns:
        DataFrame with lag columns added.
    """
    print("\n[INFO] Creating lag features...")
    for lag in [1, 3, 6, 12, 24]:
        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)  # ✅ safe — pure past values
    print("[INFO] Lag features created.")
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling mean features for pm25.

    CRITICAL: .shift(1) is applied BEFORE .rolling() so the
    current row's pm25 value is never included in the window.
    Without shift(1), the rolling mean leaks the current target.

    Only mean features are kept — std features were removed
    because they had low importance and contributed to overfitting.

    Args:
        df: Input DataFrame sorted by datetime.

    Returns:
        DataFrame with rolling mean columns added.
    """
    print("\n[INFO] Creating rolling features...")
    for window in [6, 12, 24]:
        df[f"pm25_roll_mean_{window}"] = (
            df["pm25"]
            .shift(1)           # ✅ exclude current row before rolling
            .rolling(window=window)
            .mean()
        )
    # Note: roll_mean_3, roll_std_* removed — caused leakage/overfitting
    print("[INFO] Rolling features created.")
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from datetime column.

    Args:
        df: Input DataFrame with datetime column.

    Returns:
        DataFrame with time feature columns added.
    """
    print("\n[INFO] Creating time features...")
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"]       = df["datetime"].dt.month
    df["day"]         = df["datetime"].dt.day

    # Cyclical encoding — prevents model treating 23→0 as a big jump
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    print("[INFO] Time features created.")
    return df


def final_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any NaN values (from lag/rolling warm-up period)."""

    print("\n[INFO] Final cleaning...")

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)

    print(f"[INFO] Removed {before - after} warm-up rows (lag/rolling NaNs)")
    print(f"[INFO] Final shape   : {df.shape}")

    return df


def save_features(df: pd.DataFrame) -> None:
    """Save feature dataset to parquet.

    Args:
        df: Final feature DataFrame.
    """
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"[INFO] Saved → {OUTPUT_PATH}")
    print(f"[INFO] File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")


if __name__ == "__main__":

    print("\n==============================")
    print(" AQI FEATURE ENGINEERING")
    print("==============================")

    weather_df, aqi_df = load_data()

    weather_df = preprocess_data(weather_df)
    aqi_df     = preprocess_data(aqi_df)

    weather_df = validate_data(weather_df)
    aqi_df     = validate_data(aqi_df)

    df = merge_datasets(weather_df, aqi_df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_time_features(df)
    df = final_clean(df)
    save_features(df)

    print("\n[SUCCESS] Feature engineering pipeline completed.")
    print("\n[INFO] Final Dataset Preview:\n")
    print(df.head())
