
import os
import sys

# Add project root to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

import pandas as pd
import numpy as np

from src.data_pipeline.preprocess import preprocess_data
from src.data_pipeline.validate_data import validate_data

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────

WEATHER_PATH = "data/raw/islamabad_historical_weather.parquet"

AQI_PATH = "data/raw/islamabad_historical_air_quality.parquet"

OUTPUT_PATH = "data/processed/islamabad_features.parquet"


# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────

def load_data():

    print("\n[INFO] Loading datasets...\n")

    weather_df = pd.read_parquet(WEATHER_PATH)

    aqi_df = pd.read_parquet(AQI_PATH)

    print(f"[INFO] Weather shape: {weather_df.shape}")
    print(f"[INFO] AQI shape    : {aqi_df.shape}")

    return weather_df, aqi_df


# ─────────────────────────────────────────
# MERGE DATASETS
# ─────────────────────────────────────────

def merge_datasets(weather_df, aqi_df):

    print("\n[INFO] Merging datasets...\n")

    df = pd.merge(
        aqi_df,
        weather_df,
        on="datetime",
        how="inner"
    )

    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"[INFO] Merged shape: {df.shape}")

    return df


# ─────────────────────────────────────────
# LAG FEATURES
# ─────────────────────────────────────────

def create_lag_features(df):

    print("\n[INFO] Creating lag features...\n")

    lags = [1, 3, 6, 12, 24]

    for lag in lags:

        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)

    print("[INFO] Lag features created.")

    return df


# ─────────────────────────────────────────
# ROLLING WINDOW FEATURES
# ─────────────────────────────────────────

def create_rolling_features(df):

    print("\n[INFO] Creating rolling features...\n")

    windows = [3, 6, 12, 24]

    for window in windows:

        df[f"pm25_roll_mean_{window}"] = (
            df["pm25"]
            .rolling(window=window)
            .mean()
        )

        df[f"pm25_roll_std_{window}"] = (
            df["pm25"]
            .rolling(window=window)
            .std()
        )

    print("[INFO] Rolling features created.")

    return df


# ─────────────────────────────────────────
# TIME FEATURES
# ─────────────────────────────────────────

def create_time_features(df):

    print("\n[INFO] Creating time features...\n")

    df["hour"] = df["datetime"].dt.hour

    df["day_of_week"] = df["datetime"].dt.dayofweek

    df["month"] = df["datetime"].dt.month

    df["day"] = df["datetime"].dt.day

    # Cyclical Encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)

    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    print("[INFO] Time features created.")

    return df


# ─────────────────────────────────────────
# FINAL CLEANING
# ─────────────────────────────────────────

def final_clean(df):

    print("\n[INFO] Final cleaning...\n")

    before = len(df)

    df = df.dropna().reset_index(drop=True)

    after = len(df)

    print(f"[INFO] Removed rows with NaNs: {before - after}")

    print(f"[INFO] Final dataset shape: {df.shape}")

    return df


# ─────────────────────────────────────────
# SAVE FEATURES
# ─────────────────────────────────────────

def save_features(df):

    print("\n[INFO] Saving feature dataset...\n")

    os.makedirs("data/processed", exist_ok=True)

    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"[INFO] Saved → {OUTPUT_PATH}")

    print(f"[INFO] File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")


# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("\n==============================")
    print(" AQI FEATURE ENGINEERING")
    print("==============================")

    # Load datasets
    weather_df, aqi_df = load_data()

    # Preprocess datasets
    weather_df = preprocess_data(weather_df)

    aqi_df = preprocess_data(aqi_df)

    # Validate datasets
    weather_df = validate_data(weather_df)

    aqi_df = validate_data(aqi_df)

    # Merge
    df = merge_datasets(weather_df, aqi_df)

    # Feature Engineering
    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = create_time_features(df)

    # Final Cleaning
    df = final_clean(df)

    # Save
    save_features(df)

    print("\n[SUCCESS] Feature engineering pipeline completed.")

    print("\n[INFO] Final Dataset Preview:\n")

    print(df.head())
