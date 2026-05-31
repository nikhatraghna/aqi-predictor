"""Hourly feature pipeline for Islamabad PM2.5 (Open-Meteo only — no API key).

Steps:
  1. Load data/processed/islamabad_features.parquet.
  2. Fetch the latest hour of weather + air quality from Open-Meteo (past_days=1, forecast_days=1).
  3. Keep the single most recent hour not already in the parquet (no <= now cap).
  4. Append the new raw row, recompute lag/rolling/time features on the FULL frame.
  5. Save the parquet, then upload ONLY the new row to Hopsworks aqi_features v1.
  6. If the latest hour already exists, skip upload and exit cleanly.

Hopsworks credentials come from env vars HOPSWORKS_API_KEY / HOPSWORKS_PROJECT.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ISLAMABAD_LAT, ISLAMABAD_LON = 33.6844, 73.0479
FEATURES_PATH = Path("data/processed/islamabad_features.parquet")

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AQ_URL      = "https://air-quality-api.open-meteo.com/v1/air-quality"

HOPSWORKS_FG_NAME    = "aqi_features"
HOPSWORKS_FG_VERSION = 1

DERIVED_PREFIXES = ("pm25_lag_", "pm25_roll_mean_", "pm25_roll_std_")
TIME_COLS        = {"hour", "day_of_week", "month", "day", "hour_sin", "hour_cos"}


# ── Fetch ────────────────────────────────────────────────────────────────────
def fetch_open_meteo() -> pd.DataFrame:
    """Fetch recent weather + air quality and merge on datetime."""
    w_params = {
        "latitude": ISLAMABAD_LAT, "longitude": ISLAMABAD_LON,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation",
                   "surface_pressure", "wind_speed_10m", "wind_direction_10m",
                   "cloud_cover", "visibility", "weather_code"],
        "timezone": "Asia/Karachi", "wind_speed_unit": "ms",
        "past_days": 1, "forecast_days": 1,
    }
    a_params = {
        "latitude": ISLAMABAD_LAT, "longitude": ISLAMABAD_LON,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                   "sulphur_dioxide", "ozone", "aerosol_optical_depth", "dust"],
        "timezone": "Asia/Karachi", "past_days": 1, "forecast_days": 1,
    }
    wr = requests.get(WEATHER_URL, params=w_params, timeout=60); wr.raise_for_status()
    ar = requests.get(AQ_URL,      params=a_params, timeout=60); ar.raise_for_status()
    wh, ah = wr.json()["hourly"], ar.json()["hourly"]

    df_w = pd.DataFrame({
        "datetime":       pd.to_datetime(wh["time"]),
        "temperature":    wh["temperature_2m"],
        "humidity":       wh["relative_humidity_2m"],
        "precipitation":  wh["precipitation"],
        "pressure":       wh["surface_pressure"],
        "wind_speed":     wh["wind_speed_10m"],
        "wind_direction": wh["wind_direction_10m"],
        "cloud_cover":    wh["cloud_cover"],
        "visibility":     wh["visibility"],
        "weather_code":   wh["weather_code"],
    })
    df_a = pd.DataFrame({
        "datetime": pd.to_datetime(ah["time"]),
        "pm25":     ah["pm2_5"],
        "pm10":     ah["pm10"],
        "co":       ah["carbon_monoxide"],
        "no2":      ah["nitrogen_dioxide"],
        "so2":      ah["sulphur_dioxide"],
        "o3":       ah["ozone"],
        "aod":      ah["aerosol_optical_depth"],
        "dust":     ah["dust"],
    })

    merged = pd.merge(df_a, df_w, on="datetime", how="inner")
    # Normalize to UTC-aware to match parquet convention
    merged["datetime"] = pd.to_datetime(merged["datetime"]).dt.tz_localize("UTC")
    return merged


# ── Feature recompute (schema-introspecting) ────────────────────────────────
def _specs(columns):
    """Read the lag/roll feature specs from the existing column names."""
    lags  = sorted(int(c.rsplit("_", 1)[-1]) for c in columns if c.startswith("pm25_lag_"))
    means = sorted(int(c.rsplit("_", 1)[-1]) for c in columns if c.startswith("pm25_roll_mean_"))
    stds  = sorted(int(c.rsplit("_", 1)[-1]) for c in columns if c.startswith("pm25_roll_std_"))
    return lags, means, stds


def recompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute lag/rolling/time features on the full frame, matching existing columns."""
    # Normalize all datetimes to UTC-aware before sorting
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    lags, means, stds = _specs(df.columns)

    for lag in lags:
        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)
    for w in means:
        df[f"pm25_roll_mean_{w}"] = df["pm25"].rolling(w, min_periods=1).mean()
    for w in stds:
        df[f"pm25_roll_std_{w}"] = df["pm25"].rolling(w, min_periods=1).std().fillna(0.0)

    dt = pd.to_datetime(df["datetime"], utc=True)
    df["hour"]        = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"]       = dt.dt.month
    df["day"]         = dt.dt.day
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def build_new_base_row(existing: pd.DataFrame, fetched_row: pd.Series, new_dt) -> dict:
    """Map a fetched hour onto the existing base (raw) columns; carry forward any gaps."""
    last = existing.iloc[-1]
    base_cols = [c for c in existing.columns
                 if not c.startswith(DERIVED_PREFIXES) and c not in TIME_COLS]
    row = {}
    for col in base_cols:
        if col == "datetime":
            row[col] = new_dt
        elif col in fetched_row.index and pd.notna(fetched_row[col]):
            row[col] = fetched_row[col]
        else:
            row[col] = last[col]
    return row


# ── Hopsworks upload (new row only) ─────────────────────────────────────────
def upload_new_row(new_row_df: pd.DataFrame) -> None:
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project = os.getenv("HOPSWORKS_PROJECT")
    if not api_key or not project:
        print("[WARNING] HOPSWORKS_API_KEY / HOPSWORKS_PROJECT not set — skipping upload.")
        return
    try:
        import hopsworks
        proj = hopsworks.login(api_key_value=api_key, project=project)
        fs   = proj.get_feature_store()
        fg   = fs.get_feature_group(name=HOPSWORKS_FG_NAME, version=HOPSWORKS_FG_VERSION)
        fg.insert(new_row_df)
        print(f"[SUCCESS] Uploaded 1 new row → Hopsworks {HOPSWORKS_FG_NAME} v{HOPSWORKS_FG_VERSION}.")
    except Exception as exc:
        print(f"[WARNING] Hopsworks upload failed (local parquet is still updated): {exc}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n==============================")
    print(" HOURLY FEATURE PIPELINE (Open-Meteo)")
    print("==============================")

    ensure_features_exist()
    existing = pd.read_parquet(FEATURES_PATH)
    existing["datetime"] = pd.to_datetime(existing["datetime"], utc=True)
    existing = existing.sort_values("datetime").reset_index(drop=True)

    print("[INFO] Fetching latest Open-Meteo weather + air quality...")
    fetched = fetch_open_meteo()

    # Ingest most recent API hour not already stored
    seen = set(existing["datetime"])
    candidates = fetched[~fetched["datetime"].isin(seen)]
    if candidates.empty:
        print(f"[INFO] Parquet already up to date (latest {existing['datetime'].max()}). "
              "No new hour — skipping.")
        return

    new_row_src = candidates.sort_values("datetime").iloc[-1]
    new_dt      = new_row_src["datetime"]
    print(f"[INFO] New hour to add: {new_dt}")

    # Append raw row, then recompute features on the full frame
    base     = build_new_base_row(existing, new_row_src, new_dt)
    combined = pd.concat([existing, pd.DataFrame([base])], ignore_index=True)
    combined = recompute_features(combined)
    combined = combined.reindex(columns=existing.columns)

    combined.to_parquet(FEATURES_PATH, index=False)
    print(f"[SUCCESS] Features updated → {FEATURES_PATH} "
          f"({len(combined)} rows, latest {combined['datetime'].max()})")

    # Upload only the new row
    new_row_df = combined[combined["datetime"] == new_dt].copy()
    upload_new_row(new_row_df)

    print("\n[SUCCESS] Hourly feature pipeline complete.")
def ensure_features_exist():
    """If the local feature parquet is missing (e.g., fresh CI runner), restore from Hopsworks."""
    if FEATURES_PATH.exists():
        return
    import os, hopsworks
    api, proj = os.getenv("HOPSWORKS_API_KEY"), os.getenv("HOPSWORKS_PROJECT")
    if not api or not proj:
        raise FileNotFoundError(f"{FEATURES_PATH} missing and no Hopsworks creds to restore from.")
    print("[INFO] Local features missing — restoring from Hopsworks feature group...")
    fg = hopsworks.login(api_key_value=api, project=proj).get_feature_store() \
                  .get_feature_group(name="aqi_features", version=1)
    df = fg.read()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.drop_duplicates("datetime", keep="last").sort_values("datetime").reset_index(drop=True)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH, index=False)
    print(f"[INFO] Restored {len(df)} rows from Hopsworks.")

if __name__ == "__main__":
    main()
