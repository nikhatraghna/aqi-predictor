
import os
import pandas as pd
import requests
from datetime import datetime, timedelta

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

ISLAMABAD_LAT  = 33.6844
ISLAMABAD_LON  = 73.0479

# Open-Meteo free historical endpoint — no API key needed
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Last 12 months (Open-Meteo has ~5 day lag)
END_DT   = datetime.now() - timedelta(days=5)
START_DT = datetime(END_DT.year - 1, END_DT.month, END_DT.day)

OUTPUT_PATH = "data/raw/islamabad_historical_weather.parquet"


# ─────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────

def fetch_historical_weather(
    start: datetime = START_DT,
    end: datetime   = END_DT,
) -> pd.DataFrame:

    print(f"[INFO] Fetching historical weather (Open-Meteo): {start.date()} → {end.date()}")

    params = {
        "latitude"        : ISLAMABAD_LAT,
        "longitude"       : ISLAMABAD_LON,
        "start_date"      : start.strftime("%Y-%m-%d"),
        "end_date"        : end.strftime("%Y-%m-%d"),
        "hourly"          : [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover",
            "visibility",
            "weather_code",
        ],
        "timezone"        : "Asia/Karachi",
        "wind_speed_unit" : "ms",
    }

    response = requests.get(BASE_URL, params=params, timeout=60)

    if response.status_code != 200:
        raise Exception(f"[ERROR] Open-Meteo API failed: HTTP {response.status_code}\n{response.text}")

    raw = response.json()

    # ── Flatten hourly block into DataFrame ──
    hourly = raw.get("hourly", {})

    if not hourly:
        raise ValueError("[ERROR] No hourly data in response.")

    df = pd.DataFrame({
        "datetime"      : pd.to_datetime(hourly["time"]),
        "temperature"   : hourly["temperature_2m"],
        "humidity"      : hourly["relative_humidity_2m"],
        "precipitation" : hourly["precipitation"],
        "pressure"      : hourly["surface_pressure"],
        "wind_speed"    : hourly["wind_speed_10m"],
        "wind_direction": hourly["wind_direction_10m"],
        "cloud_cover"   : hourly["cloud_cover"],
        "visibility"    : hourly["visibility"],
        "weather_code"  : hourly["weather_code"],
    })

    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"[INFO] Records fetched : {len(df)}")
    print(f"[INFO] Date range      : {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"[INFO] Missing values  :\n{df.isnull().sum()}")

    return df


# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────

def save_historical_weather(df: pd.DataFrame, path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"\n[INFO] Saved → {path}")
    print(f"[INFO] File size: {os.path.getsize(path) / 1024:.1f} KB")


# ─────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_historical_weather()
    save_historical_weather(df)
    print("\n[SUCCESS] Historical weather backfill complete.")
    print(df.head(10))
