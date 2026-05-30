import os
import pandas as pd
import requests

from datetime import datetime, timedelta


# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

ISLAMABAD_LAT = 33.6844
ISLAMABAD_LON = 73.0479

BASE_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

# Open-Meteo historical lag
END_DT = datetime.now() - timedelta(days=5)

# Last 3 months
START_DT = END_DT - timedelta(days=90)
OUTPUT_PATH = (
    "data/raw/"
    "islamabad_historical_air_quality.parquet"
)


# ─────────────────────────────────────────
# FETCH HISTORICAL AIR QUALITY
# ─────────────────────────────────────────

def fetch_historical_air_quality(
    start: datetime = START_DT,
    end: datetime = END_DT,
) -> pd.DataFrame:

    print(
        f"[INFO] Fetching historical air quality: "
        f"{start.date()} → {end.date()}"
    )

    params = {

        "latitude": ISLAMABAD_LAT,

        "longitude": ISLAMABAD_LON,

        "start_date": start.strftime("%Y-%m-%d"),

        "end_date": end.strftime("%Y-%m-%d"),

        "hourly": [

            "pm10",

            "pm2_5",

            "carbon_monoxide",

            "nitrogen_dioxide",

            "sulphur_dioxide",

            "ozone",

            "aerosol_optical_depth",

            "dust",
        ],

        "timezone": "Asia/Karachi",
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=60
    )

    if response.status_code != 200:

        raise Exception(
            f"[ERROR] Open-Meteo AQ API failed: "
            f"HTTP {response.status_code}\n"
            f"{response.text}"
        )

    raw = response.json()

    hourly = raw.get("hourly", {})

    if not hourly:
        raise ValueError(
            "[ERROR] No hourly AQ data returned."
        )

    df = pd.DataFrame({

        "datetime": pd.to_datetime(hourly["time"]),

        "pm25": hourly["pm2_5"],

        "pm10": hourly["pm10"],

        "co": hourly["carbon_monoxide"],

        "no2": hourly["nitrogen_dioxide"],

        "so2": hourly["sulphur_dioxide"],

        "o3": hourly["ozone"],

        "aod": hourly["aerosol_optical_depth"],

        "dust": hourly["dust"],
    })

    # Sort chronologically
    df = df.sort_values("datetime")

    df = df.reset_index(drop=True)

    print(f"[INFO] Records fetched : {len(df)}")

    print(
        f"[INFO] Date range      : "
        f"{df['datetime'].min()} → "
        f"{df['datetime'].max()}"
    )

    print(
        f"[INFO] Missing values  :\n"
        f"{df.isnull().sum()}"
    )

    return df


# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────

def save_air_quality(
    df: pd.DataFrame,
    path: str = OUTPUT_PATH,
) -> None:

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    df.to_parquet(path, index=False)

    print(f"\n[INFO] Saved → {path}")

    print(
        f"[INFO] File size: "
        f"{os.path.getsize(path) / 1024:.1f} KB"
    )


# ─────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────

if __name__ == "__main__":

    df = fetch_historical_air_quality()

    save_air_quality(df)

    print(
        "\n[SUCCESS] Historical AQ backfill complete."
    )

    print(df.head(10))
