"""True 72-hour-ahead PM2.5 forecast (recursive multi-step).

Future weather + co-pollutant inputs come from Open-Meteo's FORECAST endpoint;
PM2.5 lag/rolling features are rolled forward using the model's own predictions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from src.inference.load_model import load_best_model, load_scaler, load_feature_config

LAT, LON = 33.6844, 73.0479
FEATURES_PATH = "data/processed/islamabad_features.parquet"
OUT_PATH = "data/processed/forecast_3days.parquet"
HORIZON = 72
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AQ_URL      = "https://air-quality-api.open-meteo.com/v1/air-quality"

AQI_BANDS = [(0,12,"Good","🟢"),(12,35,"Moderate","🟡"),(35,55,"Unhealthy (Sensitive)","🟠"),
             (55,150,"Unhealthy","🔴"),(150,250,"Very Unhealthy","🟣"),(250,99999,"Hazardous","⛔")]
def pm25_to_category(v):          # was: def _cat(v):
    for lo, hi, l, e in AQI_BANDS:
        if lo <= v < hi: return l, e
    return "Hazardous", "⛔"


import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _session():
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=2,                 # waits 2s,4s,8s,16s
                  status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def fetch_forecast_exog() -> pd.DataFrame:
    s = _session()
    w = s.get(WEATHER_URL, params={"latitude":LAT,"longitude":LON,
        "hourly":["temperature_2m","relative_humidity_2m","precipitation","surface_pressure",
                  "wind_speed_10m","wind_direction_10m","cloud_cover","weather_code"],
        "timezone":"Asia/Karachi","wind_speed_unit":"ms","forecast_days":4}, timeout=30); w.raise_for_status()
    a = s.get(AQ_URL, params={"latitude":LAT,"longitude":LON,
        "hourly":["pm10","carbon_monoxide","nitrogen_dioxide","sulphur_dioxide","ozone",
                  "aerosol_optical_depth","dust"],"timezone":"Asia/Karachi","forecast_days":4}, timeout=30); a.raise_for_status()
    wh, ah = w.json()["hourly"], a.json()["hourly"]
    dfw = pd.DataFrame({"datetime":pd.to_datetime(wh["time"]),"temperature":wh["temperature_2m"],
        "humidity":wh["relative_humidity_2m"],"precipitation":wh["precipitation"],"pressure":wh["surface_pressure"],
        "wind_speed":wh["wind_speed_10m"],"wind_direction":wh["wind_direction_10m"],
        "cloud_cover":wh["cloud_cover"],"weather_code":wh["weather_code"]})
    dfa = pd.DataFrame({"datetime":pd.to_datetime(ah["time"]),"pm10":ah["pm10"],"co":ah["carbon_monoxide"],
        "no2":ah["nitrogen_dioxide"],"so2":ah["sulphur_dioxide"],"o3":ah["ozone"],
        "aod":ah["aerosol_optical_depth"],"dust":ah["dust"]})
    m = pd.merge(dfa, dfw, on="datetime", how="inner")
    m["datetime"] = pd.to_datetime(m["datetime"]).dt.tz_localize("UTC")
    return m

def main():
    cfg = load_feature_config()
    features, needs_scaling = cfg["features"], cfg.get("requires_scaling", False)
    model, scaler = load_best_model(), load_scaler()

    hist = pd.read_parquet(FEATURES_PATH)
    hist["datetime"] = pd.to_datetime(hist["datetime"], utc=True)
    hist = hist.sort_values("datetime").reset_index(drop=True)
    last_dt = hist["datetime"].max()
    pm = hist["pm25"].tolist()                      # actuals; predictions get appended

    exog = fetch_forecast_exog()
    fut = exog[exog["datetime"] > last_dt].head(HORIZON).reset_index(drop=True)
    if fut.empty:
        raise RuntimeError("No future hours from Open-Meteo (feature history may be stale — "
                           "run the hourly pipeline so it reaches ~now).")

    rows = []
    for i in range(len(fut)):
        ts = fut.loc[i, "datetime"]
        f = {}
        for c in ["pm10","co","no2","so2","o3","aod","dust","temperature","humidity",
                  "precipitation","pressure","wind_speed","wind_direction","cloud_cover","weather_code"]:
            if c in fut.columns: f[c] = fut.loc[i, c]
        for lag in [1,3,6,12,24]:
            f[f"pm25_lag_{lag}"] = pm[-lag] if len(pm) >= lag else pm[-1]
        for wn in [3,6,12,24]:
            win = pm[-wn:]
            f[f"pm25_roll_mean_{wn}"] = float(np.mean(win))
            f[f"pm25_roll_std_{wn}"]  = float(np.std(win, ddof=1)) if len(win) > 1 else 0.0
        f["hour"], f["day_of_week"] = ts.hour, ts.dayofweek
        f["month"], f["day"] = ts.month, ts.day
        f["hour_sin"], f["hour_cos"] = np.sin(2*np.pi*ts.hour/24), np.cos(2*np.pi*ts.hour/24)

        missing = [c for c in features if c not in f]
        if missing:
            raise ValueError(f"Cannot build forecast features (missing: {missing})")
        X = pd.DataFrame([{c: f[c] for c in features}])
        if needs_scaling:
            X = scaler.transform(X)
        yhat = float(model.predict(X)[0])
        pm.append(yhat)                              # feed prediction forward (recursive)
       # lbl, emo = _cat(yhat)
        lbl, emo = pm25_to_category(yhat)
        rows.append({"hour": i+1, "datetime": pd.Timestamp(ts),
                     "predicted_pm25": round(yhat, 2), "category": lbl, "status": emo})

    out = pd.DataFrame(rows)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"[SUCCESS] True {len(out)}-hour forward forecast → {OUT_PATH}")
    print(f"[INFO] {out['datetime'].min()} → {out['datetime'].max()}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
