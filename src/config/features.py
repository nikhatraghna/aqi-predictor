import os

os.makedirs(os.path.dirname("src/config/features.py"), exist_ok=True)

# src/config/features.py
FEATURE_COLUMNS = [
    "pm10",
    "co",
    "no2",
    "so2",
    "o3",
    "aod",
    "dust",
    "temperature",
    "humidity",
    "precipitation",
    "pressure",
    "wind_speed",
    "wind_direction",
    "cloud_cover",
    "weather_code",
    "pm25_lag_1",
    "pm25_lag_3",
    "pm25_lag_6",
    "pm25_lag_12",
    "pm25_lag_24",
    "pm25_roll_mean_3",
    "pm25_roll_std_3",
    "pm25_roll_mean_6",
    "pm25_roll_std_6",
    "pm25_roll_mean_12",
    "pm25_roll_std_12",
    "pm25_roll_mean_24",
    "pm25_roll_std_24",
    "hour",
    "day_of_week",
    "month",
    "day",
    "hour_sin",
    "hour_cos",
]

TARGET_COLUMN = "pm25"
