import os
import requests
import pandas as pd

from datetime import datetime
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


AQICN_API_KEY = os.getenv("AQICN_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = os.getenv("CITY")


def fetch_aqicn_data():

    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_API_KEY}"

    response = requests.get(url)

    data = response.json()

    if data["status"] != "ok":
        raise Exception("AQICN API failed")

    iaqi = data["data"].get("iaqi", {})

    aqi_data = {

        "datetime": datetime.now(),

        "aqi": data["data"].get("aqi"),

        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
    }

    return pd.DataFrame([aqi_data])


def fetch_openweather_data():

    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"q={CITY}&appid={OPENWEATHER_API_KEY}&units=metric"
    )

    response = requests.get(url)

    data = response.json()

    weather_data = {

        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],

        "wind_speed": data["wind"]["speed"],

        "visibility": data.get("visibility"),

        "clouds": data["clouds"]["all"],

    }

    return pd.DataFrame([weather_data])


def merge_data():

    aqi_df = fetch_aqicn_data()

    weather_df = fetch_openweather_data()

    final_df = pd.concat([aqi_df, weather_df], axis=1)

    return final_df


def save_data(df):

    os.makedirs("data/raw", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_path = (
        f"data/raw/{CITY.lower()}_aqi_{timestamp}.parquet"
    )

    df.to_parquet(file_path, index=False)

    print(f"Data saved at: {file_path}")


if __name__ == "__main__":

    final_df = merge_data()

    print(final_df)

    save_data(final_df)
