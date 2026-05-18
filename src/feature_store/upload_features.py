
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath("."))

from src.feature_store.hopsworks_connection import get_feature_store


FEATURE_PATH = "data/processed/islamabad_features.parquet"


def load_features():

    print("\n[INFO] Loading engineered features...")

    df = pd.read_parquet(FEATURE_PATH)

    print(f"[INFO] Dataset shape: {df.shape}")

    return df


def upload_to_feature_store(df):

    fs = get_feature_store()

    print("\n[INFO] Creating feature group...")

    feature_group = fs.get_or_create_feature_group(

        name="aqi_features",

        version=1,

        primary_key=["datetime"],

        event_time="datetime",

        description="AQI forecasting engineered features",

        online_enabled=True,
    )

    print("[INFO] Uploading features to Hopsworks...")

    feature_group.insert(df)

    print("\n[SUCCESS] Features uploaded successfully.")


if __name__ == "__main__":

    df = load_features()

    upload_to_feature_store(df)
