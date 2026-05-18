
import os
import hopsworks

from dotenv import load_dotenv

load_dotenv()


HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")


def get_feature_store():

    print("\n[INFO] Connecting to Hopsworks...")

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )

    fs = project.get_feature_store()

    print("[SUCCESS] Connected to Hopsworks.")

    return fs
