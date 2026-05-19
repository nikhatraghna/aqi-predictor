"""Hopsworks Model Registry connection utilities."""

import os
import hopsworks

from dotenv import load_dotenv


load_dotenv()


HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")


def get_model_registry():

    print("\n[INFO] Connecting to Hopsworks Model Registry...")

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )

    mr = project.get_model_registry()

    print("[SUCCESS] Connected to Model Registry.")

    return mr


if __name__ == "__main__":

    mr = get_model_registry()

    print("\nRegistry object:")
    print(type(mr))
