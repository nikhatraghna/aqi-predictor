"""Download model artifacts from Hopsworks Model Registry."""

import shutil
from pathlib import Path

from src.models.hopsworks_model_registry import (
    get_model_registry,
)

DOWNLOAD_DIR = "models/registry_downloads"


def download_model(
    model_name: str = "ridge_aqi_model",
    version: int | None = None,
    download_dir: str = DOWNLOAD_DIR,
):
    """
    Download model artifacts from Hopsworks.

    If version is None:
    -> automatically use latest version.
    """

    print("\n[INFO] Connecting to Model Registry...")

    mr = get_model_registry()

    print(f"\n[INFO] Fetching model: {model_name}")

    # ---------------------------------------------------
    # Get latest version automatically
    # ---------------------------------------------------

    if version is None:

        models = mr.get_models(model_name)

        latest_model = max(models, key=lambda m: m.version)

        version = latest_model.version

    print(f"[INFO] Using latest model version: {version}")

    model = mr.get_model(
        model_name,
        version=version,
    )

    download_path = Path(download_dir)

    # ---------------------------------------------------
    # Remove old artifacts before downloading
    # ---------------------------------------------------

    if download_path.exists():

        print("\n[INFO] Removing old downloaded artifacts...")

        shutil.rmtree(download_path)

    download_path.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Downloading model artifacts...")

    local_path = model.download(download_dir)

    print(f"[SUCCESS] Model downloaded to: {local_path}")

    return Path(local_path)


if __name__ == "__main__":

    download_model()
