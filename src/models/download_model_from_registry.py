"""
Download the current best model
from Hopsworks Model Registry.
"""

import json
import shutil
from pathlib import Path

from src.models.hopsworks_model_registry import get_model_registry


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

DOWNLOAD_DIR = Path("models/registry_downloads")

BEST_MODEL_FILE = Path("models/best_model.json")


# ─────────────────────────────────────────
# LOAD BEST MODEL NAME
# ─────────────────────────────────────────

def get_best_model_name():

    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing file: {BEST_MODEL_FILE}")

    with open(BEST_MODEL_FILE, "r") as f:
        data = json.load(f)

    return data["best_model"]


# ─────────────────────────────────────────
# DOWNLOAD BEST MODEL
# ─────────────────────────────────────────

def download_best_model():

    print("\n[INFO] Connecting to Model Registry...")

    mr = get_model_registry()

    # Get best model name
    best_model_name = get_best_model_name()

    registry_model_name = f"{best_model_name}_aqi_model"

    print(f"[INFO] Best model: {registry_model_name}")

    # Get models safely
    models = mr.get_models(registry_model_name)

    if not models:
        raise ValueError(
            f"No models found in registry for {registry_model_name}"
        )

    # Select latest version
    model = max(models, key=lambda m: m.version)

    print(f"[SUCCESS] Found model v{model.version}")

    # Clean old downloads
    if DOWNLOAD_DIR.exists():
        print("[INFO] Cleaning old downloads...")
        shutil.rmtree(DOWNLOAD_DIR)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # Download model
    print("\n[INFO] Downloading model artifacts...")

    path = model.download(str(DOWNLOAD_DIR))

    print(f"\n[SUCCESS] Model downloaded → {path}")

    # Verify expected artifacts
    downloaded = Path(path)

    for required in ("model.pkl", "feature_config.json"):
        if not (downloaded / required).exists():
            print(
                f"[WARNING] '{required}' missing in download "
                f"— registry artifact may be incomplete"
            )
        else:
            print(f"[OK] {required}")

    return path



# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":

    download_best_model()
