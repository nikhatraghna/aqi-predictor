"""Load production AQI model from Hopsworks Model Registry."""

from pathlib import Path

import joblib

from src.models.download_model_from_registry import (
    download_model,
)


DOWNLOAD_DIR = "models/registry_downloads"


# Global cache
_MODEL_DIR = None


def _ensure_downloaded():
    """
    Download latest production model only once.
    """

    global _MODEL_DIR

    if _MODEL_DIR is None:

        print("\n[INFO] Downloading production model...")

        _MODEL_DIR = download_model(
            model_name="ridge_aqi_model",
            download_dir=DOWNLOAD_DIR,
        )

    return Path(_MODEL_DIR)


def load_best_model():
    """
    Load latest production model.
    """

    model_dir = _ensure_downloaded()

    model_path = model_dir / "model.pkl"

    print(f"\n[INFO] Loading model: {model_path}")

    model = joblib.load(model_path)

    print("[SUCCESS] Model loaded.")

    return model


def load_scaler():
    """
    Load scaler from production artifacts.
    """

    model_dir = _ensure_downloaded()

    scaler_path = model_dir / "scaler.pkl"

    print(f"\n[INFO] Loading scaler: {scaler_path}")

    scaler = joblib.load(scaler_path)

    print("[SUCCESS] Scaler loaded.")

    return scaler


if __name__ == "__main__":

    model = load_best_model()
    scaler = load_scaler()

    print("\nLoaded Objects:")
    print("\nModel :", type(model))
    print("Scaler:", type(scaler))
