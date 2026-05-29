
"""Load production AQI model from models/best_model/."""

import json
import joblib
from pathlib import Path

BEST_MODEL_DIR = Path("models/best_model")
BEST_MODEL_JSON = Path("models/best_model.json")


def get_best_model_name() -> str:
    with open(BEST_MODEL_JSON) as f:
        return json.load(f)["best_model"]


def load_best_model():
    model_path = BEST_MODEL_DIR / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run select_best_model.py first."
        )
    print(f"[INFO] Loading model: {model_path}")
    model = joblib.load(model_path)
    print("[SUCCESS] Model loaded.")
    return model


def load_scaler():
    """Only returns scaler when Ridge is the best model."""
    model_name = get_best_model_name()

    if model_name != "ridge":
        print(f"[INFO] Model is {model_name} — no scaler needed.")
        return None

    scaler_path = BEST_MODEL_DIR / "scaler.pkl"
    if not scaler_path.exists():
        print("[WARNING] Ridge is best model but scaler.pkl missing!")
        return None

    print(f"[INFO] Loading scaler: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print("[SUCCESS] Scaler loaded.")
    return scaler


if __name__ == "__main__":
    model  = load_best_model()
    scaler = load_scaler()
    print("\nModel :", type(model))
    print("Scaler:", type(scaler))
