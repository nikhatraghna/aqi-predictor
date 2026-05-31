"""Load the production model + its inference contract from a model directory.

Source is configurable (same layout in both):
  - local promotion    → models/best_model/         (default)
  - Hopsworks download → models/registry_downloads/  (set AQI_MODEL_DIR)
"""

import os
import json
import joblib
from pathlib import Path

MODEL_DIR     = Path(os.getenv("AQI_MODEL_DIR", "models/best_model"))
MODEL_PATH    = MODEL_DIR / "model.pkl"
SCALER_PATH   = MODEL_DIR / "scaler.pkl"
CONTRACT_PATH = MODEL_DIR / "feature_config.json"


def load_feature_config() -> dict:
    """Load the inference contract (model_name, features, requires_scaling, ...)."""
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(
            f"{CONTRACT_PATH} not found. Run select_best_model.py "
            f"(or download_model_from_registry.py) first."
        )
    with open(CONTRACT_PATH) as f:
        return json.load(f)


def get_best_model_name() -> str:
    return load_feature_config()["model_name"]


def load_best_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("[SUCCESS] Model loaded.")
    return model


def load_scaler():
    """Load scaler ONLY if the contract requires scaling, else None."""
    config = load_feature_config()
    if not config.get("requires_scaling", False):
        print(f"[INFO] Model '{config['model_name']}' — no scaling required.")
        return None
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Contract requires scaling but {SCALER_PATH} is missing.")
    print(f"[INFO] Loading scaler: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    print("[SUCCESS] Scaler loaded.")
    return scaler


if __name__ == "__main__":
    cfg    = load_feature_config()
    model  = load_best_model()
    scaler = load_scaler()
    print(f"\nSource dir      : {MODEL_DIR}")
    print(f"Model name      : {cfg['model_name']}")
    print(f"Requires scaling: {cfg['requires_scaling']}")
    print(f"Features ({cfg['n_features']}): {cfg['features']}")
    print(f"Model  type     : {type(model)}")
    print(f"Scaler type     : {type(scaler)}")
