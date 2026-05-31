"""Load the promoted production model + its inference contract from models/best_model/."""

import json
import joblib
from pathlib import Path

BEST_MODEL_DIR  = Path("models/best_model")
MODEL_PATH      = BEST_MODEL_DIR / "model.pkl"
SCALER_PATH     = BEST_MODEL_DIR / "scaler.pkl"
CONTRACT_PATH   = BEST_MODEL_DIR / "feature_config.json"


def load_feature_config() -> dict:
    """Load the inference contract written by select_best_model.py.

    Returns:
        Dict with model_name, features, requires_scaling, target, etc.
    """
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(
            f"{CONTRACT_PATH} not found. Run select_best_model.py first."
        )
    with open(CONTRACT_PATH) as f:
        return json.load(f)


def get_best_model_name() -> str:
    """Return the promoted model's name (from the contract)."""
    return load_feature_config()["model_name"]


def load_best_model():
    """Load the promoted production model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\nRun select_best_model.py first."
        )
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("[SUCCESS] Model loaded.")
    return model


def load_scaler():
    """Load the scaler IF the contract says scaling is required, else None."""
    config = load_feature_config()

    if not config.get("requires_scaling", False):
        print(f"[INFO] Model '{config['model_name']}' — no scaling required.")
        return None

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Contract requires scaling but {SCALER_PATH} is missing. "
            "Re-run select_best_model.py."
        )
    print(f"[INFO] Loading scaler: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    print("[SUCCESS] Scaler loaded.")
    return scaler


if __name__ == "__main__":
    cfg    = load_feature_config()
    model  = load_best_model()
    scaler = load_scaler()
    print(f"\nModel name      : {cfg['model_name']}")
    print(f"Requires scaling: {cfg['requires_scaling']}")
    print(f"Features ({cfg['n_features']}): {cfg['features']}")
    print(f"Model  type     : {type(model)}")
    print(f"Scaler type     : {type(scaler)}")
