
"""Load production models for inference."""

import joblib

from pathlib import Path

from src.models.model_registry import (
    get_best_model_name,
)


MODELS_DIR = Path(
    "models/saved_models"
)


# ─────────────────────────────────────────
# LOAD BEST MODEL
# ─────────────────────────────────────────

def load_best_model():

    """Load the best production model."""

    model_name = (
        get_best_model_name()
    )

    model_path = (
        MODELS_DIR /
        f"{model_name}_model.pkl"
    )

    if not model_path.exists():

        raise FileNotFoundError(

            f"Model not found: {model_path}"
        )

    print(
        f"[INFO] Loading model: "
        f"{model_path}"
    )

    model = joblib.load(model_path)

    print(
        "[SUCCESS] Model loaded."
    )

    return model


# ─────────────────────────────────────────
# LOAD SCALER
# ─────────────────────────────────────────

def load_scaler():

    """Load Ridge scaler if available."""

    scaler_path = (
        MODELS_DIR /
        "ridge_scaler.pkl"
    )

    if not scaler_path.exists():

        print(
            "[WARNING] Scaler not found."
        )

        return None

    print(
        f"[INFO] Loading scaler: "
        f"{scaler_path}"
    )

    scaler = joblib.load(
        scaler_path
    )

    print(
        "[SUCCESS] Scaler loaded."
    )

    return scaler


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    model = load_best_model()

    scaler = load_scaler()

    print("\nLoaded Objects:\n")

    print("Model :", type(model))

    print("Scaler:", type(scaler))
