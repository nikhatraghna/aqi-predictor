"""Upload BEST AQI model directly to Hopsworks Model Registry."""

import json
from pathlib import Path

from src.models.hopsworks_model_registry import get_model_registry


METRICS_DIR = Path("models/metrics")
MODELS_DIR = Path("models/saved_models")


# ─────────────────────────────────────────
# FIND BEST MODEL (NO EXTRA SCRIPT)
# ─────────────────────────────────────────

def get_best_model():

    best_model_name = None
    best_r2 = -1

    print("\n[INFO] Reading model metrics...")

    for file in METRICS_DIR.glob("*.json"):

        with open(file, "r") as f:
            data = json.load(f)

        model_name = file.stem

        # support both formats safely
        test_metrics = data.get("test", data)

        r2 = test_metrics.get("r2")

        if r2 is not None and r2 > best_r2:
            best_r2 = r2
            best_model_name = model_name

    if best_model_name is None:
        raise ValueError("No valid model found in metrics")

    # clean suffix if exists
    best_model_name = best_model_name.replace("_advanced", "")

    return best_model_name, best_r2


# ─────────────────────────────────────────
# UPLOAD BEST MODEL
# ─────────────────────────────────────────

def upload_best_model():

    model_name, best_r2 = get_best_model()

    print(f"\n[INFO] Best model selected: {model_name} (R2={best_r2})")

    model_dir = MODELS_DIR / model_name

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model dir: {model_dir}")

    mr = get_model_registry()

    hops_model = mr.python.create_model(
        name=f"{model_name}_aqi_model",
        description="Auto-selected best AQI model",
        metrics={"r2": float(best_r2)},
    )

    print("\n[INFO] Uploading best model...")

    hops_model.save(str(model_dir))

    print("\n[SUCCESS] Best model uploaded!")

    return hops_model


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    upload_best_model()
