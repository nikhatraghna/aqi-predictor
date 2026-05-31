"""Upload the promoted best model (models/best_model/) to the Hopsworks Model Registry."""

import json
from pathlib import Path
from src.models.hopsworks_model_registry import get_model_registry

PROD_DIR          = Path("models/best_model")
BEST_MODEL_JSON   = Path("models/best_model.json")
BEST_METRICS_JSON = Path("models/best_model_metrics.json")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run evaluate_models.py + select_best_model.py first."
        )
    with open(path) as f:
        return json.load(f)


def _registry_metrics(m: dict) -> dict:
    """Numeric metrics for the registry UI (selection criterion + accuracy)."""
    out = {}
    for key in ("rmse", "mae", "r2", "cv_val_r2", "cv_gap", "r2_gap"):
        if m.get(key) is not None:
            out[key] = float(m[key])
    return out


def upload_best_model():
    if not (PROD_DIR / "model.pkl").exists():
        raise FileNotFoundError(f"{PROD_DIR/'model.pkl'} missing. Run select_best_model.py first.")

    # Read the decision already made by evaluate_models.py — do NOT re-select here
    model_name = _load_json(BEST_MODEL_JSON)["best_model"]
    metrics    = _load_json(BEST_METRICS_JSON)
    reg_metrics = _registry_metrics(metrics)

    print(f"\n[INFO] Best model : {model_name}")
    print(f"[INFO] Selected by: {metrics.get('selected_by', 'n/a')}")
    print(f"[INFO] Metrics    : {reg_metrics}")

    mr = get_model_registry()
    hops_model = mr.python.create_model(
        name=f"{model_name}_aqi_model",
        description=(
            f"Auto-selected best AQI model ({model_name}); "
            f"selected by {metrics.get('selected_by', 'n/a')}. "
            f"Promoted from models/best_model/ with feature_config.json contract."
        ),
        metrics=reg_metrics,
    )

    print("[INFO] Uploading promoted artifacts (model.pkl + feature_config.json [+ scaler.pkl])...")
    hops_model.save(str(PROD_DIR), keep_original_files=True)   # keep local copy intact

    print(f"[SUCCESS] '{model_name}' uploaded → '{model_name}_aqi_model' (version {hops_model.version})")
    return hops_model


if __name__ == "__main__":
    upload_best_model()
