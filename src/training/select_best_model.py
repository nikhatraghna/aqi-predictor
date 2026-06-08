"""Promote the best model (selected by CV Val R²) into models/best_model/ with its inference contract."""

import json
import shutil
from pathlib import Path

METRICS_DIR     = Path("models/metrics")
SAVED_DIR       = Path("models/saved_models")
BEST_MODEL_JSON = Path("models/best_model.json")
PROD_DIR        = Path("models/best_model")

# model name → (artifact path, needs a scaler?)
MODEL_ARTIFACTS = {
    "ridge":         (SAVED_DIR / "ridge_model.pkl",         True),
    "random_forest": (SAVED_DIR / "random_forest_model.pkl", False),
    "xgboost":       (SAVED_DIR / "xgboost_model.pkl",       False),
    "lightgbm":      (SAVED_DIR / "lightgbm_model.pkl",      False),
}
METRIC_FILES = {
    "ridge":         METRICS_DIR / "ridge.json",
    "random_forest": METRICS_DIR / "random_forest.json",
    "xgboost":       METRICS_DIR / "xgboost.json",
    "lightgbm":      METRICS_DIR / "lightgbm.json",
}


def get_best_model_name() -> str:
    if not BEST_MODEL_JSON.exists():
        raise FileNotFoundError(f"{BEST_MODEL_JSON} not found. Run evaluate_models.py first.")
    with open(BEST_MODEL_JSON) as f:
        return json.load(f)["best_model"]


def promote(name: str) -> None:
    if name not in MODEL_ARTIFACTS:
        raise ValueError(f"Unknown model '{name}'. Known: {list(MODEL_ARTIFACTS)}")
    model_src, needs_scaler = MODEL_ARTIFACTS[name]
    if not model_src.exists():
        raise FileNotFoundError(f"Model artifact missing: {model_src}. Retrain '{name}'.")

    with open(METRIC_FILES[name]) as f:
        metrics = json.load(f)
    features = metrics.get("features")
    if features is None:
        raise KeyError(f"'features' missing in {METRIC_FILES[name]}. Retrain '{name}'.")

    cv = metrics.get("cv", {})

    if PROD_DIR.exists():
        shutil.rmtree(PROD_DIR)
    PROD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Model artifact
    shutil.copy(model_src, PROD_DIR / "model.pkl")
    print(f"[SUCCESS] Model copied → {PROD_DIR / 'model.pkl'}")

    # 2. Scaler — only for models trained on scaled inputs (Ridge)
    if needs_scaler:
        scaler_src = SAVED_DIR / f"{name}_scaler.pkl"
        if not scaler_src.exists():
            raise FileNotFoundError(f"'{name}' requires a scaler but {scaler_src} is missing.")
        shutil.copy(scaler_src, PROD_DIR / "scaler.pkl")
        print(f"[SUCCESS] Scaler copied → {PROD_DIR / 'scaler.pkl'}")

    # 3. Inference contract — single source of truth for predict.py + dashboard
    contract = {
        "model_name":       name,
        "target":           "pm25",
        "requires_scaling": needs_scaler,
        "features":         features,
        "n_features":       len(features),
        "selected_by":      "min_rmse_then_cv_val_r2_among_healthy",
        "cv_val_r2":        cv.get("mean_val_r2"),
        "cv_r2_gap":        cv.get("r2_gap"),
        "test_metrics":     metrics.get("test"),
        "test_r2_gap":      metrics.get("test_r2_gap"),
    }
    with open(PROD_DIR / "feature_config.json", "w") as f:
        json.dump(contract, f, indent=4)
    print(f"[SUCCESS] Contract written → {PROD_DIR / 'feature_config.json'}")


def main():
    print("\n==============================")
    print(" BEST MODEL")
    print("==============================")
    name = get_best_model_name()
    print(f"\n[INFO] Best model (by CV Val R²): {name}")
    promote(name)
    print("\n[INFO] Production artifacts:")
    for p in sorted(PROD_DIR.iterdir()):
        print(f"   - {p}  ({p.stat().st_size:,} bytes)")
    print(f"\n[SUCCESS] '{name}' promoted to {PROD_DIR}/ — ready for inference.")


if __name__ == "__main__":
    main()
