"""Select best AQI forecasting model and deploy to models/best_model/."""

import json
import shutil
from pathlib import Path

BEST_MODEL_FILE = "models/best_model.json"
MODEL_DIR       = Path("models/saved_models")
BEST_MODEL_DIR  = Path("models/best_model")

BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load best model name
with open(BEST_MODEL_FILE, "r") as f:
    best_model = json.load(f)["best_model"]

print(f"\n[SUCCESS] Best model selected: {best_model}")

MODEL_MAP = {
    "ridge": {
        "model":  MODEL_DIR / "ridge_model.pkl",
        "scaler": MODEL_DIR / "ridge_scaler.pkl",
    },
    "random_forest": {
        "model": MODEL_DIR / "random_forest_model.pkl",
    },
    "xgboost": {
        "model": MODEL_DIR / "xgboost_model.pkl",
    },
    "prophet": {
        "model": MODEL_DIR / "prophet_model.pkl",
    },
}

selected = MODEL_MAP[best_model]

# --- Copy model ---
shutil.copy(selected["model"], BEST_MODEL_DIR / "model.pkl")
print(f"[SUCCESS] Model copied → {BEST_MODEL_DIR / 'model.pkl'}")

# --- Copy scaler only if it exists (Ridge only) ---
scaler_src = selected.get("scaler")
scaler_dst = BEST_MODEL_DIR / "scaler.pkl"

if scaler_src and Path(scaler_src).exists():
    shutil.copy(scaler_src, scaler_dst)
    print(f"[SUCCESS] Scaler copied → {scaler_dst}")
else:
    # Remove stale scaler from a previous Ridge run
    if scaler_dst.exists():
        scaler_dst.unlink()
        print("[INFO] Removed stale scaler (not needed for this model).")

print("\n[SUCCESS] Best model artifacts prepared.")
