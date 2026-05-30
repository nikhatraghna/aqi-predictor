"""Select best AQI forecasting model and deploy to models/best_model/."""

import json
import shutil
from pathlib import Path

# ─────────────────────────────────────
# PATHS
# ─────────────────────────────────────

METRICS_DIR = Path("models/metrics")
BEST_MODEL_FILE = Path("models/best_model.json")

MODEL_DIR = Path("models/saved_models")
BEST_MODEL_DIR = Path("models/best_model")

BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────
# LOAD METRICS (ONLY CLEAN FILES)
# ─────────────────────────────────────

metrics_files = list(METRICS_DIR.glob("*_advanced.json"))

metrics = {}

print("\n[INFO] Model comparison:")

for file in metrics_files:
    with open(file, "r") as f:
        data = json.load(f)

    model_name = file.stem.replace("_advanced", "")

    # ── SAFE RMSE EXTRACTION ──
    if "test" in data and "rmse" in data["test"]:
        rmse = data["test"]["rmse"]

    elif "rmse" in data:
        rmse = data["rmse"]

    else:
        print(f"[WARNING] Skipping {file.stem} (no RMSE found)")
        continue

    metrics[model_name] = rmse

    print(f"{model_name:<15} RMSE={rmse:.4f}")

# ─────────────────────────────────────
# SELECT BEST MODEL
# ─────────────────────────────────────

best_model = min(metrics, key=metrics.get)
best_rmse = metrics[best_model]

print("\n==============================")
print(" BEST MODEL")
print("==============================")

print(f"Model : {best_model}")
print(f"RMSE  : {best_rmse:.4f}")

# ─────────────────────────────────────
# SAVE BEST MODEL INFO
# ─────────────────────────────────────

with open(BEST_MODEL_FILE, "w") as f:
    json.dump({
        "best_model": best_model,
        "rmse": best_rmse
    }, f, indent=4)

print("\n[SUCCESS] Best model saved → models/best_model.json")

# ─────────────────────────────────────
# COPY ARTIFACTS
# ─────────────────────────────────────

MODEL_MAP = {
    "ridge": {
        "model": MODEL_DIR / "ridge_model.pkl",
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

# copy model
shutil.copy(selected["model"], BEST_MODEL_DIR / "model.pkl")
print(f"[SUCCESS] Model copied → {BEST_MODEL_DIR / 'model.pkl'}")

# copy scaler only for ridge
scaler_src = selected.get("scaler")
scaler_dst = BEST_MODEL_DIR / "scaler.pkl"

if scaler_src and Path(scaler_src).exists():
    shutil.copy(scaler_src, scaler_dst)
    print(f"[SUCCESS] Scaler copied → {scaler_dst}")
else:
    if scaler_dst.exists():
        scaler_dst.unlink()
        print("[INFO] Removed old scaler (not needed for this model).")

print("\n[SUCCESS] Best model deployment complete.")


