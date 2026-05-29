"""End-to-end AQI training + evaluation + best model registration pipeline."""

import subprocess

from src.models.upload_model_to_registry import upload_best_model


# ─────────────────────────────────────────
# RUN TRAINING SCRIPTS
# ─────────────────────────────────────────

def run_script(script_path: str):

    print(f"\n[INFO] Running: {script_path}")

    subprocess.run(
        ["python", script_path],
        check=True
    )


# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────

def main():

    print("\n=================================================")
    print(" AQI TRAINING + REGISTRY PIPELINE (CLEAN VERSION) ")
    print("=================================================")

    # -----------------------------
    # 1. Train models
    # -----------------------------

    run_script("src/training/train_ridge.py")
    run_script("src/training/train_random_forest.py")
    run_script("src/training/train_xgboost.py")
    run_script("src/training/train_prophet.py")

    # -----------------------------
    # 2. Evaluate models
    # -----------------------------

    run_script("src/training/evaluate_models.py")

    # -----------------------------
    # 3. Upload BEST model (auto-selected)
    # -----------------------------

    print("\n[INFO] Uploading best model to registry...")

    upload_best_model()

    print("\n=================================================")
    print(" PIPELINE COMPLETE ")
    print("=================================================")


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    main()
