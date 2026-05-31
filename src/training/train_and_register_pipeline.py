"""End-to-end AQI pipeline: train 4 models → evaluate → promote best → upload to Hopsworks."""

import subprocess

from src.models.upload_model_to_registry import upload_best_model


TRAIN_MODULES = [
    "src.training.train_ridge",
    "src.training.train_random_forest",
    "src.training.train_xgboost",
    "src.training.train_lightgbm",     # Prophet retired; LightGBM is the 4th model
]


def run(module: str) -> None:
    """Run a project module as `python -m module` so `from src...` imports resolve."""
    print(f"\n[INFO] Running: {module}")
    subprocess.run(["python", "-m", module], check=True)


def main():
    print("\n=================================================")
    print(" AQI TRAINING + REGISTRY PIPELINE (4 MODELS) ")
    print("=================================================")

    # 1. Train all four models
    for module in TRAIN_MODULES:
        run(module)

    # 2. Compare on test + CV (writes best_model.json + best_model_metrics.json)
    run("src.training.evaluate_models")

    # 3. Promote winner → models/best_model/ (model.pkl + feature_config.json [+ scaler.pkl])
    run("src.training.select_best_model")

    # 4. Upload the promoted artifact to Hopsworks
    print("\n[INFO] Uploading best model to Hopsworks registry...")
    upload_best_model()

    print("\n=================================================")
    print(" PIPELINE COMPLETE ")
    print("=================================================")


if __name__ == "__main__":
    main()
