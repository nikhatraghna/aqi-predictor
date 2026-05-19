"""Full AQI training + model selection + registry pipeline."""

import json
import subprocess

from src.models.upload_model_to_registry import upload_model


def run_script(script_path: str):
    """
    Execute training/evaluation scripts.
    """

    print(f"\n[INFO] Running: {script_path}")

    result = subprocess.run(
        ["python", script_path],
        check=True,
    )

    return result


def load_best_model_metrics():
    """
    Load best model metrics.
    """

    with open("models/best_model_metrics.json", "r") as f:
        return json.load(f)


def load_best_model_name():
    """
    Load selected best model name.
    """

    with open("models/best_model.json", "r") as f:
        data = json.load(f)

    return data["best_model"]


def main():

    print("\n=================================================")
    print(" AQI TRAINING + REGISTRY PIPELINE ")
    print("=================================================")

    # ---------------------------------------------------
    # Train models
    # ---------------------------------------------------

    run_script("src/training/train_ridge.py")

    run_script("src/training/train_random_forest.py")

    run_script("src/training/train_xgboost.py")

    # ---------------------------------------------------
    # Evaluate models
    # ---------------------------------------------------

    run_script("src/training/evaluate_models.py")

    # ---------------------------------------------------
    # Select best model
    # ---------------------------------------------------

    run_script("src/training/select_best_model.py")

    best_model = load_best_model_name()

    metrics = load_best_model_metrics()

    print(f"\n[SUCCESS] Best model selected: {best_model}")

    print("\n[INFO] Uploading best model to registry...")

    upload_model(
        model_name=f"{best_model}_aqi_model",
        metrics=metrics,
    )

    print("\n=================================================")
    print(" PIPELINE COMPLETE ")
    print("=================================================")


if __name__ == "__main__":
    main()
