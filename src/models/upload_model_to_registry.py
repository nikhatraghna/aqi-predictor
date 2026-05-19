"""Upload trained models to Hopsworks Model Registry."""

from pathlib import Path

from src.models.hopsworks_model_registry import (
    get_model_registry,
)


MODEL_DIR = "models/saved_models/ridge"


def upload_model(
    model_name: str,
    model_dir: str = MODEL_DIR,
    version: int | None = None,
    metrics: dict | None = None,
):

    print("\n[INFO] Connecting to Model Registry...")

    mr = get_model_registry()

    local_model_dir = Path(model_dir)

    if not local_model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {local_model_dir}"
        )

    print(f"\n[INFO] Creating registry entry: {model_name}")

    hops_model = mr.python.create_model(
        name=model_name,
        version=version,
        description="Ridge AQI forecasting model",
        metrics=metrics or {},
    )

    print("\n[INFO] Uploading model artifacts...")

    hops_model.save(str(local_model_dir))

    print("\n[SUCCESS] Model uploaded to Hopsworks Registry.")

    return hops_model


if __name__ == "__main__":

    upload_model(
        model_name="ridge_aqi_model",
    )
