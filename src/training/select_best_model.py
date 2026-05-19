"""Select best AQI forecasting model."""

import json
from pathlib import Path


METRICS_PATH = "models/model_metrics.json"


def main():

    print("\n[INFO] Loading model metrics...")

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    print("\n=================================================")
    print(" MODEL COMPARISON ")
    print("=================================================")

    for model_name, vals in metrics.items():

        print(
            f"{model_name:<20}"
            f"RMSE={vals['rmse']:.4f}   "
            f"MAE={vals['mae']:.4f}   "
            f"R2={vals['r2']:.4f}"
        )

    # ---------------------------------------------------
    # Select best model using RMSE
    # ---------------------------------------------------

    best_model = min(
        metrics,
        key=lambda x: metrics[x]["rmse"]
    )

    best_metrics = metrics[best_model]

    print("\n=================================================")
    print(f" BEST MODEL: {best_model}")
    print("=================================================")

    # ---------------------------------------------------
    # Save best model name
    # ---------------------------------------------------

    Path("models").mkdir(exist_ok=True)

    with open("models/best_model.json", "w") as f:

        json.dump(
            {
                "best_model": best_model
            },
            f,
            indent=4,
        )

    # ---------------------------------------------------
    # Save best metrics
    # ---------------------------------------------------

    with open("models/best_model_metrics.json", "w") as f:

        json.dump(
            best_metrics,
            f,
            indent=4,
        )

    print("\n[SUCCESS] Best model saved.")


if __name__ == "__main__":
    main()
