
"""
Compare all trained models using freshly saved metrics.
"""

import json
from pathlib import Path

import pandas as pd


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

METRICS_DIR = Path("models/metrics")

MODEL_FILES = {
    "ridge": METRICS_DIR / "ridge.json",
    "random_forest": METRICS_DIR / "random_forest.json",
    "xgboost": METRICS_DIR / "xgboost.json",
    "prophet": METRICS_DIR / "prophet.json",
}


# ─────────────────────────────────────────
# LOAD METRICS
# ─────────────────────────────────────────

def load_metrics():

    rows = []

    for model_name, path in MODEL_FILES.items():

        if not path.exists():

            print(f"[WARNING] Missing metrics file: {path}")

            continue

        with open(path, "r") as f:

            metrics = json.load(f)

        rows.append({

            "Model": model_name,

            "MAE": metrics["mae"],

            "RMSE": metrics["rmse"],

            "R2": metrics["r2"],
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# SELECT BEST MODEL
# ─────────────────────────────────────────

def select_best_model(df):

    best_row = df.sort_values(
        by="RMSE"
    ).iloc[0]

    return {

        "model": best_row["Model"],

        "mae": float(best_row["MAE"]),

        "rmse": float(best_row["RMSE"]),

        "r2": float(best_row["R2"]),
    }


# ─────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────

def save_outputs(df, best_model):

    Path("models").mkdir(
        exist_ok=True
    )

    # Save full comparison

    comparison = df.to_dict(
        orient="records"
    )

    with open(
        "models/model_metrics.json",
        "w"
    ) as f:

        json.dump(
            comparison,
            f,
            indent=4,
        )

    # Save best model name

    with open(
        "models/best_model.json",
        "w"
    ) as f:

        json.dump({

            "best_model": best_model["model"]

        }, f, indent=4)

    # Save best model metrics

    with open(
        "models/best_model_metrics.json",
        "w"
    ) as f:

        json.dump(
            best_model,
            f,
            indent=4,
        )

    print(
        "\n[SUCCESS] Evaluation artifacts saved."
    )


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():

    print("\n==============================")

    print(" MODEL COMPARISON")

    print("==============================\n")

    df = load_metrics()

    if df.empty:

        raise ValueError(
            "No metrics found."
        )

    # Sort by RMSE

    df = df.sort_values(
        by="RMSE"
    ).reset_index(drop=True)

    print(df)

    # Best model

    best_model = select_best_model(df)

    print("\n==============================")

    print(" BEST MODEL")

    print("==============================\n")

    print(f"Model : {best_model['model']}")

    print(f"MAE   : {best_model['mae']:.4f}")

    print(f"RMSE  : {best_model['rmse']:.4f}")

    print(f"R²    : {best_model['r2']:.4f}")

    # Save outputs

    save_outputs(
        df,
        best_model,
    )


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":

    main()
