"""Compare all trained models using freshly saved test-set metrics."""

import json
from pathlib import Path
import pandas as pd


METRICS_DIR = Path("models/metrics")

MODEL_FILES = {
    "ridge":         METRICS_DIR / "ridge.json",
    "random_forest": METRICS_DIR / "random_forest.json",
    "xgboost":       METRICS_DIR / "xgboost.json",
    "prophet":       METRICS_DIR / "prophet.json",
}


def load_metrics() -> pd.DataFrame:
    """Load test-set metrics from all saved model JSON files.

    Returns:
        DataFrame with Model, MAE, RMSE, R2 columns.
    """
    rows = []

    for model_name, path in MODEL_FILES.items():
        if not path.exists():
            print(f"[WARNING] Missing metrics file: {path} — skipping")
            continue

        with open(path, "r") as f:
            metrics = json.load(f)

        # ── FIX: all metric files store results under "test" key ──
        if "test" not in metrics:
            print(f"[WARNING] No 'test' key in {path} — skipping")
            continue

        test = metrics["test"]
        rows.append({
            "Model": model_name,
            "MAE":   test["mae"],
            "RMSE":  test["rmse"],
            "R2":    test["r2"],
        })

    return pd.DataFrame(rows)


def select_best_model(df: pd.DataFrame) -> dict:
    """Select best model by lowest test RMSE.

    Args:
        df: DataFrame with model metrics.

    Returns:
        Dict with model name and its metrics.
    """
    best_row = df.sort_values(by="RMSE").iloc[0]
    return {
        "model": best_row["Model"],
        "mae":   float(best_row["MAE"]),
        "rmse":  float(best_row["RMSE"]),
        "r2":    float(best_row["R2"]),
    }


def save_outputs(df: pd.DataFrame, best_model: dict) -> None:
    """Save comparison table and best model info to disk.

    Args:
        df:         Full model comparison DataFrame.
        best_model: Dict with best model name and metrics.
    """
    Path("models").mkdir(exist_ok=True)

    with open("models/model_metrics.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=4)

    with open("models/best_model.json", "w") as f:
        json.dump({"best_model": best_model["model"]}, f, indent=4)

    with open("models/best_model_metrics.json", "w") as f:
        json.dump(best_model, f, indent=4)

    print("[SUCCESS] Evaluation artifacts saved.")


def main():
    """Run full model comparison and save results."""

    print("\n==============================")
    print(" MODEL COMPARISON")
    print("==============================\n")

    df = load_metrics()

    if df.empty:
        raise ValueError(
            "No metrics found. Train at least one model first."
        )

    df = df.sort_values(by="RMSE").reset_index(drop=True)

    # ── Print full comparison table ───────────────────────────────
    print(df.to_string(index=False))

    # ── Overfitting warning per model ─────────────────────────────
    print("\n[INFO] Lower RMSE = better | R² closer to 1.0 = better")

    best_model = select_best_model(df)

    print("\n==============================")
    print(" BEST MODEL (by Test RMSE)")
    print("==============================\n")
    print(f"  Model : {best_model['model']}")
    print(f"  MAE   : {best_model['mae']:.4f}")
    print(f"  RMSE  : {best_model['rmse']:.4f}")
    print(f"  R²    : {best_model['r2']:.4f}")

    save_outputs(df, best_model)
    print("\n[SUCCESS] Evaluation complete.")


if __name__ == "__main__":
    main()
