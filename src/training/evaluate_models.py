"""Compare the 4 trained models; select best by smallest CV gap, tie-break highest CV Val R²."""

import json
from pathlib import Path
import pandas as pd


METRICS_DIR = Path("models/metrics")

MODEL_FILES = {
    "ridge":         METRICS_DIR / "ridge.json",
    "random_forest": METRICS_DIR / "random_forest.json",
    "xgboost":       METRICS_DIR / "xgboost.json",
    "lightgbm":      METRICS_DIR / "lightgbm.json",
}

# Selection policy: most stable model first (smallest gap), break ties by best CV accuracy.
SORT_COLS = ["CV_Gap", "CV_Val_R2"]
SORT_ASC  = [True,     False]      # CV_Gap ascending, CV_Val_R2 descending


def load_metrics() -> pd.DataFrame:
    """Load test-set + CV metrics from all saved model JSON files."""
    rows = []
    for model_name, path in MODEL_FILES.items():
        if not path.exists():
            print(f"[WARNING] Missing metrics file: {path} — skipping")
            continue

        with open(path, "r") as f:
            metrics = json.load(f)

        if "test" not in metrics:
            print(f"[WARNING] No 'test' key in {path} — skipping")
            continue

        test = metrics["test"]
        cv   = metrics.get("cv", {})
        rows.append({
            "Model":     model_name,
            "MAE":       test["mae"],
            "RMSE":      test["rmse"],
            "R2":        test["r2"],
            "R2_gap":    metrics.get("test_r2_gap"),
            "CV_Val_R2": cv.get("mean_val_r2"),
            "CV_Gap":    cv.get("r2_gap"),
        })
        print(f"[INFO] Loaded '{model_name}' ← {path.name}")

    return pd.DataFrame(rows)


def select_best_model(df: pd.DataFrame) -> dict:
    """Select best model: smallest CV gap first, tie-break by highest CV Val R²."""
    candidates = df.dropna(subset=["CV_Gap", "CV_Val_R2"])
    if candidates.empty:
        raise ValueError("No models expose CV metrics. Retrain so each metrics file has a 'cv' block.")
    best = candidates.sort_values(by=SORT_COLS, ascending=SORT_ASC).iloc[0]
    return {
        "model":       best["Model"],
        "mae":         float(best["MAE"]),
        "rmse":        float(best["RMSE"]),
        "r2":          float(best["R2"]),
        "r2_gap":      None if pd.isna(best["R2_gap"]) else float(best["R2_gap"]),
        "cv_val_r2":   float(best["CV_Val_R2"]),
        "cv_gap":      float(best["CV_Gap"]),
        "selected_by": "min_cv_gap_then_cv_val_r2",
    }


def save_outputs(df: pd.DataFrame, best_model: dict) -> None:
    """Save comparison table and best model info to disk."""
    Path("models").mkdir(exist_ok=True)
    with open("models/model_metrics.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=4)
    with open("models/best_model.json", "w") as f:
        json.dump({"best_model": best_model["model"]}, f, indent=4)
    with open("models/best_model_metrics.json", "w") as f:
        json.dump(best_model, f, indent=4)
    print("[SUCCESS] Evaluation artifacts saved.")


def main():
    print("\n==============================")
    print(" MODEL COMPARISON")
    print("==============================\n")

    df = load_metrics()
    if df.empty:
        raise ValueError("No metrics found. Train at least one model first.")

    # Rank by selection policy: smallest CV gap first, then highest CV Val R²
    df = df.sort_values(by=SORT_COLS, ascending=SORT_ASC).reset_index(drop=True)
    print("\n" + df.to_string(index=False))

    print("\n[INFO] Selection: smallest CV_Gap first, tie-break highest CV_Val_R2 | "
          "R2_gap ≤ 0.05 = no overfitting")
    for _, r in df.iterrows():
        cvr2  = "n/a" if pd.isna(r["CV_Val_R2"]) else f"{r['CV_Val_R2']:.4f}"
        cvgap = "n/a" if pd.isna(r["CV_Gap"])    else f"{r['CV_Gap']:.4f}"
        print(f"  {r['Model']:<15} CV_Gap={cvgap:<8} CV_Val_R2={cvr2}")

    best_model = select_best_model(df)

    print("\n==============================")
    print(" BEST MODEL (smallest CV gap)")
    print("==============================\n")
    print(f"  Model      : {best_model['model']}")
    print(f"  CV Gap     : {best_model['cv_gap']:.4f}   ← selection metric")
    print(f"  CV Val R²  : {best_model['cv_val_r2']:.4f}   ← tie-breaker")
    print(f"  Test RMSE  : {best_model['rmse']:.4f}")
    print(f"  Test R²    : {best_model['r2']:.4f}")

    save_outputs(df, best_model)
    print("\n[SUCCESS] Evaluation complete.")


if __name__ == "__main__":
    main()
