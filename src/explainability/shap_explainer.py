"""SHAP explainability for the promoted AQI model — writes a reusable importance artifact."""

from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.inference.load_model import load_best_model, load_scaler, load_feature_config

FEATURES_PATH = "data/processed/islamabad_features.parquet"
OUTPUT_DIR    = Path("reports/shap")
SAMPLE_SIZE   = 200   # rows used to estimate SHAP (kept small for speed)


def load_sample(features):
    """Load the most recent rows for the model's contract features."""
    df = pd.read_parquet(FEATURES_PATH)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Feature dataset missing required columns: {missing}")
    return df.dropna(subset=features)[features].tail(SAMPLE_SIZE).copy()


def main():
    print("\n[INFO] SHAP explainability for the promoted model...")
    cfg              = load_feature_config()
    features         = cfg["features"]
    requires_scaling = cfg.get("requires_scaling", False)

    model  = load_best_model()
    scaler = load_scaler()                 # None unless the contract requires scaling

    X = load_sample(features)
    X_input = X.copy()
    if requires_scaling:
        if scaler is None:
            raise RuntimeError("Contract requires scaling but no scaler was loaded.")
        X_input = pd.DataFrame(scaler.transform(X), columns=features, index=X.index)

    print(f"[INFO] Computing SHAP on {len(X_input)} rows × {len(features)} features...")
    explainer   = shap.Explainer(model, X_input)   # auto-selects TreeExplainer for XGBoost/LGBM
    shap_values = explainer(X_input)

    # mean(|SHAP|) per feature → the importance artifact the dashboard reads
    mean_abs = np.abs(np.asarray(shap_values.values)).mean(axis=0)
    imp = (pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
           .sort_values("mean_abs_shap", ascending=False)
           .reset_index(drop=True))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    imp.to_parquet(OUTPUT_DIR / "shap_importance.parquet", index=False)
    imp.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)
    print(f"[SUCCESS] SHAP importance → {OUTPUT_DIR}/shap_importance.parquet")
    print(imp.head(10).to_string(index=False))

    # Optional: keep the visual summary too
    try:
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_values, X_input, show=False)
        plt.savefig(OUTPUT_DIR / "shap_summary.png", bbox_inches="tight")
        plt.close()
        print(f"[SUCCESS] Summary plot → {OUTPUT_DIR}/shap_summary.png")
    except Exception as exc:
        print(f"[WARNING] Could not save summary plot: {exc}")


if __name__ == "__main__":
    main()
