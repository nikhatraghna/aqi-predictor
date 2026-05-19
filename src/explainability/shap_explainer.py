"""SHAP explainability for AQI prediction models."""

from pathlib import Path

import matplotlib.pyplot as plt
import shap

from src.feature_store.hopsworks_connection import (
    get_feature_store,
)

from src.inference.load_model import (
    load_best_model,
    load_scaler,
)


OUTPUT_DIR = Path("reports/shap")


def load_feature_data():
    """
    Load features from Hopsworks Feature Store.
    """

    print("\n[INFO] Loading features from Hopsworks...")

    fs = get_feature_store()

    feature_group = fs.get_feature_group(
        name="aqi_features",
    )

    df = feature_group.read()

    # Remove target column if present
    if "aqi" in df.columns:
        X = df.drop(columns=["aqi"])

    else:
        X = df

    # Remove datetime if present
    if "datetime" in X.columns:
        X = X.drop(columns=["datetime"])

    print(f"[SUCCESS] Loaded {len(X)} rows.")

    return X


def create_shap_explainer(model, X_sample):
    """
    Create SHAP explainer object.
    """

    print("\n[INFO] Creating SHAP explainer...")

    explainer = shap.Explainer(
        model,
        X_sample,
    )

    print("[SUCCESS] SHAP explainer created.")

    return explainer


def generate_global_explanations(
    explainer,
    X_sample,
    feature_names,
):
    """
    Generate global SHAP explanations.
    """

    print("\n[INFO] Generating SHAP values...")

    shap_values = explainer(X_sample)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[INFO] Saving SHAP summary plot...")

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
    )

    plt.savefig(
        OUTPUT_DIR / "shap_summary.png",
        bbox_inches="tight",
    )

    plt.close()

    print("[SUCCESS] SHAP summary saved.")

    return shap_values


def main():

    print("\n========================================")
    print(" AQI MODEL EXPLAINABILITY ")
    print("========================================")

    # Load production artifacts
    model = load_best_model()

    scaler = load_scaler()

    # Load features
    X = load_feature_data()

    feature_names = X.columns.tolist()

    # Keep only training features
    # Keep only training features
    print("\n[INFO] Aligning feature columns...")

    X = X[scaler.feature_names_in_]

    # Scale features
    print("\n[INFO] Scaling features...")

    X_scaled = scaler.transform(X)

    # Use sample for faster SHAP
    X_sample = X_scaled[:100]

    # Create SHAP explainer
    explainer = create_shap_explainer(
        model,
        X_sample,
    )

    # Generate explanations
    generate_global_explanations(
        explainer,
        X_sample,
        feature_names,
    )

    print("\n========================================")
    print(" EXPLAINABILITY COMPLETE ")
    print("========================================")


if __name__ == "__main__":
    main()
