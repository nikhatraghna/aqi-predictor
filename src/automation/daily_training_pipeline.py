"""Daily pipeline: retrain → evaluate → promote → forecast → monitor → register.

One command that refreshes EVERYTHING the dashboard reads — the 4-model comparison,
the promoted best model, the 3-day forecast, the drift/alert reports, and the
Hopsworks model registry. This is what GitHub Actions calls on a daily cron.

Assumes features are already fresh (that's the hourly_feature_pipeline's job).
"""

import subprocess
import sys

# (label, module, critical?) — critical steps abort the run; others warn and continue
STEPS = [
    ("Train Ridge",         "src.training.train_ridge",            True),
    ("Train Random Forest", "src.training.train_random_forest",    True),
    ("Train XGBoost",       "src.training.train_xgboost",          True),
    ("Train LightGBM",      "src.training.train_lightgbm",         True),
    ("Evaluate models",     "src.training.evaluate_models",        True),
    ("Promote best model",  "src.training.select_best_model",      True),
   # ("3-day forecast",      "src.inference.forecast_next_3_days",  False),
    ("72h forward forecast", "src.inference.forecast_future", False),
    ("72h backtest",         "src.inference.forecast_next_3_days",  False),

    ("Data drift",          "src.monitoring.data_drift",           False),
    ("Model drift",         "src.monitoring.model_drift",          False),
    ("Alerts",              "src.monitoring.alerts",               False),
    ("SHAP explainability", "src.explainability.shap_explainer", False),
    ("Upload to registry",  "src.models.upload_model_to_registry", False),
    ("Upload dashboard bundle", "src.automation.upload_dashboard_bundle", False),

]


def run(label: str, module: str, critical: bool) -> bool:
    print(f"\n{'=' * 60}\n▶ {label}   ({module})\n{'=' * 60}")
    result = subprocess.run([sys.executable, "-m", module])
    if result.returncode != 0:
        msg = f"[FAILED] {label} ({module}) — exit {result.returncode}"
        if critical:
            raise RuntimeError(msg + "  → aborting pipeline (critical step).")
        print(msg + "  → continuing (non-critical).")
        return False
    return True


def main():
    print("\n############################################")
    print("#   DAILY TRAINING + REFRESH PIPELINE      #")
    print("############################################")

    failed = []
    for label, module, critical in STEPS:
        ok = run(label, module, critical)
        if not ok:
            failed.append(label)

    print("\n############################################")
    if failed:
        print("#   PIPELINE COMPLETE (with warnings)      #")
        print("############################################")
        print(f"[WARNING] Non-critical steps that failed: {failed}")
    else:
        print("#   PIPELINE COMPLETE — dashboard is fresh #")
        print("############################################")


if __name__ == "__main__":
    main()
