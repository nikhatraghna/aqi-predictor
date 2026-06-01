"""Zip the dashboard artifacts and upload to Hopsworks (Resources/aqi_dashboard/)."""
import os, zipfile, tempfile
from pathlib import Path
import hopsworks
from dotenv import load_dotenv
load_dotenv()

FILES = [
    "models/model_metrics.json", "models/best_model.json", "models/best_model_metrics.json",
    "data/processed/forecast_3days.parquet", "data/processed/islamabad_features.parquet",
    "data/processed/current_conditions.json",
    "reports/drift/data_drift_report.json", "reports/drift/model_drift_report.json",
    "reports/drift/alerts_report.json",
    "reports/shap/shap_importance.parquet",

]
BEST = Path("models/best_model")

def main():
    tmp = Path(tempfile.mkdtemp()); zpath = tmp / "dashboard_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            if Path(f).exists(): z.write(f, f)          # store with relative path
        if BEST.exists():
            for p in BEST.rglob("*"):
                if p.is_file(): z.write(p, str(p))
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"),
                              project=os.getenv("HOPSWORKS_PROJECT"))
    ds = project.get_dataset_api()
    try: ds.mkdir("Resources/aqi_dashboard")
    except Exception: pass
    ds.upload(str(zpath), "Resources/aqi_dashboard", overwrite=True)
    print("[SUCCESS] Uploaded dashboard_bundle.zip → Hopsworks Resources/aqi_dashboard/")

if __name__ == "__main__":
    main()
