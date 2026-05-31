# 🌫️ Islamabad AQI Forecasting System

An end-to-end **MLOps system** that forecasts **PM2.5 air quality** for Islamabad, Pakistan.
It ingests weather + air-quality data, engineers time-series features, trains and compares
four ML models, serves predictions via an API and a dashboard, and runs automated
data/model **drift monitoring** — all orchestrated on a schedule with **GitHub Actions** and
backed by a **Hopsworks** feature store + model registry.

> **Scope note:** The 3-day output is currently a *hindcast* (it scores the most recent
> 72 feature rows), not a true forward forecast. True multi-step forecasting (forward-knowable
> features + a weather-forecast feed) is documented under [Roadmap](#-roadmap).

---

## 🏗️ Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        OM[Open-Meteo<br/>weather + air quality] --> HFP[hourly_feature_pipeline]
        HFP --> FS[(Hopsworks<br/>Feature Store)]
    end

    subgraph Training [Daily Training]
        FS --> T[Train 4 models<br/>Ridge · RF · XGBoost · LightGBM]
        T --> EV[Evaluate<br/>TimeSeriesSplit CV]
        EV --> SEL[Select best<br/>min CV-gap]
        SEL --> PROMO[Promote → models/best_model/<br/>+ feature_config.json]
        PROMO --> MR[(Hopsworks<br/>Model Registry)]
    end

    subgraph Serving
        PROMO --> INF[predict / forecast]
        INF --> API[FastAPI]
        INF --> DASH[Streamlit Dashboard]
    end

    subgraph Monitoring
        FS --> DD[Data drift<br/>KS + PSI]
        INF --> MD[Model drift<br/>baseline degradation]
        DD --> AL[Alerts +<br/>retraining flag]
        MD --> AL
        AL --> DASH
    end

    AQICN[AQICN + OpenWeather<br/>live conditions] -.display only.-> DASH
    GHA[GitHub Actions<br/>hourly + daily cron] -.orchestrates.-> HFP & T
Design principle: code lives in Git; features live in the Hopsworks Feature Store;
models live in the Hopsworks Model Registry. CI is stateless and reproduces state from Hopsworks.

✨ Key Features
Hourly feature pipeline — Open-Meteo ingestion, lag/rolling/time feature engineering, feature-store upsert.
Daily training pipeline — trains 4 models, leakage-free TimeSeriesSplit CV, anti-overfit configs, auto-selects + promotes the best.
Contract-driven inference — every model carries a feature_config.json (features, scaling flag, metrics), so swapping the winning model needs zero code changes.
Drift monitoring — KS + PSI data drift, baseline-relative model drift with rolling history, aggregated alerts + a retraining-readiness flag.
Two surfaces — a 5-page Streamlit dashboard and a 6-endpoint FastAPI service.
Full automation — both pipelines scheduled on GitHub Actions.
🧰 Tech Stack
Layer	Tools
Data	Open-Meteo API, AQICN, OpenWeather, pandas, pyarrow
ML	scikit-learn, XGBoost, LightGBM
Feature store / registry	Hopsworks
Explainability	SHAP, model feature importances
Serving	FastAPI, Streamlit, Altair
MLOps	GitHub Actions, python-dotenv
📂 Project Structure

aqi-predictor/
├── data/processed/          # engineered features, forecast (gitignored; restored from Hopsworks)
├── models/
│   ├── best_model/          # promoted model + scaler + feature_config.json (inference contract)
│   ├── saved_models/        # per-model artifacts
│   └── *.json               # comparison + best-model metadata
├── reports/drift/           # data/model drift + alert reports
├── src/
│   ├── data_pipeline/       # historical + realtime ingestion
│   ├── feature_engineering/ # feature creation + selection
│   ├── feature_store/       # Hopsworks connection + upload
│   ├── training/            # train_*, evaluate, select_best
│   ├── models/              # registry upload/download
│   ├── inference/           # load_model, predict, forecast_next_3_days
│   ├── monitoring/          # data_drift, model_drift, alerts
│   ├── api/                 # fastapi_app
│   ├── dashboard/           # Streamlit Home + pages/
│   └── automation/          # hourly_feature_pipeline, daily_training_pipeline
├── .github/workflows/       # feature_pipeline.yml, training_pipeline.yml
└── requirements.txt
🤖 Models & Selection
Four models are trained and compared on a chronological train/dev/test split with
nested feature selection inside a 3-fold TimeSeriesSplit (no leakage):

Model	Notes
Ridge	scaled, alpha tuned via inner CV
Random Forest	depth-limited, subsampled
XGBoost	early stopping, regularized
LightGBM	early stopping, regularized
Selection metric: smallest train↔validation R² gap (most stable generalization),
tie-broken by highest CV validation R². The winner is promoted to models/best_model/
with its inference contract and uploaded to the Hopsworks Model Registry.

Overfitting is judged by the R² gap (≤0.05 = healthy), not the RMSE ratio — the latter
is misleading when training error is very small.

🚀 Setup
1. Clone & install

git clone https://github.com/nikhatraghna/aqi-predictor.git
cd aqi-predictor
pip install -r requirements.txt
# dashboard / explainability extras:
pip install streamlit altair shap
2. Configure credentials (.env — never commit this)
Create a .env file in the repo root (a template is in .env.example):


HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_PROJECT=your_hopsworks_project_name
AQICN_API_KEY=your_aqicn_token_here          # optional — live dashboard panel only
OPENWEATHER_API_KEY=your_openweather_key_here # optional — live dashboard panel only
CITY=Islamabad
Hopsworks (required): create a free project at hopsworks.ai and generate an API key.
AQICN / OpenWeather (optional): only power the dashboard's "live conditions" panel; the model uses Open-Meteo (no key needed).
In GitHub Actions, store HOPSWORKS_API_KEY and HOPSWORKS_PROJECT as
Repository Secrets (Settings → Secrets and variables → Actions) — never in the repo.

▶️ Usage

# Ingest the latest hour into the feature store
python -m src.automation.hourly_feature_pipeline

# Full daily loop: train → evaluate → promote → forecast → monitor → register
python -m src.automation.daily_training_pipeline

# Inference
python -m src.inference.forecast_next_3_days

# Monitoring
python -m src.monitoring.data_drift
python -m src.monitoring.model_drift
python -m src.monitoring.alerts

# Dashboard (5 pages: Home · Forecast · Explanations · EDA · Monitoring)
streamlit run src/dashboard/Home.py

# API (Swagger UI at http://localhost:8000/docs)
uvicorn src.api.fastapi_app:app --reload
API endpoints
Method	Path	Description
GET	/	health + loaded model
GET	/model	model name, metrics, features
GET	/forecast	latest 3-day PM2.5 forecast
POST	/predict	predict for supplied feature rows
GET	/monitoring	latest drift / alert status
GET	/live	live AQICN + OpenWeather snapshot
📡 Monitoring
Data drift — KS test + PSI per feature (recent window vs reference); a feature is flagged DRIFT only when both agree (PSI ≥ 0.2 and KS significant).
Model drift — current RMSE vs the promoted model's baseline test RMSE; ≥20% → WARNING, ≥40% → DRIFT; rolling history saved for trends.
Alerts — aggregates both into an overall status and a retrain_recommended flag (triggered only on sustained model drift, not input drift alone).
⚙️ Automation (GitHub Actions)
Workflow	Schedule	Action
feature_pipeline.yml	hourly	restore features from Hopsworks → append latest hour → upload to feature store
training_pipeline.yml	daily	refresh features → retrain → promote → forecast → monitor → upload to model registry
Both are stateless: they restore state from Hopsworks at the start, so no data/model
binaries are committed to Git.

🗺️ Roadmap
True multi-step forecasting — a forecast-specific model on forward-knowable features (recursive lags + future weather from a forecast API), replacing the current hindcast.
Multi-city support — Rawalpindi, Lahore, Karachi.
Hosted dashboard reading directly from Hopsworks.
Retraining automation wired to the monitoring retrain_recommended flag.
📝 Design Notes
Source consistency: the model is trained and served on Open-Meteo data (same units/schema); AQICN/OpenWeather are used only for the human-readable live panel.
Feature store as source of truth: CI restores features from Hopsworks rather than committing parquet files.
Honest evaluation: leakage-free nested CV, R²-gap overfitting check, and a clearly-labeled hindcast.
