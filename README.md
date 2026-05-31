# 🌫️ Islamabad AQI Forecasting System

An end-to-end **MLOps system** that forecasts **PM2.5 air quality** for Islamabad, Pakistan —
from automated data ingestion and feature engineering, through multi-model training and
selection, to serving (API + dashboard), drift monitoring, and scheduled retraining on
**GitHub Actions**, all backed by a **Hopsworks** feature store and model registry.

![Tests](https://github.com/nikhatraghna/aqi-predictor/actions/workflows/tests.yml/badge.svg)
![Docker Build](https://github.com/nikhatraghna/aqi-predictor/actions/workflows/docker.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)

**Note on the forecast:** the 3-day output is currently a *hindcast* (it scores the most
recent 72 feature rows), not a true forward forecast. True multi-step forecasting is on the
Roadmap below.

---

## 🏗️ Architecture

    Open-Meteo (weather + air quality)
            │
            ▼
    hourly_feature_pipeline ──▶ Hopsworks Feature Store   (source of truth: data)
                                        │
                                        ▼
    daily_training_pipeline:
       restore features → train 4 models → CV evaluate → select best
       → promote (feature_config.json) → forecast → drift → alerts
                │                                   │
                ▼                                   ▼
       Hopsworks Model Registry            models/best_model/  (model + contract)
       (versioned + metrics)                        │
                                                     ▼
                                          Serving:
                                            • FastAPI  (/predict, /forecast)
                                            • Streamlit dashboard (5 pages)

    AQICN + OpenWeather ····▶ live "current conditions" panel (dashboard, display-only)

**Principle:** code in Git · features in the Hopsworks Feature Store · models in the
Hopsworks Model Registry. CI is stateless and reproduces state from Hopsworks.

---

## ✨ Features

- **Hourly feature pipeline** — Open-Meteo ingestion + lag/rolling/time features → feature store.
- **Daily training pipeline** — 4 models, leakage-free `TimeSeriesSplit` CV, anti-overfit configs, auto-select + promote.
- **Contract-driven inference** — each model ships a `feature_config.json` (features, scaling flag, metrics), so swapping the winning model needs zero code changes.
- **Drift monitoring** — KS + PSI data drift, baseline-relative model drift, alerts + retraining flag.
- **Two surfaces** — a 5-page Streamlit dashboard and a 6-endpoint FastAPI service.
- **Full automation** — both pipelines scheduled on GitHub Actions.

---

## 🧰 Tech Stack

| Layer | Tools |
|-------|-------|
| Data | Open-Meteo, AQICN, OpenWeather, pandas, pyarrow |
| ML | scikit-learn, XGBoost, LightGBM |
| Feature store / registry | Hopsworks |
| Explainability | SHAP, feature importances |
| Serving | FastAPI, Streamlit, Altair |
| MLOps | GitHub Actions, python-dotenv, Docker |

---

## 📂 Project Structure

    aqi-predictor/
    ├── data/processed/          # features + forecast (gitignored; restored from Hopsworks)
    ├── models/
    │   ├── best_model/          # promoted model + scaler + feature_config.json (contract)
    │   ├── saved_models/        # per-model artifacts
    │   └── *.json               # model comparison + best-model metadata
    ├── reports/drift/           # data/model drift + alert reports
    ├── src/
    │   ├── data_pipeline/       # historical + realtime ingestion
    │   ├── feature_engineering/ # feature creation + selection
    │   ├── feature_store/       # Hopsworks connection + upload
    │   ├── training/            # train_*, evaluate_models, select_best_model
    │   ├── models/              # registry upload/download
    │   ├── inference/           # load_model, predict, forecast_next_3_days
    │   ├── monitoring/          # data_drift, model_drift, alerts
    │   ├── api/                 # fastapi_app
    │   ├── dashboard/           # Streamlit Home + pages/
    │   └── automation/          # hourly_feature_pipeline, daily_training_pipeline
    ├── tests/                   # pytest smoke tests
    ├── .github/workflows/       # feature_pipeline, training_pipeline, tests, docker
    ├── Dockerfile
    └── requirements.txt

---

## 🤖 Models & Selection

Four models are trained and compared with **nested feature selection inside a 3-fold
`TimeSeriesSplit`** (no leakage) on a chronological train/dev/test split:

| Model | Notes |
|-------|-------|
| Ridge | scaled, alpha tuned via inner CV |
| Random Forest | depth-limited, subsampled |
| XGBoost | early stopping, regularized |
| LightGBM | early stopping, regularized |

**Selection metric:** smallest **train↔validation R² gap** (most stable generalization),
tie-broken by highest CV validation R². The winner is promoted to `models/best_model/`
with its inference contract and uploaded to the Hopsworks Model Registry.

Overfitting is judged by the **R² gap** (≤0.05 = healthy), not the RMSE ratio.

---

## 🚀 Setup

    git clone https://github.com/nikhatraghna/aqi-predictor.git
    cd aqi-predictor
    pip install -r requirements.txt
    pip install streamlit altair shap     # dashboard / explainability extras

Create a `.env` in the repo root (never commit it):

    HOPSWORKS_API_KEY=your_hopsworks_api_key_here
    HOPSWORKS_PROJECT=your_hopsworks_project_name
    AQICN_API_KEY=your_aqicn_token_here            # optional (live dashboard panel only)
    OPENWEATHER_API_KEY=your_openweather_key_here  # optional (live dashboard panel only)
    CITY=Islamabad

- **Hopsworks** (required): create a free project at hopsworks.ai and generate an API key.
- **AQICN / OpenWeather** (optional): power only the dashboard's live panel; the model uses Open-Meteo (no key).
- In **GitHub Actions**, store `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT` as **Repository Secrets**.

---

## ▶️ Usage

    # Ingest the latest hour into the feature store
    python -m src.automation.hourly_feature_pipeline

    # Full daily loop: train → evaluate → promote → forecast → monitor → register
    python -m src.automation.daily_training_pipeline

    # Inference / monitoring
    python -m src.inference.forecast_next_3_days
    python -m src.monitoring.alerts

    # Dashboard (Home · Forecast · Explanations · EDA · Monitoring)
    streamlit run src/dashboard/Home.py

    # API (Swagger UI at http://localhost:8000/docs)
    uvicorn src.api.fastapi_app:app --reload

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | health + loaded model |
| GET | `/model` | model name, metrics, features |
| GET | `/forecast` | latest 3-day PM2.5 forecast |
| POST | `/predict` | predict for supplied feature rows |
| GET | `/monitoring` | latest drift / alert status |
| GET | `/live` | live AQICN + OpenWeather snapshot |

---

## 📡 Monitoring

- **Data drift** — KS test + PSI per feature; flagged DRIFT only when both agree.
- **Model drift** — current RMSE vs the promoted model's baseline (≥20% → WARNING, ≥40% → DRIFT); rolling history saved.
- **Alerts** — aggregate status + `retrain_recommended` flag (triggered on sustained model drift only).

---

## ⚙️ Automation (GitHub Actions)

| Workflow | Schedule (UTC) | Action |
|----------|----------------|--------|
| feature_pipeline.yml | hourly `0 * * * *` | restore features → append latest hour → upload to feature store |
| training_pipeline.yml | daily `30 0 * * *` | refresh features → retrain → promote → forecast → monitor → upload to model registry |
| tests.yml | on push / PR | run pytest |
| docker.yml | on push | build the Docker image |

Both data pipelines are **stateless**: they restore state from Hopsworks, so no data/model binaries are committed to Git.

---

## 🗺️ Roadmap

- **True multi-step forecasting** — forecast-specific model on forward-knowable features + a weather-forecast feed (replaces the hindcast).
- **Multi-city** support (Rawalpindi, Lahore, Karachi).
- **Hosted dashboard** reading directly from Hopsworks.
- **Auto-retraining** wired to the monitoring `retrain_recommended` flag.

---

## 📝 Design Notes

- **Source consistency:** the model is trained and served on Open-Meteo data; AQICN/OpenWeather feed only the live dashboard panel.
- **Feature store as source of truth:** CI restores features from Hopsworks rather than committing parquet files.
- **Honest evaluation:** leakage-free nested CV, R²-gap overfitting check, and a clearly-labeled hindcast.
