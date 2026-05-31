# ── Islamabad AQI Forecasting — container image ──
FROM python:3.11-slim

# libgomp1 is required by LightGBM/XGBoost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime deps (+ API server)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# Copy the project
COPY . .

EXPOSE 8000

# Default: serve the FastAPI app. Override CMD to run a pipeline, e.g.:
#   docker run --env-file .env aqi-forecaster python -m src.automation.daily_training_pipeline
CMD ["uvicorn", "src.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
