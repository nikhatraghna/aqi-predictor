# ── Islamabad AQI Forecasting — container image ──
FROM python:3.11-slim

# build-essential: gcc to compile source-only deps (twofish, via hopsworks→pyjks)
# libgomp1: required by LightGBM / XGBoost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
