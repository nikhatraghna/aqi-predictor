"""Schema checks for pipeline outputs — skipped if artifacts absent."""
import json
from pathlib import Path
import pytest

ROOT     = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "models/best_model/feature_config.json"
METRICS  = ROOT / "models/model_metrics.json"
FORECAST = ROOT / "data/processed/forecast_3days.parquet"


@pytest.mark.skipif(not CONTRACT.exists(), reason="no promoted model yet")
def test_feature_config_contract():
    c = json.loads(CONTRACT.read_text())
    for key in ("model_name", "features", "requires_scaling", "target"):
        assert key in c, f"missing '{key}' in feature_config.json"
    assert isinstance(c["features"], list) and c["features"]
    assert isinstance(c["requires_scaling"], bool)


@pytest.mark.skipif(not METRICS.exists(), reason="no model comparison yet")
def test_model_metrics_schema():
    rows = json.loads(METRICS.read_text())
    assert isinstance(rows, list) and rows
    for r in rows:
        assert {"Model", "RMSE", "R2"}.issubset(r.keys())


@pytest.mark.skipif(not FORECAST.exists(), reason="no forecast yet")
def test_forecast_schema():
    import pandas as pd
    df = pd.read_parquet(FORECAST)
    assert {"datetime", "predicted_pm25"}.issubset(df.columns)
    assert len(df) > 0
    assert df["predicted_pm25"].notna().all()
