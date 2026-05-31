"""Inference smoke test — skipped if artifacts missing."""
from pathlib import Path
import numpy as np
import pytest

ROOT     = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "models/best_model/feature_config.json"
FEATURES = ROOT / "data/processed/islamabad_features.parquet"


@pytest.mark.skipif(not (CONTRACT.exists() and FEATURES.exists()),
                    reason="promoted model or feature data not present")
def test_predict_shape_and_finiteness():
    import pandas as pd
    from src.inference.predict import predict

    df = pd.read_parquet(FEATURES).tail(5).copy()
    preds = predict(df)

    assert len(preds) == len(df)
    assert np.isfinite(np.asarray(preds, dtype=float)).all()
