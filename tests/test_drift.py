"""PSI drift detection math tests."""
import numpy as np
from src.monitoring.data_drift import compute_psi


def test_psi_near_zero_for_identical():
    rng = np.random.default_rng(0)
    x = rng.normal(50, 10, 2000)
    assert compute_psi(x, x) < 0.01


def test_psi_large_for_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(50, 10, 2000)
    b = rng.normal(85, 10, 2000)
    assert compute_psi(a, b) > 0.2


def test_psi_handles_empty():
    assert compute_psi(np.array([]), np.array([])) == 0.0
