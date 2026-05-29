import json
from pathlib import Path


BEST_MODEL_PATH = "models/best_model.json"
BEST_METRICS_PATH = "models/best_model_metrics.json"


# ─────────────────────────────────────────
# LOAD BEST MODEL FILE
# ─────────────────────────────────────────

def load_best_model_file(path: str = BEST_MODEL_PATH) -> dict:
    """Load best model selection file."""

    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(
            f"Best model file not found: {p}"
        )

    with open(p, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────
# LOAD METRICS FILE
# ─────────────────────────────────────────

def load_best_metrics_file(path: str = BEST_METRICS_PATH) -> dict:
    """Load best model metrics file."""

    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(
            f"Best metrics file not found: {p}"
        )

    with open(p, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────

def get_best_model_name() -> str:
    """Return best model name."""

    data = load_best_model_file()

    return data["best_model"]


def get_best_model_metrics() -> dict:
    """Return best model metrics."""

    return load_best_metrics_file()


def get_best_model_info() -> tuple:
    """Return (model_name, metrics)."""

    return (
        get_best_model_name(),
        get_best_model_metrics()
    )
