import json
from pathlib import Path

REGISTRY_PATH = "models/model_registry.json"

def load_registry(path=REGISTRY_PATH):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Registry not found: {p}")
    with open(p) as f:
        return json.load(f)

def get_best_model_info(path=REGISTRY_PATH):
    r = load_registry(path)
    return r["best_model"], r["metrics"]

def get_best_model_name(path=REGISTRY_PATH):
    name, _ = get_best_model_info(path)
    return name

def get_best_model_metrics(path=REGISTRY_PATH):
    _, metrics = get_best_model_info(path)
    return metrics
