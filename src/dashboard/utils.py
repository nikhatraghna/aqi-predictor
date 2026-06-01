"""Shared loaders, AQI helpers, and theme for the Islamabad PM2.5 dashboard.

Every loader reads a local project artifact and returns None (never raises) when the
file is missing, so pages degrade gracefully. Heavy objects use Streamlit caching.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os, tempfile, zipfile

# Repo root = two levels up from src/dashboard/utils.py → paths are cwd-independent
#PROJECT_ROOT = Path(__file__).resolve().parents[2]
#the above line was replaced by the below block
@st.cache_resource(show_spinner="Loading data from Hopsworks…")
def _bootstrap_from_hopsworks():
    """Download the dashboard bundle from Hopsworks and unzip it. Returns its root path."""
    import hopsworks
    key  = os.getenv("HOPSWORKS_API_KEY")  or st.secrets.get("HOPSWORKS_API_KEY")
    proj = os.getenv("HOPSWORKS_PROJECT") or st.secrets.get("HOPSWORKS_PROJECT")
    project = hopsworks.login(api_key_value=key, project=proj)
    ds  = project.get_dataset_api()
    tmp = Path(tempfile.mkdtemp())
    ds.download("Resources/aqi_dashboard/dashboard_bundle.zip", local_path=str(tmp), overwrite=True)
    with zipfile.ZipFile(tmp / "dashboard_bundle.zip") as zf:
        zf.extractall(tmp / "bundle")
    return tmp / "bundle"

PROJECT_ROOT = _bootstrap_from_hopsworks()

PATHS = {
    "forecast":      PROJECT_ROOT / "data/processed/forecast_3days.parquet",
    "features":      PROJECT_ROOT / "data/processed/islamabad_features.parquet",
    "model_metrics": PROJECT_ROOT / "models/model_metrics.json",
    "best_metrics":  PROJECT_ROOT / "models/best_model_metrics.json",
    "contract":      PROJECT_ROOT / "models/best_model/feature_config.json",
    "model_pkl":     PROJECT_ROOT / "models/best_model/model.pkl",
    "alerts":        PROJECT_ROOT / "reports/drift/alerts_report.json",
    "model_drift":   PROJECT_ROOT / "reports/drift/model_drift_report.json",
    "data_drift":    PROJECT_ROOT / "reports/drift/data_drift_report.json",
    "drift_history": PROJECT_ROOT / "reports/drift/model_drift_history.parquet",
    "drift_state":   PROJECT_ROOT / "reports/drift/drift_state.json",
}

SHAP_CANDIDATES = [
    PROJECT_ROOT / "reports/shap/shap_importance.parquet",
    PROJECT_ROOT / "reports/shap/shap_importance.csv",
    PROJECT_ROOT / "reports/shap/shap_importance.json",
]

RETRAIN_THRESHOLD = 3  # consecutive DRIFT runs before retraining (mirrors monitoring config)

# (lo, hi, label, emoji, color)
AQI_BANDS = [
    (0,    12,     "Good",                  "🟢", "#2E9E5B"),
    (12,   35,     "Moderate",              "🟡", "#C9A227"),
    (35,   55,     "Unhealthy (Sensitive)", "🟠", "#E8743B"),
    (55,   150,    "Unhealthy",             "🔴", "#D7263D"),
    (150,  250,    "Very Unhealthy",        "🟣", "#7B2CBF"),
    (250,  100000, "Hazardous",             "⛔", "#6A040F"),
]

STATUS_COLORS = {"NORMAL": "#2E9E5B", "WARNING": "#C9A227", "DRIFT": "#D7263D", "UNKNOWN": "#9aa0a6"}
STATUS_EMOJI  = {"NORMAL": "🟢", "WARNING": "🟡", "DRIFT": "🔴", "UNKNOWN": "⚪"}


# ── AQI helpers ─────────────────────────────────────────────────────────────
def pm25_to_category(value):
    """Return (label, emoji, color) for a PM2.5 value (µg/m³)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ("Unknown", "⚪", "#9aa0a6")
    for lo, hi, label, emoji, color in AQI_BANDS:
        if lo <= value < hi:
            return (label, emoji, color)
    return ("Hazardous", "⛔", "#6A040F")


def dominant_category(values):
    """Return the most frequent (label, emoji, color) across a series of PM2.5 values."""
    labels = [pm25_to_category(v)[0] for v in values if v is not None]
    if not labels:
        return ("Unknown", "⚪", "#9aa0a6")
    top = max(set(labels), key=labels.count)
    for _, _, label, emoji, color in AQI_BANDS:
        if label == top:
            return (label, emoji, color)
    return (top, "⚪", "#9aa0a6")


def normalize_status(s):
    """Map any '🟡 WARNING'-style string to NORMAL / WARNING / DRIFT / UNKNOWN."""
    s = (s or "").upper()
    if "DRIFT" in s:   return "DRIFT"
    if "WARNING" in s: return "WARNING"
    if "NORMAL" in s:  return "NORMAL"
    return "UNKNOWN"


def feature_family(name):
    """Group a feature name into a production-friendly family."""
    n = name.lower()
    if "lag" in n:                                                  return "Lagged PM2.5"
    if "roll" in n:                                                 return "Rolling Window Features"
    if any(k in n for k in ["hour", "day", "month", "sin", "cos", "week"]): return "Calendar/Cyclical"
    if any(k in n for k in ["temp", "humid", "press", "wind", "cloud", "precip", "visib", "weather"]): return "Weather Features"
    return "Pollutants"


# ── Low-level readers (never raise) ─────────────────────────────────────────
def _read_json(path):
    p = Path(path)
    try:
        return json.load(open(p)) if p.exists() else None
    except (json.JSONDecodeError, OSError):
        return None


def _read_parquet(path):
    p = Path(path)
    try:
        return pd.read_parquet(p) if p.exists() else None
    except Exception:
        return None


def file_mtime(key):
    """Last-modified time of a PATHS key (or raw path), or None if absent."""
    p = Path(PATHS.get(key, key))
    return datetime.fromtimestamp(p.stat().st_mtime) if p.exists() else None


# ── Cached data loaders ──────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_forecast():
    df = _read_parquet(PATHS["forecast"])
    if df is not None and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_data(ttl=300)
def load_features():
    df = _read_parquet(PATHS["features"])
    if df is not None and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_data(ttl=300)
def load_model_metrics(): return _read_json(PATHS["model_metrics"])

@st.cache_data(ttl=300)
def load_best_metrics():  return _read_json(PATHS["best_metrics"])

@st.cache_data(ttl=300)
def load_contract():      return _read_json(PATHS["contract"])

@st.cache_data(ttl=300)
def load_alerts():        return _read_json(PATHS["alerts"])

@st.cache_data(ttl=300)
def load_model_drift():   return _read_json(PATHS["model_drift"])

@st.cache_data(ttl=300)
def load_data_drift():    return _read_json(PATHS["data_drift"])

@st.cache_data(ttl=300)
def load_drift_state():   return _read_json(PATHS["drift_state"])

@st.cache_data(ttl=300)
def load_drift_history(): return _read_parquet(PATHS["drift_history"])


@st.cache_resource
def load_best_estimator():
    """Load the promoted model object once per session (cached resource)."""
    p = PATHS["model_pkl"]
    return joblib.load(p) if p.exists() else None


@st.cache_data(ttl=300)
def load_shap_importance():
    """Load precomputed SHAP importance if an artifact exists, normalized to
    columns [feature, importance]. Returns None if no SHAP artifact is present."""
    for p in SHAP_CANDIDATES:
        if not p.exists():
            continue
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix == ".csv":
            df = pd.read_csv(p)
        else:
            df = pd.DataFrame(json.load(open(p)))
        cols = {c.lower(): c for c in df.columns}
        feat = cols.get("feature") or cols.get("feature_name")
        val  = cols.get("importance") or cols.get("mean_abs_shap") or cols.get("shap")
        if feat and val:
            out = df[[feat, val]].rename(columns={feat: "feature", val: "importance"})
            return out.sort_values("importance", ascending=False).reset_index(drop=True)
    return None


def compute_feature_importance(prefer_shap=True):
    """Return (DataFrame[feature, importance], source) where source is 'SHAP' or 'model'.

    Prefers a SHAP artifact when present; otherwise falls back to the model's native
    feature_importances_ (trees) or |coef_| (linear). Returns (None, None) if neither
    is available.
    """
    if prefer_shap:
        shap_df = load_shap_importance()
        if shap_df is not None and not shap_df.empty:
            return shap_df, "SHAP"

    model    = load_best_estimator()
    contract = load_contract()
    if model is None or not contract:
        return None, None
    features = contract.get("features") or []

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        imp = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        return None, None
    if len(imp) != len(features):
        return None, None

    df = pd.DataFrame({"feature": features, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df, "model"


# ── Theme ─────────────────────────────────────────────────────────────────────
def inject_css():
    """Inject the shared dark slate + teal theme used across all pages."""
    st.markdown("""
    <style>
      .stApp { background: linear-gradient(180deg,#0f1620 0%,#131c28 100%); color:#e6edf3; }
      .block-container { padding-top: 2rem; max-width: 1200px; }
      .hero-title { font-size: 2.4rem; font-weight: 800; letter-spacing:-.5px;
                    background: linear-gradient(90deg,#5eead4,#38bdf8);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
      .hero-sub   { color:#9fb0c0; font-size:1.02rem; margin-top:-.3rem; margin-bottom:.6rem; }
      .card { background:#1b2531; border:1px solid #283545; border-radius:16px;
              padding:1.1rem 1.3rem; box-shadow:0 6px 18px rgba(0,0,0,.25); height:100%; }
      .card h4 { margin:0 0 .4rem 0; color:#9fb0c0; font-weight:600; font-size:.78rem;
                 text-transform:uppercase; letter-spacing:.6px; }
      .big { font-size:2.0rem; font-weight:800; line-height:1.1; }
      .pill { display:inline-block; padding:.28rem .7rem; border-radius:999px;
              font-size:.78rem; font-weight:700; color:#0b0f14; margin:.15rem; }
      .banner { border-radius:14px; padding:.9rem 1.2rem; font-weight:700;
                border:1px solid rgba(255,255,255,.08); }
      .muted { color:#7e8c9b; font-size:.85rem; }
      /* ── Sidebar branding + nav restyle ── */
section[data-testid="stSidebar"] { background:#0c131c; border-right:1px solid #1f2a37; }

/* Branded header injected above the auto page-nav */
div[data-testid="stSidebarNav"]::before{
  content:"🌫️  Islamabad AQI";
  display:block; padding:.4rem .9rem 1rem .9rem; margin-bottom:.4rem;
  font-size:1.15rem; font-weight:800; color:#5eead4;
  border-bottom:1px solid #1f2a37;
}
/* Nav links */
div[data-testid="stSidebarNav"] a{ border-radius:10px; margin:2px 6px; padding:.35rem .6rem; }
div[data-testid="stSidebarNav"] a:hover{ background:#16202c; }
div[data-testid="stSidebarNav"] a[aria-current="page"]{
  background:linear-gradient(90deg,#0e7490,#155e75); color:#e6fffb !important; font-weight:700;
}
    </style>
    """, unsafe_allow_html=True)


def aqi_legend_html():
    """Return HTML for the AQI category legend pills."""
    pills = "".join(
        f'<span class="pill" style="background:{c}">{e} {lbl}</span>'
        for _, _, lbl, e, c in AQI_BANDS
    )
    return f'<div style="margin:.4rem 0 1rem 0">{pills}</div>'


def status_card(title, status, sub=""):
    """Return HTML for a colored status card (NORMAL/WARNING/DRIFT)."""
    s = normalize_status(status)
    color, emoji = STATUS_COLORS[s], STATUS_EMOJI[s]
    return (f'<div class="card" style="border-left:5px solid {color}"><h4>{title}</h4>'
            f'<div class="big" style="color:{color}">{emoji} {s}</div>'
            f'<div class="muted">{sub}</div></div>')
def render_sidebar():
    """Branded footer below the auto page-nav (call once per page after inject_css)."""
    contract = load_contract() or {}
    updated  = file_mtime("forecast") or file_mtime("model_metrics")
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            f'<div style="font-size:.82rem;color:#7e8c9b;line-height:1.6">'
            f'<b style="color:#9fb0c0">Production model</b><br>'
            f'{str(contract.get("model_name","—")).upper()}<br>'
            f'<b style="color:#9fb0c0">Updated</b><br>'
            f'{updated.strftime("%b %d, %H:%M") if updated else "—"}<br><br>'
            f'<span style="color:#5eead4">PM2.5 forecasting · MLOps</span>'
            f'</div>', unsafe_allow_html=True)          
