"""Shared loaders, AQI helpers, and theme. Data is sourced from a Hopsworks bundle."""

from __future__ import annotations
import os, json, tempfile, zipfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ── Bootstrap: download the dashboard bundle from Hopsworks (once per session) ──
@st.cache_resource(show_spinner="Loading data from Hopsworks…")
def _bootstrap_from_hopsworks():
    import hopsworks
    try:
        sk = st.secrets.get("HOPSWORKS_API_KEY"); sp = st.secrets.get("HOPSWORKS_PROJECT")
    except Exception:
        sk = sp = None
    key  = os.getenv("HOPSWORKS_API_KEY")  or sk
    proj = os.getenv("HOPSWORKS_PROJECT") or sp
    if not key or not proj:
        st.error("Hopsworks credentials not found. In **Settings → Secrets** add:\n\n"
                 'HOPSWORKS_API_KEY = "..."\nHOPSWORKS_PROJECT = "aqi_forecasting_system"')
        st.stop()
    project = hopsworks.login(api_key_value=key, project=proj)
    ds  = project.get_dataset_api()
    tmp = Path(tempfile.mkdtemp())
    ds.download("Resources/aqi_dashboard/dashboard_bundle.zip", local_path=str(tmp), overwrite=True)
    with zipfile.ZipFile(tmp / "dashboard_bundle.zip") as zf:
        zf.extractall(tmp / "bundle")
    return tmp / "bundle"

PROJECT_ROOT = _bootstrap_from_hopsworks()          # ← defined BEFORE PATHS

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
    "current":       PROJECT_ROOT / "data/processed/current_conditions.json",
}
SHAP_CANDIDATES = [PROJECT_ROOT / "reports/shap/shap_importance.parquet",
                   PROJECT_ROOT / "reports/shap/shap_importance.csv",
                   PROJECT_ROOT / "reports/shap/shap_importance.json"]
RETRAIN_THRESHOLD = 3

AQI_BANDS = [(0,12,"Good","🟢","#2E9E5B"),(12,35,"Moderate","🟡","#C9A227"),
             (35,55,"Unhealthy (Sensitive)","🟠","#E8743B"),(55,150,"Unhealthy","🔴","#D7263D"),
             (150,250,"Very Unhealthy","🟣","#7B2CBF"),(250,100000,"Hazardous","⛔","#6A040F")]
US_AQI_BANDS = [(0,50,"Good","#2E9E5B"),(51,100,"Moderate","#C9A227"),(101,150,"Unhealthy (Sensitive)","#E8743B"),
                (151,200,"Unhealthy","#D7263D"),(201,300,"Very Unhealthy","#7B2CBF"),(301,10000,"Hazardous","#6A040F")]
STATUS_COLORS = {"NORMAL":"#2E9E5B","WARNING":"#C9A227","DRIFT":"#D7263D","UNKNOWN":"#9aa0a6"}
STATUS_EMOJI  = {"NORMAL":"🟢","WARNING":"🟡","DRIFT":"🔴","UNKNOWN":"⚪"}


# ── AQI helpers ──
def pm25_to_category(value):
    if value is None or (isinstance(value, float) and np.isnan(value)): return ("Unknown","⚪","#9aa0a6")
    for lo,hi,label,emoji,color in AQI_BANDS:
        if lo <= value < hi: return (label,emoji,color)
    return ("Hazardous","⛔","#6A040F")

def dominant_category(values):
    labels=[pm25_to_category(v)[0] for v in values if v is not None]
    if not labels: return ("Unknown","⚪","#9aa0a6")
    top=max(set(labels),key=labels.count)
    for _,_,l,e,c in AQI_BANDS:
        if l==top: return (l,e,c)
    return (top,"⚪","#9aa0a6")

def normalize_status(s):
    s=(s or "").upper()
    return "DRIFT" if "DRIFT" in s else "WARNING" if "WARNING" in s else "NORMAL" if "NORMAL" in s else "UNKNOWN"

def us_aqi_category(aqi):
    if aqi is None: return ("Unknown","#9aa0a6")
    for lo,hi,lbl,c in US_AQI_BANDS:
        if lo <= aqi <= hi: return (lbl,c)
    return ("Hazardous","#6A040F")

def feature_family(name):
    n=name.lower()
    if "lag" in n: return "Lagged PM2.5"
    if "roll" in n: return "Rolling Window Features"
    if any(k in n for k in ["hour","day","month","sin","cos","week"]): return "Calendar/Cyclical"
    if any(k in n for k in ["temp","humid","press","wind","cloud","precip","visib","weather"]): return "Weather Features"
    return "Pollutants"


# ── readers ──
def _read_json(path):
    p=Path(path)
    try: return json.load(open(p)) if p.exists() else None
    except Exception: return None

def _read_parquet(path):
    p=Path(path)
    try: return pd.read_parquet(p) if p.exists() else None
    except Exception: return None

def file_mtime(key):
    p=Path(PATHS.get(key,key))
    return datetime.fromtimestamp(p.stat().st_mtime) if p.exists() else None


# ── cached loaders ──
@st.cache_data(ttl=300)
def load_forecast():
    df=_read_parquet(PATHS["forecast"])
    if df is not None and "datetime" in df.columns: df["datetime"]=pd.to_datetime(df["datetime"])
    return df

@st.cache_data(ttl=300)
def load_features():
    df=_read_parquet(PATHS["features"])
    if df is not None and "datetime" in df.columns: df["datetime"]=pd.to_datetime(df["datetime"])
    return df
@st.cache_data(ttl=300)
def load_hindcast():
    df = _read_parquet(PROJECT_ROOT / "data/processed/hindcast_3days.parquet")
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
@st.cache_data(ttl=300)
def load_current_conditions(): return _read_json(PATHS["current"])

@st.cache_resource
def load_best_estimator():
    p=PATHS["model_pkl"]
    return joblib.load(p) if p.exists() else None

@st.cache_data(ttl=300)
def load_shap_importance():
    for p in SHAP_CANDIDATES:
        if not p.exists(): continue
        df = pd.read_parquet(p) if p.suffix==".parquet" else (pd.read_csv(p) if p.suffix==".csv" else pd.DataFrame(json.load(open(p))))
        cols={c.lower():c for c in df.columns}
        feat=cols.get("feature") or cols.get("feature_name")
        val=cols.get("importance") or cols.get("mean_abs_shap") or cols.get("shap")
        if feat and val:
            return df[[feat,val]].rename(columns={feat:"feature",val:"importance"}).sort_values("importance",ascending=False).reset_index(drop=True)
    return None

def compute_feature_importance(prefer_shap=True):
    if prefer_shap:
        s=load_shap_importance()
        if s is not None and not s.empty: return s,"SHAP"
    model=load_best_estimator(); contract=load_contract()
    if model is None or not contract: return None,None
    features=contract.get("features") or []
    if hasattr(model,"feature_importances_"): imp=np.asarray(model.feature_importances_,dtype=float)
    elif hasattr(model,"coef_"): imp=np.abs(np.asarray(model.coef_,dtype=float)).ravel()
    else: return None,None
    if len(imp)!=len(features): return None,None
    df=pd.DataFrame({"feature":features,"importance":imp}).sort_values("importance",ascending=False).reset_index(drop=True)
    return df,"model"


# ── theme ──
def inject_css():
    st.markdown("""
    <style>
      .stApp { background: linear-gradient(180deg,#0f1620 0%,#131c28 100%); color:#e6edf3; }
      .block-container { padding-top: 2rem; max-width: 1200px; }
      .hero-title { font-size:2.4rem; font-weight:800;
        background:linear-gradient(90deg,#5eead4,#38bdf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
      .hero-sub { color:#9fb0c0; margin-top:-.3rem; margin-bottom:.6rem; }
      .card { background:#1b2531; border:1px solid #283545; border-radius:16px; padding:1.1rem 1.3rem;
        box-shadow:0 6px 18px rgba(0,0,0,.25); height:100%; }
      .card h4 { margin:0 0 .4rem 0; color:#9fb0c0; font-size:.78rem; text-transform:uppercase; letter-spacing:.6px; }
      .big { font-size:2rem; font-weight:800; line-height:1.1; }
      .pill { display:inline-block; padding:.28rem .7rem; border-radius:999px; font-size:.78rem;
        font-weight:700; color:#0b0f14; margin:.15rem; }
      .banner { border-radius:14px; padding:.9rem 1.2rem; font-weight:700; border:1px solid rgba(255,255,255,.08); }
      .muted { color:#7e8c9b; font-size:.85rem; }
      div[data-testid="stSidebarNav"]::before { content:"🌫️  Islamabad AQI"; display:block;
        padding:.4rem .9rem 1rem .9rem; font-size:1.1rem; font-weight:800; color:#5eead4; border-bottom:1px solid #1f2a37; }
    </style>
    """, unsafe_allow_html=True)

def aqi_legend_html():
    pills="".join(f'<span class="pill" style="background:{c}">{e} {lbl}</span>' for _,_,lbl,e,c in AQI_BANDS)
    return f'<div style="margin:.4rem 0 1rem 0">{pills}</div>'

def status_card(title, status, sub=""):
    s=normalize_status(status); color,emoji=STATUS_COLORS[s],STATUS_EMOJI[s]
    return (f'<div class="card" style="border-left:5px solid {color}"><h4>{title}</h4>'
            f'<div class="big" style="color:{color}">{emoji} {s}</div><div class="muted">{sub}</div></div>')

def render_sidebar():
    contract=load_contract() or {}
    with st.sidebar:
        st.markdown("---")
        st.markdown(f'<div class="muted">Production model<br><b style="color:#9fb0c0">'
                    f'{str(contract.get("model_name","—")).upper()}</b><br><br>'
                    f'<span style="color:#5eead4">PM2.5 forecasting · MLOps</span></div>', unsafe_allow_html=True)
