"""Islamabad PM2.5 — Home: health status, best model, 3-day forecast overview."""

import altair as alt
import pandas as pd
import streamlit as st

from utils import (
    load_forecast, load_contract, load_model_metrics, load_alerts,
    pm25_to_category, normalize_status, aqi_legend_html, inject_css, STATUS_COLORS, render_sidebar,
)

st.set_page_config(page_title="Islamabad PM2.5 Forecast", page_icon="🌫️", layout="wide")
inject_css()
render_sidebar()
# ── Header ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌫️ Islamabad Air Quality</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">3-day PM2.5 forecast • model health • automatically selected best model</div>',
            unsafe_allow_html=True)
st.write("")

# ── Health banner (from alerts_report.json) ────────────────────────────────
alerts = load_alerts()
if alerts:
    status = normalize_status(alerts.get("overall_status"))
    color  = STATUS_COLORS.get(status, "#9aa0a6")
    retrain = alerts.get("retrain_recommended", False)
    msg = "Retraining recommended" if retrain else "No action required"
    st.markdown(
        f'<div class="banner" style="background:{color}22;border-color:{color}">'
        f'System health: <span style="color:{color}">{status}</span> — {msg}</div>',
        unsafe_allow_html=True)
else:
    st.info("No monitoring report yet — run `src.monitoring.alerts` to populate system health.")

st.write("")

# ── Best model + comparison ────────────────────────────────────────────────
contract = load_contract()
metrics  = load_model_metrics()

c1, c2 = st.columns([1, 1.4])
with c1:
    st.markdown('<div class="card"><h4>Production model</h4>', unsafe_allow_html=True)
    if contract:
        tm = contract.get("test_metrics") or {}
        st.markdown(f'<div class="big">{contract.get("model_name","?").upper()}</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="muted">selected by <b>{contract.get("selected_by","?")}</b></div>'
            f'<div style="margin-top:.5rem">Test R²: <b>{tm.get("r2","?")}</b> &nbsp;|&nbsp; '
            f'Test RMSE: <b>{tm.get("rmse","?")}</b><br>'
            f'CV Val R²: <b>{contract.get("cv_val_r2","?")}</b> &nbsp;|&nbsp; '
            f'CV gap: <b>{contract.get("cv_r2_gap","?")}</b></div>',
            unsafe_allow_html=True)
    else:
        st.write("No model promoted yet. Run `select_best_model.py`.")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card"><h4>Model comparison</h4>', unsafe_allow_html=True)
    if metrics:
        mdf = pd.DataFrame(metrics)
        show_cols = [c for c in ["Model", "RMSE", "R2", "R2_gap", "CV_Val_R2", "CV_Gap"] if c in mdf.columns]
        st.dataframe(mdf[show_cols], hide_index=True, use_container_width=True)
    else:
        st.write("No comparison yet. Run `evaluate_models.py`.")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.markdown("#### AQI categories")
st.markdown(aqi_legend_html(), unsafe_allow_html=True)

# ── 3-day forecast overview ────────────────────────────────────────────────
st.markdown("#### 3-day PM2.5 forecast")
fc = load_forecast()
if fc is None or fc.empty:
    st.warning("No forecast yet — run `src.inference.forecast_next_3_days`.")
else:
    avg  = float(fc["predicted_pm25"].mean())
    peak = float(fc["predicted_pm25"].max())
    a_lbl, a_emo, a_col = pm25_to_category(avg)
    p_lbl, p_emo, p_col = pm25_to_category(peak)

    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(f'<div class="card" style="border-left:5px solid {a_col}"><h4>72h average</h4>'
                    f'<div class="big" style="color:{a_col}">{avg:.1f}</div>'
                    f'<div class="muted">{a_emo} {a_lbl} · µg/m³</div></div>', unsafe_allow_html=True)
    with h2:
        st.markdown(f'<div class="card" style="border-left:5px solid {p_col}"><h4>Peak (worst hour)</h4>'
                    f'<div class="big" style="color:{p_col}">{peak:.1f}</div>'
                    f'<div class="muted">{p_emo} {p_lbl} · µg/m³</div></div>', unsafe_allow_html=True)
    with h3:
        rng = f'{fc["datetime"].min():%b %d %H:%M} → {fc["datetime"].max():%b %d %H:%M}'
        st.markdown(f'<div class="card"><h4>Forecast window</h4>'
                    f'<div style="font-size:1.05rem;font-weight:700">{len(fc)} hours</div>'
                    f'<div class="muted">{rng}</div></div>', unsafe_allow_html=True)

    st.write("")
    chart = (
        alt.Chart(fc).mark_area(
            line={"color": "#38bdf8"},
            color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color="#38bdf833", offset=0),
                       alt.GradientStop(color="#5eead400", offset=1)],
                x1=1, x2=1, y1=1, y2=0),
        ).encode(
            x=alt.X("datetime:T", title=None),
            y=alt.Y("predicted_pm25:Q", title="PM2.5 (µg/m³)"),
            tooltip=["datetime:T", "predicted_pm25:Q", "category:N"],
        ).properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Note: forecast is computed on the most recent feature rows (hindcast), "
               "not live future data — live forecasting comes with the hourly feature pipeline.")
