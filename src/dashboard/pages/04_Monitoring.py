"""MLOps monitoring: data drift, model drift, alerting, and retraining readiness."""

import altair as alt
import pandas as pd
import streamlit as st

from utils import (
    load_alerts, load_model_drift, load_data_drift, load_drift_state,
    load_drift_history, normalize_status, status_card, file_mtime,
    inject_css, STATUS_COLORS, RETRAIN_THRESHOLD, render_sidebar,
)

st.set_page_config(page_title="Monitoring • Islamabad PM2.5", page_icon="📡", layout="wide")
inject_css()
render_sidebar()
st.markdown('<div class="hero-title">📡 Model & Data Monitoring</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Drift detection · alerting · retraining readiness</div>',
            unsafe_allow_html=True)

alerts = load_alerts()
md     = load_model_drift()
dd     = load_data_drift()

if not any([alerts, md, dd]):
    st.warning("No monitoring reports yet — run `data_drift`, `model_drift`, then `alerts`.")
    st.stop()

alerts = alerts or {}
md     = md or {}
dd     = dd or []

# ── Derived KPIs ─────────────────────────────────────────────────────────────
overall   = normalize_status(alerts.get("overall_status"))
data_sev  = normalize_status((alerts.get("data_drift") or {}).get("severity"))
model_sev = normalize_status((alerts.get("model_drift") or {}).get("severity") or md.get("status"))
rmse_deg  = md.get("rmse_degradation_pct")
psi_vals  = [r.get("psi") for r in dd if isinstance(r, dict) and r.get("psi") is not None]
ks_vals   = [r.get("ks_pvalue") for r in dd if isinstance(r, dict) and r.get("ks_pvalue") is not None]
max_psi   = max(psi_vals) if psi_vals else None
min_ksp   = min(ks_vals) if ks_vals else None
last_run  = file_mtime("model_drift") or file_mtime("alerts") or file_mtime("data_drift")

# ── KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Overall", overall)
k2.metric("Data drift", data_sev)
k3.metric("Model drift", model_sev)
k4.metric("RMSE degradation", f"{rmse_deg:.1f}%" if rmse_deg is not None else "—")
k5.metric("Max PSI", f"{max_psi:.3f}" if max_psi is not None else "—")
k6.metric("Last run", last_run.strftime("%b %d %H:%M") if last_run else "—")

# ── Status cards ─────────────────────────────────────────────────────────────
st.markdown("#### Drift status")
c1, c2, c3 = st.columns(3)
c1.markdown(status_card("Overall system", overall,
            f"min KS p={min_ksp:.3f}" if min_ksp is not None else ""), unsafe_allow_html=True)
c2.markdown(status_card("Data drift", data_sev,
            f"{(alerts.get('data_drift') or {}).get('drifted','?')}/"
            f"{(alerts.get('data_drift') or {}).get('total','?')} features"), unsafe_allow_html=True)
c3.markdown(status_card("Model drift", model_sev,
            f"baseline RMSE {md.get('baseline_rmse','?')} → current "
            f"{(md.get('current_metrics') or {}).get('rmse','?')}"), unsafe_allow_html=True)

# ── RMSE degradation vs thresholds ──────────────────────────────────────────
st.markdown("#### Model performance degradation")
if rmse_deg is not None:
    deg_df = pd.DataFrame({"metric": ["RMSE degradation"], "value": [rmse_deg]})
    bar = alt.Chart(deg_df).mark_bar(color=STATUS_COLORS[model_sev], cornerRadiusEnd=4).encode(
        x=alt.X("value:Q", title="% worse than baseline",
                scale=alt.Scale(domain=[min(0, rmse_deg), max(50, rmse_deg + 10)])),
        y=alt.Y("metric:N", title=None),
        tooltip=[alt.Tooltip("value:Q", format=".2f")])
    rules = alt.Chart(pd.DataFrame({"t": [20, 40], "label": ["WARNING", "DRIFT"]})).mark_rule(
        strokeDash=[4, 4], color="#9fb0c0").encode(x="t:Q")
    st.altair_chart((bar + rules).properties(height=120), use_container_width=True)
    st.caption("Dashed lines: 20% → WARNING, 40% → DRIFT (relative to baseline test RMSE).")
else:
    st.info("No model-drift performance metrics available yet.")

# ── Distribution drift summary (PSI per feature) ────────────────────────────
st.markdown("#### Feature distribution drift (PSI)")
if dd:
    ddf = pd.DataFrame(dd)
    if "psi" in ddf.columns:
        ddf["sev"] = ddf["status"].apply(normalize_status) if "status" in ddf.columns else "UNKNOWN"
        chart = alt.Chart(ddf).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("psi:Q", title="PSI (≥0.2 = significant)"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.Color("sev:N", title="Status",
                            scale=alt.Scale(domain=list(STATUS_COLORS.keys()),
                                            range=list(STATUS_COLORS.values()))),
            tooltip=[alt.Tooltip("feature:N"),
                     alt.Tooltip("psi:Q", format=".3f"),
                     alt.Tooltip("ks_pvalue:Q", title="KS p", format=".4f")
                     if "ks_pvalue" in ddf.columns else alt.Tooltip("psi:Q"),
                     alt.Tooltip("status:N") if "status" in ddf.columns else alt.Tooltip("psi:Q")],
        ).properties(height=max(220, 26 * len(ddf)))
        ref = alt.Chart(pd.DataFrame({"t": [0.2]})).mark_rule(
            strokeDash=[4, 4], color="#D7263D").encode(x="t:Q")
        st.altair_chart((chart + ref).properties(height=max(240, 26 * len(ddf))),
                        use_container_width=True)
    else:
        st.dataframe(ddf, hide_index=True, use_container_width=True)
else:
    st.info("No data-drift report available.")

# ── Trends over time (if history is tracked) ────────────────────────────────
st.markdown("#### Drift history")
hist = load_drift_history()
if hist is not None and not hist.empty and "timestamp" in hist.columns:
    hist = hist.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    cols = st.columns(2)
    if "rmse_degradation_pct" in hist.columns:
        cols[0].altair_chart(
            alt.Chart(hist).mark_line(point=True, color="#f59e0b").encode(
                x=alt.X("timestamp:T", title=None),
                y=alt.Y("rmse_degradation_pct:Q", title="RMSE degradation %")).properties(height=240),
            use_container_width=True)
    if "psi" in hist.columns:
        cols[1].altair_chart(
            alt.Chart(hist).mark_line(point=True, color="#5eead4").encode(
                x=alt.X("timestamp:T", title=None),
                y=alt.Y("psi:Q", title="PSI")).properties(height=240),
            use_container_width=True)
else:
    st.info("History tracking not enabled yet. Trends appear once the monitoring pipeline "
            "runs on a schedule and appends to `reports/drift/model_drift_history.parquet`.")

# ── Retraining readiness ────────────────────────────────────────────────────
st.markdown("#### Retraining readiness")
state = load_drift_state() or {}
consecutive = state.get("consecutive_drift_runs")
retrain = alerts.get("retrain_recommended", False)

r1, r2, r3 = st.columns(3)
r1.metric("Consecutive DRIFT runs",
          consecutive if consecutive is not None else "n/a")
r2.metric("Retrain threshold", RETRAIN_THRESHOLD)
r3.metric("Retraining recommended", "YES" if retrain else "No")

if consecutive is not None:
    st.progress(min(consecutive / RETRAIN_THRESHOLD, 1.0),
                text=f"{consecutive}/{RETRAIN_THRESHOLD} consecutive DRIFT runs toward retraining")
if retrain:
    st.error("⚠️ Retraining is recommended — sustained model drift detected.")
else:
    st.success("✅ No retraining required at this time.")
