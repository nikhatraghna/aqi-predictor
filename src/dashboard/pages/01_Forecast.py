"""Detailed 3-day PM2.5 forecast: model metadata, KPIs, hourly chart, table, export."""

import altair as alt
import streamlit as st

from utils import (
    load_forecast, load_contract, pm25_to_category, dominant_category,
    aqi_legend_html, inject_css, file_mtime, render_sidebar,
)

st.set_page_config(page_title="Forecast • Islamabad PM2.5", page_icon="🔮", layout="wide")
inject_css()
render_sidebar()
st.markdown('<div class="hero-title">🔮 Detailed 3-Day Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Hour-by-hour PM2.5 outlook for Islamabad</div>', unsafe_allow_html=True)

fc = load_forecast()
if fc is None or fc.empty:
    st.warning("No forecast available — run `python -m src.inference.forecast_next_3_days`.")
    st.stop()

fc = fc.sort_values("datetime").reset_index(drop=True)
_cats = fc["predicted_pm25"].apply(pm25_to_category)
fc["category"] = [c[0] for c in _cats]
fc["color"]    = [c[2] for c in _cats]
fc["day"]      = fc["datetime"].dt.date

# ── Model metadata ──────────────────────────────────────────────────────────
contract = load_contract() or {}
generated = file_mtime("forecast")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Production model", str(contract.get("model_name", "—")).upper())
m2.metric("Model version", str(contract.get("version", "local")))
m3.metric("Generated", generated.strftime("%b %d %H:%M") if generated else "—")
m4.metric("Forecast hours", len(fc))

# ── Forecast summary KPIs ───────────────────────────────────────────────────
avg, mx, mn = fc["predicted_pm25"].mean(), fc["predicted_pm25"].max(), fc["predicted_pm25"].min()
dom_lbl, dom_emo, _ = dominant_category(fc["predicted_pm25"])
k1, k2, k3, k4 = st.columns(4)
k1.metric("Average PM2.5", f"{avg:.1f}")
k2.metric("Maximum PM2.5", f"{mx:.1f}")
k3.metric("Minimum PM2.5", f"{mn:.1f}")
k4.metric("Dominant AQI", f"{dom_emo} {dom_lbl}")

st.markdown(aqi_legend_html(), unsafe_allow_html=True)
st.caption(f"Window: {fc['datetime'].min():%Y-%m-%d %H:%M} → "
           f"{fc['datetime'].max():%Y-%m-%d %H:%M}  ·  {len(fc)} hours")

# ── Daily outlook cards ─────────────────────────────────────────────────────
st.markdown("#### Daily outlook")
day_groups = list(fc.groupby("day"))
for col, (day, grp) in zip(st.columns(len(day_groups)), day_groups):
    d_avg = grp["predicted_pm25"].mean()
    lbl, emo, c = pm25_to_category(d_avg)
    with col:
        st.markdown(
            f'<div class="card" style="border-left:5px solid {c}"><h4>{day:%a · %b %d}</h4>'
            f'<div class="big" style="color:{c}">{d_avg:.1f}</div>'
            f'<div class="muted">{emo} {lbl} · avg µg/m³<br>'
            f'min {grp.predicted_pm25.min():.0f} · max {grp.predicted_pm25.max():.0f}</div></div>',
            unsafe_allow_html=True)

# ── Hourly chart (colored by AQI band, rich tooltips) ───────────────────────
st.markdown("#### Hourly forecast")
tooltip = [
    alt.Tooltip("datetime:T", title="Time", format="%a %b %d, %H:%M"),
    alt.Tooltip("predicted_pm25:Q", title="PM2.5 (µg/m³)", format=".1f"),
    alt.Tooltip("category:N", title="AQI category"),
]
base = alt.Chart(fc).encode(x=alt.X("datetime:T", title="Date / time"))
line = base.mark_line(color="#38bdf8", opacity=0.45, interpolate="monotone").encode(
    y=alt.Y("predicted_pm25:Q", title="PM2.5 (µg/m³)"))
pts = base.mark_circle(size=60).encode(
    y="predicted_pm25:Q",
    color=alt.Color("color:N", scale=None, legend=None),
    tooltip=tooltip)
st.altair_chart(
    (line + pts).properties(height=360).interactive(bind_y=False),
    width='stretch')

from utils import load_hindcast
hb = load_hindcast()
if hb is not None and not hb.empty:
    st.markdown("#### Model backtest — predicted vs actual (last 72 observed hours)")
    m = hb.melt("datetime", value_vars=["actual_pm25", "predicted_pm25"],
                var_name="series", value_name="pm25")
    st.altair_chart(
        alt.Chart(m).mark_line().encode(
            x=alt.X("datetime:T", title=None),
            y=alt.Y("pm25:Q", title="PM2.5 (µg/m³)"),
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(range=["#38bdf8", "#f59e0b"]))
        ).properties(height=300), width='stretch')
    mae = (hb["actual_pm25"] - hb["predicted_pm25"]).abs().mean()
    st.caption(f"Backtest MAE: {mae:.2f} µg/m³ — how closely the model tracked the last 72 observed hours.")
# ── Hourly table + export ───────────────────────────────────────────────────
st.markdown("#### Hourly table")
cols = [c for c in ["datetime", "predicted_pm25", "category", "status"] if c in fc.columns]
st.dataframe(fc[cols], hide_index=True, width='stretch',
             column_config={"predicted_pm25": st.column_config.NumberColumn("PM2.5 (µg/m³)", format="%.1f"),
                            "datetime": st.column_config.DatetimeColumn("Time", format="MMM D, HH:mm")})
st.download_button("⬇️ Download forecast CSV", fc[cols].to_csv(index=False).encode(),
                   "islamabad_forecast_3days.csv", "text/csv")
