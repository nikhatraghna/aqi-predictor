"""EDA: dataset overview, data quality, and distribution/temporal analysis."""

import altair as alt
import pandas as pd
import streamlit as st

from utils import load_features, pm25_to_category, feature_family, inject_css, render_sidebar

st.set_page_config(page_title="EDA • Islamabad PM2.5", page_icon="📊", layout="wide")
inject_css()
render_sidebar()
st.markdown('<div class="hero-title">📊 Data Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Overview, data quality, and patterns in the feature dataset</div>',
            unsafe_allow_html=True)

df = load_features()
if df is None or df.empty:
    st.warning("No feature dataset — run the feature engineering pipeline first.")
    st.stop()

df = df.sort_values("datetime").reset_index(drop=True)

st.sidebar.header("Controls")
hours = st.sidebar.slider("Analysis window (hours)", 24, len(df), min(24 * 14, len(df)), step=24)
view  = df.tail(hours).copy()

# ── Dataset overview ────────────────────────────────────────────────────────
st.markdown("#### Dataset overview")
total_missing = int(df.isnull().sum().sum())
o1, o2, o3, o4 = st.columns(4)
o1.metric("Total rows", f"{len(df):,}")
o2.metric("Total features", f"{df.shape[1]}")
o3.metric("Date range", f"{df['datetime'].min():%b %d} → {df['datetime'].max():%b %d}")
o4.metric("Missing values", f"{total_missing:,}")

# ── Data quality indicators ─────────────────────────────────────────────────
dup = int(df.duplicated().sum())
miss_pct = total_missing / (df.shape[0] * df.shape[1]) * 100 if df.size else 0
q1, q2, q3 = st.columns(3)
q1.metric("Missing %", f"{miss_pct:.2f}%")
q2.metric("Duplicate rows", f"{dup:,}")
q3.metric("Window rows", f"{len(view):,}")

# ── Window KPIs ──────────────────────────────────────────────────────────────
k1, k2, k3 = st.columns(3)
k1.metric("Mean PM2.5", f"{view['pm25'].mean():.1f}")
if "pm10" in view.columns:
    k2.metric("Mean PM10", f"{view['pm10'].mean():.1f}")
k3.metric("Max PM2.5", f"{view['pm25'].max():.1f}")

# ── PM2.5 trend + rolling mean ──────────────────────────────────────────────
st.markdown("#### PM2.5 trend")
v = view[["datetime", "pm25"]].copy()
v["6h rolling mean"] = v["pm25"].rolling(6).mean()
vt = v.melt("datetime", var_name="series", value_name="value")
st.altair_chart(
    alt.Chart(vt).mark_line(interpolate="monotone").encode(
        x=alt.X("datetime:T", title="Date / time"),
        y=alt.Y("value:Q", title="PM2.5 (µg/m³)"),
        color=alt.Color("series:N", title=None, scale=alt.Scale(range=["#38bdf8", "#f59e0b"])),
        tooltip=[alt.Tooltip("datetime:T", title="Time", format="%b %d %H:%M"),
                 alt.Tooltip("series:N"), alt.Tooltip("value:Q", format=".1f")],
    ).properties(height=300).interactive(bind_y=False),
    use_container_width=True)

# ── AQI category distribution ───────────────────────────────────────────────
st.markdown("#### AQI category distribution")
cat = view["pm25"].apply(lambda v: pm25_to_category(v)[0]).value_counts().reset_index()
cat.columns = ["category", "count"]
order = [b[2] for b in [(0,) * 5]]  # placeholder, replaced below
from utils import AQI_BANDS
band_order = [b[2] for b in AQI_BANDS]
band_color = {b[2]: b[4] for b in AQI_BANDS}
cat["color"] = cat["category"].map(band_color)
st.altair_chart(
    alt.Chart(cat).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("category:N", sort=band_order, title=None),
        y=alt.Y("count:Q", title="Hours"),
        color=alt.Color("color:N", scale=None, legend=None),
        tooltip=["category:N", "count:Q"],
    ).properties(height=280),
    use_container_width=True)

# ── Diurnal & weekday ────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("#### Diurnal pattern (by hour)")
    view["hour"] = view["datetime"].dt.hour
    st.altair_chart(
        alt.Chart(view).mark_boxplot(extent="min-max", color="#38bdf8").encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("pm25:Q", title="PM2.5 (µg/m³)")).properties(height=300),
        use_container_width=True)
with col_b:
    st.markdown("#### Weekday average")
    view["weekday"] = view["datetime"].dt.day_name()
    wd_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wk = view.groupby("weekday")["pm25"].mean().reindex(wd_order).reset_index()
    st.altair_chart(
        alt.Chart(wk).mark_bar(color="#5eead4", cornerRadiusEnd=3).encode(
            x=alt.X("weekday:N", sort=wd_order, title=None),
            y=alt.Y("pm25:Q", title="Mean PM2.5")).properties(height=300),
        use_container_width=True)

# ── Correlation & distribution ──────────────────────────────────────────────
col_c, col_d = st.columns(2)
with col_c:
    st.markdown("#### |Correlation| with PM2.5")
    num = view.select_dtypes("number")
    if "pm25" in num.columns:
        corr = (num.corr(numeric_only=True)["pm25"].drop("pm25").abs()
                .sort_values(ascending=False).head(12).reset_index())
        corr.columns = ["feature", "abs_corr"]
        st.altair_chart(
            alt.Chart(corr).mark_bar(color="#7B2CBF", cornerRadiusEnd=3).encode(
                x=alt.X("abs_corr:Q", title="|correlation|"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=["feature:N", alt.Tooltip("abs_corr:Q", format=".2f")],
            ).properties(height=320),
            use_container_width=True)
with col_d:
    st.markdown("#### PM2.5 distribution")
    st.altair_chart(
        alt.Chart(view).mark_bar(color="#38bdf8", opacity=0.8).encode(
            x=alt.X("pm25:Q", bin=alt.Bin(maxbins=40), title="PM2.5 (µg/m³)"),
            y=alt.Y("count()", title="Count")).properties(height=320),
        use_container_width=True)

# ── Feature family summary ──────────────────────────────────────────────────
st.markdown("#### Engineered feature families")
feat_cols = [c for c in df.columns if c not in ("datetime", "pm25")]
fam = pd.Series([feature_family(c) for c in feat_cols]).value_counts().reset_index()
fam.columns = ["family", "count"]
st.altair_chart(
    alt.Chart(fam).mark_bar(color="#C9A227", cornerRadiusEnd=3).encode(
        x=alt.X("count:Q", title="Number of features"),
        y=alt.Y("family:N", sort="-x", title=None),
        tooltip=["family:N", "count:Q"]).properties(height=220),
    use_container_width=True)
