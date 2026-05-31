"""Explainability: SHAP importance when available, else native model importance."""

import altair as alt
import streamlit as st

from utils import (
    load_contract, load_best_metrics, compute_feature_importance,
    feature_family, inject_css, render_sidebar,
)

st.set_page_config(page_title="Explanations • Islamabad PM2.5", page_icon="🧠", layout="wide")
inject_css()
render_sidebar()
st.markdown('<div class="hero-title">🧠 Why the model predicts this</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Feature drivers of the production model</div>', unsafe_allow_html=True)

contract = load_contract()
if contract is None:
    st.warning("No promoted model — run `python -m src.training.select_best_model`.")
    st.stop()

imp, source = compute_feature_importance(prefer_shap=True)
if imp is None or imp.empty:
    st.error("Feature importance is unavailable for this model type and no SHAP artifact was found.")
    st.stop()

imp = imp.copy()
total = imp["importance"].sum()
imp["pct"]    = imp["importance"] / total if total else 0.0
imp["family"] = imp["feature"].apply(feature_family)

# ── Model information card ──────────────────────────────────────────────────
metrics = load_best_metrics() or {}
src_label = "SHAP values" if source == "SHAP" else "model importances"
st.markdown(
    f'<div class="card"><h4>Production model · explained via {src_label}</h4>'
    f'<div class="big">{str(contract.get("model_name","?")).upper()}</div>'
    f'<div class="muted">selected by {contract.get("selected_by","?")}</div></div>',
    unsafe_allow_html=True)

i1, i2, i3, i4 = st.columns(4)
i1.metric("Test RMSE", metrics.get("rmse", "—"))
i2.metric("Test MAE",  metrics.get("mae", "—"))
i3.metric("Test R²",   metrics.get("r2", "—"))
i4.metric("Features",  len(imp))
st.write("")

# ── Controls + driver KPIs ──────────────────────────────────────────────────
k = st.slider("Show top K features", 3, len(imp), min(12, len(imp)))
topk = imp.head(k)
d1, d2, d3 = st.columns(3)
d1.metric("Top driver", topk.iloc[0]["feature"])
d2.metric(f"Top-{k} coverage", f'{topk["pct"].sum()*100:.1f}%')
d3.metric("Importance source", "SHAP" if source == "SHAP" else "Native")

# ── Importance bar (grouped by family) ──────────────────────────────────────
st.markdown("#### Feature importance")
st.altair_chart(
    alt.Chart(topk).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("importance:Q", title="Importance"),
        y=alt.Y("feature:N", sort="-x", title=None),
        color=alt.Color("family:N", title="Feature family",
                        scale=alt.Scale(scheme="tealblues")),
        tooltip=[alt.Tooltip("feature:N", title="Feature"),
                 alt.Tooltip("family:N", title="Family"),
                 alt.Tooltip("importance:Q", title="Importance", format=".4f"),
                 alt.Tooltip("pct:Q", title="Share", format=".1%")],
    ).properties(height=28 * len(topk) + 50),
    use_container_width=True)

# ── Cumulative importance ───────────────────────────────────────────────────
st.markdown("#### Cumulative importance")
cum = imp.copy()
cum["rank"]       = range(1, len(cum) + 1)
cum["cumulative"] = cum["pct"].cumsum()
st.altair_chart(
    alt.Chart(cum).mark_line(point=True, color="#5eead4").encode(
        x=alt.X("rank:Q", title="Number of top features"),
        y=alt.Y("cumulative:Q", title="Cumulative importance", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("rank:Q", title="Top-N"),
                 alt.Tooltip("feature:N", title="Feature added"),
                 alt.Tooltip("cumulative:Q", title="Cumulative", format=".1%")],
    ).properties(height=300).interactive(bind_x=False),
    use_container_width=True)

# ── Family summary + export ─────────────────────────────────────────────────
st.markdown("#### Importance by feature family")
fam = imp.groupby("family", as_index=False)["pct"].sum().sort_values("pct", ascending=False)
st.altair_chart(
    alt.Chart(fam).mark_bar(color="#38bdf8", cornerRadiusEnd=3).encode(
        x=alt.X("pct:Q", title="Share of importance", axis=alt.Axis(format="%")),
        y=alt.Y("family:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("family:N", title="Family"),
                 alt.Tooltip("pct:Q", title="Share", format=".1%")],
    ).properties(height=200),
    use_container_width=True)

with st.expander("Full importance table"):
    st.dataframe(imp, hide_index=True, use_container_width=True,
                 column_config={"importance": st.column_config.NumberColumn(format="%.5f"),
                                "pct": st.column_config.NumberColumn("share", format="%.2f%%")})
    st.download_button("⬇️ Download CSV", imp.to_csv(index=False).encode(),
                       "feature_importance.csv", "text/csv")

if source != "SHAP":
    st.caption("Showing native model importance. To prefer SHAP, save a SHAP artifact to "
               "`reports/shap/shap_importance.{parquet,csv,json}` with columns "
               "`feature`, `mean_abs_shap` — this page will use it automatically.")
