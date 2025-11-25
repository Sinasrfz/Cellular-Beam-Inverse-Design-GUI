# ============================================================
# diagnostics_page.py — Model Diagnostics & Global Validity Map
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18


# ============================================================
# MAIN DIAGNOSTICS RENDER FUNCTION
# ============================================================

def render():

    st.header("📊 Diagnostics — Model Validation & Data Coverage")

    st.write("""
    This page provides diagnostics for the inverse and forward models,
    data coverage, sensitivity analysis, and a **Global Validity Map**
    indicating where the surrogate model is reliable.
    """)

    # ------------------------------------------------------------
    # Load required objects
    # ------------------------------------------------------------
    inv_model = st.session_state.get("inv_model", None)
    fwd_p50 = st.session_state.get("fwd_p50", None)
    fwd_p10 = st.session_state.get("fwd_p10", None)
    fwd_p90 = st.session_state.get("fwd_p90", None)
    df_full = st.session_state.get("df_full", None)

    if any(x is None for x in [inv_model, fwd_p50, fwd_p10, fwd_p90, df_full]):
        st.error("❌ Diagnostics cannot load: models or dataset missing.")
        return

    # Normalize column names (already normalized in app.py)
    # Required columns exist:
    # wu_FEA, L, h0, s, fy, H, bf, tw, tf, s0, se

    # ============================================================
    # SECTION 1 — INVERSE MODEL SUMMARY
    # ============================================================
    st.subheader("🧠 Inverse Model Summary")

    st.write("Model:", type(inv_model).__name__)
    st.write("Number of section classes:", len(inv_model.classes_))

    if hasattr(inv_model, "feature_importances_"):
        st.write("Feature importance:")

        fig, ax = plt.subplots(figsize=(7,4))
        features = ["wu", "L", "h0", "s", "fy"]
        ax.bar(features, inv_model.feature_importances_)
        ax.set_ylabel("Importance")
        st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # SECTION 2 — FORWARD MODEL ERROR ANALYSIS
    # ============================================================

    st.subheader("📈 Forward Surrogate Accuracy (vs FEA)")

    X = df_full[["H","bf","tw","tf","L","h0","s","s0","se","fy"]].values
    y = df_full["wu_FEA"].values

    p50 = fwd_p50.predict(X)
    p10 = fwd_p10.predict(X)
    p90 = fwd_p90.predict(X)

    err = p50 - y

    st.write(f"MAE = {np.mean(np.abs(err)):.3f} kN/m")
    st.write(f"RMSE = {np.sqrt(np.mean(err**2)):.3f} kN/m")

    # Parity plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y, p50, alpha=0.4)
    lims = [min(y), max(y)]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("FEA wu (kN/m)")
    ax.set_ylabel("P50 predicted wu")
    ax.set_title("Parity Plot")
    st.pyplot(fig)

    # Quantile coverage
    coverage = np.mean((y >= p10) & (y <= p90))
    st.write(f"P10–P90 coverage: **{coverage*100:.1f}%**")

    st.markdown("---")

    # ============================================================
    # SECTION 3 — DATASET COVERAGE (NN distance)
    # ============================================================

    st.subheader("🌐 Dataset Coverage — Normalized Nearest Neighbor Distance")

    feat_cols = ["H","bf","tw","tf","L","h0","s","s0","se","fy"]

    Xnorm = (df_full[feat_cols] - df_full[feat_cols].min()) / (
        df_full[feat_cols].max() - df_full[feat_cols].min()
    )

    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(Xnorm.values)
    dist, _ = nn.kneighbors(Xnorm.values)
    dvals = dist[:,1:].mean(axis=1)

    st.write(f"Mean NN distance: {np.mean(dvals):.3f}")
    st.write(f"95th percentile: {np.percentile(dvals,95):.3f}")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(dvals, bins=40, alpha=0.75)
    ax.set_xlabel("Mean NN distance")
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # SECTION 4 — FAILURE MODES
    # ============================================================

    st.subheader("⚠ Failure Mode Distribution")

    fig, ax = plt.subplots(figsize=(6,4))
    df_full["Failure_mode"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # SECTION 5 — GLOBAL VALIDITY MAP
    # ============================================================

    st.subheader("🗺 Global Validity Map — Training Domain Coverage")

    st.write("""
    This tool shows whether your chosen design point is **inside** the 
    dataset domain (safe for prediction) or **outside** (extrapolation).
    """)

    # User inputs
    L_u  = st.number_input("L (mm)", 2000.0, 20000.0, 8000.0)
    h0_u = st.number_input("h0 (mm)", 150.0, 800.0, 350.0)
    s_u  = st.number_input("s (mm)", 200.0, 2000.0, 600.0)
    fy_u = st.number_input("fy (MPa)", 200.0, 600.0, 355.0)
    wu_u = st.number_input("Target wu (kN/m)", 5.0, 300.0, 40.0)

    # Extract dataset ranges
    rL  = df_full["L"].min(), df_full["L"].max()
    r0  = df_full["h0"].min(), df_full["h0"].max()
    rs  = df_full["s"].min(), df_full["s"].max()
    rfy = df_full["fy"].min(), df_full["fy"].max()
    rwu = df_full["wu_FEA"].min(), df_full["wu_FEA"].max()

    # Check each parameter
    def check_range(v, r):
        return (r[0] <= v <= r[1])

    safe_L  = check_range(L_u, rL)
    safe_0  = check_range(h0_u, r0)
    safe_s  = check_range(s_u, rs)
    safe_fy = check_range(fy_u, rfy)
    safe_wu = check_range(wu_u, rwu)

    st.markdown("### Validity Status")

    def flag(ok):  
        return "🟩 **Safe**" if ok else "🟥 **Outside dataset**"

    st.write(f"L  → {flag(safe_L)}  (dataset: {rL})")
    st.write(f"h0 → {flag(safe_0)}  (dataset: {r0})")
    st.write(f"s  → {flag(safe_s)}  (dataset: {rs})")
    st.write(f"fy → {flag(safe_fy)} (dataset: {rfy})")
    st.write(f"wu → {flag(safe_wu)} (dataset: {rwu})")

    if all([safe_L, safe_0, safe_s, safe_fy, safe_wu]):
        st.success("✔ Your design lies fully inside the training domain.")
    else:
        st.error("❌ Warning: Some inputs are **outside the training dataset**. "
                 "Predictions may be unreliable.")

    st.markdown("---")

    # ============================================================
    # SECTION 6 — VALIDATION REPORT
    # ============================================================

    st.subheader("📝 Validation Summary")

    st.write(f"""
    • P50 MAE = {np.mean(np.abs(err)):.3f}  
    • P50 RMSE = {np.sqrt(np.mean(err**2)):.3f}  
    • Quantile coverage = {coverage*100:.1f}%  
    • Mean NN distance = {np.mean(dvals):.3f}  
    • 95% NN distance = {np.percentile(dvals,95):.3f}  
    """)

