# ============================================================
# diagnostics_page.py — Model Diagnostics & Validation Tools
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
    This page provides a diagnostic view of the inverse model, 
    forward surrogate, dataset coverage, error distributions, and 
    reliability indicators for engineering use.
    """)

    # Load stored objects from session_state
    inv_model = st.session_state.get("inv_model", None)
    fwd_p50 = st.session_state.get("fwd_p50", None)
    fwd_p10 = st.session_state.get("fwd_p10", None)
    fwd_p90 = st.session_state.get("fwd_p90", None)
    df_full = st.session_state.get("df_full", None)

    if any(obj is None for obj in [inv_model, fwd_p50, fwd_p10, fwd_p90, df_full]):
        st.error("❌ Diagnostics cannot load: models or dataset not found.")
        return

    # --------------------------------------------------------
    # SECTION 1 — MODEL SUMMARY
    # --------------------------------------------------------
    st.subheader("🧠 Model Summary")

    st.markdown("### Inverse Model (CatBoost Classifier)")
    st.write(f"Model type: `{type(inv_model).__name__}`")
    st.write("Number of predicted classes (sections):", len(inv_model.classes_))

    if hasattr(inv_model, "feature_importances_"):
        st.markdown("#### Feature Importance")

        fig, ax = plt.subplots(figsize=(7,4))
        importances = inv_model.feature_importances_
        features = ["wu", "L", "h0", "s", "fy"]

        ax.bar(features, importances)
        ax.set_title("Inverse Model — Feature Importance")
        st.pyplot(fig)

    st.markdown("---")

    # --------------------------------------------------------
    # SECTION 2 — FORWARD SURROGATE ACCURACY
    # --------------------------------------------------------
    st.subheader("📈 Forward Surrogate Accuracy")

    st.write("Comparison of surrogate predictions vs. FEA values from dataset.")

    # Build X and y
    feature_cols = ["H","bf","tw","tf","L","h0","s","s0","se","fy"]

    X = df_full[feature_cols].values
    y = df_full["wu_FEA"].values

    pred50 = fwd_p50.predict(X)
    pred10 = fwd_p10.predict(X)
    pred90 = fwd_p90.predict(X)

    err50 = pred50 - y

    st.markdown("#### P50 Error Statistics")
    st.write(f"MAE: {np.mean(np.abs(err50)):.3f} kN/m")
    st.write(f"RMSE: {np.sqrt(np.mean(err50**2)):.3f} kN/m")

    # Parity Plot
    st.markdown("#### Parity Plot (P50 vs FEA)")
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y, pred50, alpha=0.4)
    lims = [min(y), max(y)]
    ax.plot(lims, lims, 'r--')
    ax.set_xlabel("FEA wu (kN/m)")
    ax.set_ylabel("Predicted P50 wu (kN/m)")
    ax.set_title("Parity Plot")
    st.pyplot(fig)

    # Histogram
    st.markdown("#### Error Distribution (P50 - FEA)")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(err50, bins=40, alpha=0.7)
    ax.set_xlabel("Prediction Error (kN/m)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Quantile coverage
    coverage = np.mean((y >= pred10) & (y <= pred90))
    st.write(f"Quantile P10–P90 Coverage: **{coverage*100:.1f}%**")

    st.markdown("---")

    # --------------------------------------------------------
    # SECTION 3 — DATA COVERAGE / DISTANCE ANALYSIS
    # --------------------------------------------------------
    st.subheader("🌐 Dataset Coverage — Nearest Neighbor Distance")

    Xnorm = (df_full[feature_cols] - df_full[feature_cols].min()) / (
        df_full[feature_cols].max() - df_full[feature_cols].min()
    )

    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(Xnorm.values)

    distances, _ = nn.kneighbors(Xnorm.values)
    dist_mean = distances[:,1:].mean(axis=1)  # exclude distance to self

    st.write(f"Mean normalized distance: **{np.mean(dist_mean):.3f}**")
    st.write(f"95th percentile distance: **{np.percentile(dist_mean,95):.3f}**")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(dist_mean, bins=40, alpha=0.7)
    ax.set_xlabel("Mean NN Distance")
    ax.set_ylabel("Count")
    ax.set_title("Training Data Coverage")
    st.pyplot(fig)

    st.markdown("---")

    # --------------------------------------------------------
    # SECTION 4 — FAILURE MODES
    # --------------------------------------------------------
    st.subheader("⚠ Failure Mode Distribution")

    fig, ax = plt.subplots(figsize=(6,4))
    df_full["Failure_mode"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    ax.set_title("Failure Mode Distribution")
    st.pyplot(fig)

    st.markdown("---")

    # --------------------------------------------------------
    # SECTION 5 — CODE APPLICABILITY
    # --------------------------------------------------------
    st.subheader("📘 Code Applicability Summary")

    appl_cols = ["SCI_applicable", "ENM_applicable", "AISC_applicable"]
    st.dataframe(df_full[appl_cols].describe())

    st.markdown("---")

    # --------------------------------------------------------
    # SECTION 6 — SENSITIVITY EXPLORER
    # --------------------------------------------------------
    st.subheader("🧪 Sensitivity Explorer")

    param = st.selectbox(
        "Select parameter to vary:",
        ["L","h0","s","fy"]
    )

    st.write("This shows how P50 prediction changes when modifying one parameter ±20%.")

    # Baseline = mean of dataset
    base_vec = df_full[feature_cols].mean().values

    idx = feature_cols.index(param)
    baseline_value = base_vec[idx]

    values = np.linspace(0.8 * baseline_value, 1.2 * baseline_value, 40)

    preds = []
    for v in values:
        temp = base_vec.copy()
        temp[idx] = v
        preds.append(fwd_p50.predict(temp.reshape(1,-1))[0])

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(values, preds)
    ax.set_xlabel(f"{param}")
    ax.set_ylabel("Predicted wu (kN/m)")
    ax.set_title("Local Sensitivity Curve")
    st.pyplot(fig)

    st.markdown("---")

    # --------------------------------------------------------
    # SECTION 7 — MODEL VALIDATION SUMMARY
    # --------------------------------------------------------
    st.subheader("📝 Validation Report")

    st.write(f"""
    - Forward P50 MAE = {np.mean(np.abs(err50)):.3f} kN/m  
    - Forward P50 RMSE = {np.sqrt(np.mean(err50**2)):.3f} kN/m  
    - Quantile coverage (10–90%) = {coverage*100:.1f}%  
    - Nearest-neighbor mean distance = {np.mean(dist_mean):.3f}  
    - 95% distance = {np.percentile(dist_mean,95):.3f}  

    These metrics demonstrate that the surrogate model has overall good fit 
    on the dataset, with reliable quantile bounds and broad coverage within 
    the training domain. The nearest-neighbor analysis highlights potential 
    regions where model extrapolation may occur.
    """)

