# ============================================================
# designer_page.py — 3-Stage Inverse Design (KNN Version)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stage1_functions import (
    compute_weight,
    multiobjective_score,
    code_to_emoji
)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20


# ------------------------------------------------------------
# MAIN RENDER FUNCTION (MUST MATCH app.py)
# ------------------------------------------------------------
def render(knn_model, scaler_knn, X_knn_raw, knn_section_ids,
           fwd_p50, fwd_p10, fwd_p90,
           section_lookup, df_full):

    st.header("🏗 Designer Tool — 3-Stage Inverse Design (KNN Version)")

    st.sidebar.subheader("🧮 Design Inputs")
    wu_target = st.sidebar.number_input("Target wu (kN/m)", 5.0, 300.0, 30.0)
    L = st.sidebar.number_input("Beam span L (mm)", 5000.0, 30000.0, 12000.0)
    h0 = st.sidebar.number_input("Opening diameter h0 (mm)", 200.0, 800.0, 400.0)
    s = st.sidebar.number_input("Centre spacing s (mm)", 300.0, 2000.0, 600.0)
    fy = st.sidebar.number_input("Steel fy (MPa)", 200.0, 600.0, 355.0)
    N0 = st.sidebar.number_input("Number of openings N0", 1, 50, 10)

    s0 = s - h0
    se = (L - (h0*N0 + s0*(N0-1))) / 2

    st.sidebar.write(f"Computed s0 = {s0:.1f} mm")
    st.sidebar.write(f"Computed se = {se:.1f} mm")

    if se < 0:
        st.error("❌ Selected N0 is NOT feasible for this span. Reduce N0.")
        return

    if st.button("Run Inverse Design", type="primary"):
        run_inverse(
            wu_target, L, h0, s, s0, se, fy,
            knn_model, scaler_knn, X_knn_raw, knn_section_ids,
            fwd_p50, fwd_p10, fwd_p90,
            section_lookup, df_full
        )


# ------------------------------------------------------------
# CORE PIPELINE
# ------------------------------------------------------------
def run_inverse(wu_target, L, h0, s, s0, se, fy,
                knn_model, scaler_knn, X_knn_raw, knn_section_ids,
                fwd_p50, fwd_p10, fwd_p90,
                section_lookup, df_full):

    st.subheader("🔍 Phase 1 — Inverse Retrieval Using KNN")

    # Build query
    q = np.array([[

        wu_target,
        L,
        h0,
        s,
        fy,

        L / (L / 400.0),     
        s / h0,
        h0 / L,
        1.5
    ]])

    q_scaled = scaler_knn.transform(q)

    dist, idx = knn_model.kneighbors(q_scaled, n_neighbors=10)
    top_sections = knn_section_ids[idx[0]]

    st.write("Top retrieved SectionIDs:", list(top_sections))

    # ------------------------------------------------------------
    # PHASE 2 — Forward Model feasibility check
    # ------------------------------------------------------------

    all_wu_p50 = []
    for _, row in section_lookup.iterrows():
        H = row.H; bf = row.bf; tw = row.tw; tf = row.tf
        Xtest = np.array([[H, bf, tw, tf, L, h0, s, s0, se, fy]])
        val = fwd_p50.predict(Xtest)[0]
        all_wu_p50.append(val)

    wu_min = min(all_wu_p50)
    wu_max = max(all_wu_p50)

    if wu_target < wu_min or wu_target > wu_max:
        st.error("❌ Target strength is NOT physically achievable with this geometry.")
        st.write(f"Min wu: {wu_min:.2f}  |  Max wu: {wu_max:.2f}")
        return

    st.subheader("🧠 Phase 2 — Forward Predictions for Retrieved Sections")

    # ------------------------------------------------------------
    # PHASE 3 — Evaluate Candidates
    # ------------------------------------------------------------

    results = []
    quantile_cross_issue = False

    for sec in top_sections:

        row = section_lookup[section_lookup.SectionID == sec].iloc[0]

        H = int(row.H); bf = int(row.bf)
        tw = float(row.tw); tf = float(row.tf)

        Xtest = np.array([[H, bf, tw, tf, L, h0, s, s0, se, fy]])

        Pred_wu = fwd_p50.predict(Xtest)[0]
        LB_wu   = fwd_p10.predict(Xtest)[0]
        UB_wu   = fwd_p90.predict(Xtest)[0]

        if UB_wu < Pred_wu:
            quantile_cross_issue = True

        error_ratio = abs(Pred_wu - wu_target) / wu_target

        candidates = df_full[df_full.SectionID == sec].copy()
        if len(candidates) == 0:
            continue

        candidates["geom_dist"] = (
                (candidates["L"] - L)**2 +
                (candidates["h0"] - h0)**2 +
                (candidates["s"] - s)**2 +
                (candidates["s0"] - s0)**2 +
                (candidates["se"] - se)**2 +
                (candidates["fy"] - fy)**2
        )**0.5

        app = candidates.sort_values("geom_dist").iloc[0]

        def to_binary(x):
            if isinstance(x, str):
                v = x.strip().lower()
                if v in ("yes", "y", "1", "true"): return 1
                if v in ("no",  "n", "0", "false"): return 0
            try: return int(x)
            except: return -1

        SCI_app  = to_binary(app["SCI_applicable"])
        ENM_app  = to_binary(app["ENM_applicable"])
        AISC_app = to_binary(app["AISC_applicable"])

        SCI = ENM = AISC = -1
        wu_demand = Pred_wu

        if SCI_app == 1:
            SCI = 1 if wu_demand <= app["wSCI"] else 0
        if ENM_app == 1:
            ENM = 1 if wu_demand <= app["wENM"] else 0
        if AISC_app == 1:
            AISC = 1 if wu_demand <= app["wAISC"] else 0

        fm = app["Failure_mode"]

        weight = compute_weight(H, bf, tw, tf, L)

        score = multiobjective_score(
            wu_target, Pred_wu, weight, SCI, ENM, AISC, fm
        )

        results.append({
            "H (mm)": H, "bf (mm)": bf, "tw (mm)": tw, "tf (mm)": tf,
            "Predicted_wu": Pred_wu,
            "LowerBound_wu": LB_wu,
            "UpperBound_wu": UB_wu,
            "ErrorRatio": error_ratio,
            "SCI": code_to_emoji(SCI),
            "ENM": code_to_emoji(ENM),
            "AISC": code_to_emoji(AISC),
            "FailureMode": fm,
            "Weight (kg)": weight,
            "Score": score,
            "AbsErr": abs(Pred_wu - wu_target)
        })

    if len(results) == 0:
        st.error("❌ No matching sections found.")
        return

    df_res = pd.DataFrame(results).sort_values("Score").reset_index(drop=True)

    if quantile_cross_issue:
        st.info("ℹ Quantile inconsistency detected (normal for quantile models).")

    best_wu = df_res.iloc[0]["Predicted_wu"]
    err = abs(best_wu - wu_target)
    tolerance = max(0.15 * wu_target, 5)

    if err > tolerance:
        st.error("❌ No feasible section meets the target strength.")
        st.write(df_res.head(1))
        return

    st.subheader("🎯 Exact Strength-Matching Section")
    st.write(df_res.sort_values("AbsErr").iloc[0:1])

    st.subheader("🏅 Optimal Balanced Section")
    st.write(df_res.head(1))

    st.subheader("📊 Full Ranking Table")
    st.dataframe(df_res)

    st.download_button(
        "Download Results as CSV",
        df_res.to_csv(index=False),
        file_name="inverse_design_results.csv"
    )
