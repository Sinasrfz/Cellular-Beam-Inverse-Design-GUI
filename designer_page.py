# ============================================================
# designer_page.py — 3-Stage Inverse Design (Deterministic Code Checks)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stage1_functions import (
    check_SCI,
    check_ENM,
    check_AISC,
    compute_weight,
    multiobjective_score,
    applicability_to_emoji,   # ← N/A / Pass / Fail → emoji
)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20


# ------------------------------------------------------------
# MAIN RENDER FUNCTION
# ------------------------------------------------------------

def render(inv_model, fwd_p50, fwd_p10, fwd_p90, section_lookup, df_full):

    st.header("🏗 Designer Tool — 3-Stage Inverse Design")

    st.sidebar.subheader("🧮 Design Inputs")
    wu_target = st.sidebar.number_input("Target wu (kN/m)", 5.0, 300.0, 30.0)
    L = st.sidebar.number_input("Beam span L (mm)", 5000.0, 30000.0, 12000.0)
    h0 = st.sidebar.number_input("Opening diameter h0 (mm)", 200.0, 800.0, 400.0)
    s = st.sidebar.number_input("Centre spacing s (mm)", 300.0, 2000.0, 600.0)
    fy = st.sidebar.number_input("Steel fy (MPa)", 200.0, 600.0, 355.0)
    N0 = st.sidebar.number_input("Number of openings N0", 1, 50, 10)

    # Derived geometric values
    s0 = s - h0
    se = (L - (h0 * N0 + s0 * (N0 - 1))) / 2

    st.sidebar.write(f"Computed s0 = {s0:.1f} mm")
    st.sidebar.write(f"Computed se = {se:.1f} mm")

    if se < 0:
        st.error("❌ Selected N0 is NOT feasible for this span. Reduce N0.")
        return

    if st.button("Run Inverse Design", type="primary"):
        run_inverse(
            wu_target, L, h0, s, s0, se, fy,
            inv_model, fwd_p50, fwd_p10, fwd_p90,
            section_lookup, df_full
        )


# ------------------------------------------------------------
# CORE PIPELINE
# ------------------------------------------------------------

def run_inverse(wu_target, L, h0, s, s0, se, fy,
                inv_model, fwd_p50, fwd_p10, fwd_p90,
                section_lookup, df_full):

    st.subheader("🔍 Phase 1 — Inverse Model Prediction")

    # Feasible strength limits
    all_wu_p50 = []
    for _, row in section_lookup.iterrows():
        H = row.H; bf = row.bf; tw = row.tw; tf = row.tf
        Xtest = np.array([[H, bf, tw, tf, L, h0, s, s0, se, fy]])
        val = fwd_p50.predict(Xtest)[0]
        all_wu_p50.append(val)

    wu_min = min(all_wu_p50)
    wu_max = max(all_wu_p50)

    if wu_target < wu_min or wu_target > wu_max:
        st.error("❌ The requested target strength is NOT physically achievable.")
        st.write(f"Target wu = {wu_target:.2f}")
        st.write(f"Achievable range = {wu_min:.2f} to {wu_max:.2f}")
        return

    # PHASE 1 — inverse selection
    proba = inv_model.predict_proba([[wu_target, L, h0, s, fy]])[0]
    top_sections = proba.argsort()[-10:][::-1]

    results = []
    quantile_cross_issue = False

    for sec in top_sections:

        # --- Geometry ---
        row = section_lookup[section_lookup.SectionID == sec].iloc[0]
        H = int(row.H)
        bf = int(row.bf)
        tw = float(row.tw)
        tf = float(row.tf)

        # Forward surrogate
        Xtest = np.array([[H, bf, tw, tf, L, h0, s, s0, se, fy]])
        Pred_wu = fwd_p50.predict(Xtest)[0]
        LB_wu   = fwd_p10.predict(Xtest)[0]
        UB_wu   = fwd_p90.predict(Xtest)[0]

        if UB_wu < Pred_wu:
            quantile_cross_issue = True

        # -----------------------------
        # DETERMINISTIC APPLICABILITY
        # -----------------------------
        area_row = df_full[df_full.SectionID == sec].iloc[0]
        wu_fea = float(area_row["wu_FEA"])

        # SCI
        SCI_app = int(area_row["SCI_applicable"])
        wSCI    = float(area_row["wSCI"]) if SCI_app == 1 else None
        if SCI_app == 0:
            SCI_val = -1              # N/A
        else:
            SCI_val = 1 if wu_fea <= wSCI else 0

        # ENM
        ENM_app = int(area_row["ENM_applicable"])
        wENM    = float(area_row["wENM"]) if ENM_app == 1 else None
        if ENM_app == 0:
            ENM_val = -1
        else:
            ENM_val = 1 if wu_fea <= wENM else 0

        # ENA
        ENA_app = int(area_row["ENA_applicable"])
        wENA    = float(area_row["wENA"]) if ENA_app == 1 else None
        if ENA_app == 0:
            ENA_val = -1
        else:
            ENA_val = 1 if wu_fea <= wENA else 0

        # AISC
        AISC_app = int(area_row["AISC_applicable"])
        wAISC    = float(area_row["wAISC"]) if AISC_app == 1 else None
        if AISC_app == 0:
            AISC_val = -1
        else:
            AISC_val = 1 if wu_fea <= wAISC else 0

        # Failure mode (database)
        FM_val = area_row["Failure_mode"]

        # -----------------------------
        # SCORING (deterministic)
        # -----------------------------
        weight = compute_weight(H, bf, tw, tf, L)

        score = multiobjective_score(
            wu_target, Pred_wu, weight,
            SCI_val, ENM_val, AISC_val, FM_val
        )

        results.append({
            "SectionID": sec,
            "H (mm)": H,
            "bf (mm)": bf,
            "tw (mm)": tw,
            "tf (mm)": tf,

            "Predicted_wu": Pred_wu,
            "LowerBound_wu": LB_wu,
            "UpperBound_wu": UB_wu,

            "SCI":  applicability_to_emoji(SCI_val),
            "ENM":  applicability_to_emoji(ENM_val),
            "ENA":  applicability_to_emoji(ENA_val),
            "AISC": applicability_to_emoji(AISC_val),

            "Failure_mode": FM_val,
            "Weight (kg)": weight,
            "Score": score,
            "AbsErr": abs(Pred_wu - wu_target),
        })

    df_res = pd.DataFrame(results).sort_values("Score").reset_index(drop=True)

    # Match tolerance
    best_wu = df_res.iloc[0]["Predicted_wu"]
    if abs(best_wu - wu_target) > max(0.15 * wu_target, 5):
        st.error("❗ No section can reach the target strength within tolerance.")
        return

    if quantile_cross_issue:
        st.info("ℹ Some quantiles crossed. Surrogate uncertainty is high.")

    st.subheader("🎯 Exact Strength-Matching Section")
    st.write(df_res.sort_values("AbsErr").head(1))

    st.subheader("🏅 Optimal Balanced Section (Lowest Score)")
    st.write(df_res.head(1))

    st.subheader("📊 Full Ranking Table")
    st.dataframe(df_res)

    st.download_button(
        "Download Results as CSV",
        df_res.to_csv(index=False),
        file_name="inverse_design_results.csv"
    )
