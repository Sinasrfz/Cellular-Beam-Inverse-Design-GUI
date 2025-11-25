# ============================================================
# designer_page.py — 3-Stage Inverse Design (Updated Version)
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
    code_to_emoji
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

    # Derived values
    s0 = s - h0
    se = (L - (h0 * N0 + s0 * (N0 - 1))) / 2

    # Show computed values
    st.sidebar.write(f"Computed s0 = {s0:.1f} mm")
    st.sidebar.write(f"Computed se = {se:.1f} mm")

    # Check feasibility
    if se < 0:
        st.error("❌ The selected number of openings N0 is NOT feasible for this span. "
                 "Please reduce N0.")
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

    # Top 10 predictions
    proba = inv_model.predict_proba([[wu_target, L, h0, s, fy]])[0]
    top_sections = proba.argsort()[-10:][::-1]

    results = []
    quantile_cross_issue = False

    for sec in top_sections:

        row = section_lookup[section_lookup.SectionID == sec].iloc[0]
        H = int(row.H)
        bf = int(row.bf)
        tw = float(row.tw)
        tf = float(row.tf)

        # Forward predictions
        X = np.array([[H, bf, tw, tf, L, h0, s, s0, se, fy]])
        wu50 = fwd_p50.predict(X)[0]
        wu10 = fwd_p10.predict(X)[0]
        wu90 = fwd_p90.predict(X)[0]

        # Check for quantile crossing
        if wu90 < wu50:
            quantile_cross_issue = True

        error_ratio = abs(wu50 - wu_target) / wu_target

        # ----------------------------------------------------
        # APPLICABILITY CHECK
        # ----------------------------------------------------
        app_row = df_full[df_full.SectionID == sec].iloc[0]

        SCI_app = app_row["SCI_applicable"]
        ENM_app = app_row["ENM_applicable"]
        AISC_app = app_row["AISC_applicable"]

        # Default: N/A
        SCI = ENM = AISC = -1

        if SCI_app == 1:
            SCI = check_SCI(H, bf, tw, tf, h0, s0, se)

        if ENM_app == 1:
            ENM = check_ENM(H, bf, tw, tf, h0, s0)

        if AISC_app == 1:
            AISC = check_AISC(H, bf, tw, tf, h0, s)

        # Failure mode
        fm_series = df_full[df_full.SectionID == sec]["Failure_mode"]
        fm = fm_series.mode()[0] if not fm_series.mode().empty else "Unknown"

        # Weight
        weight = compute_weight(H, bf, tw, tf, L)

        # Score
        score = multiobjective_score(
            wu_target, wu50, weight, SCI, ENM, AISC, fm
        )

        # Add row
        results.append({
            "SectionID": sec,
            "H": H, "bf": bf, "tw": tw, "tf": tf,
            "Wu_50": wu50, "Wu_10": wu10, "Wu_90": wu90,
            "ErrorRatio": error_ratio,
            "SCI": code_to_emoji(SCI),
            "ENM": code_to_emoji(ENM),
            "AISC": code_to_emoji(AISC),
            "FailureMode": fm,
            "Weight_kg": weight,
            "Score": score,
            "AbsErr": abs(wu50 - wu_target)
        })

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("Score").reset_index(drop=True)

    # ------------------------------------------------------------
    # Quantile crossing message (only once)
    # ------------------------------------------------------------
    if quantile_cross_issue:
        st.info("ℹ️ Note: In some cases, p90 < p50. "
                "This is normal in quantile regression and does not affect design validity.")

    # ------------------------------------------------------------
    # Exact match block
    # ------------------------------------------------------------
    exact_match = df_res.sort_values("AbsErr").iloc[0]

    st.subheader("🎯 Exact Strength-Matching Section")
    st.write(exact_match.to_frame().T)

    # ------------------------------------------------------------
    # Optimal section block
    # ------------------------------------------------------------
    st.subheader("🏅 Optimal Balanced Section (Lowest Score)")
    st.write(df_res.head(1))

    # ------------------------------------------------------------
    # Full ranking table
    # ------------------------------------------------------------
    st.subheader("📊 Full Ranking Table")
    st.dataframe(df_res)

    st.download_button(
        "Download Results as CSV",
        df_res.to_csv(index=False),
        file_name="inverse_design_results.csv"
    )
