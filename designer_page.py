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

    # Derived geometric values
    s0 = s - h0
    se = (L - (h0 * N0 + s0 * (N0 - 1))) / 2

    st.sidebar.write(f"Computed s0 = {s0:.1f} mm")
    st.sidebar.write(f"Computed se = {se:.1f} mm")

    # Basic geometrical feasibility
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

    # ============================================================
    # 🔧 ADVANCED FEASIBILITY SYSTEM (Option D)
    # ============================================================

    wu_min = 1e9
    wu_max = -1e9

    # Check all sections to determine feasible strength range
    for _, row in section_lookup.iterrows():
        H = int(row.H)
        bf = int(row.bf)
        tw = float(row.tw)
        tf = float(row.tf)

        Xtest = np.array([[H, bf, tw, tf, L, h0, s, s0, se, fy]])

        LB = fwd_p10.predict(Xtest)[0]
        UB = fwd_p90.predict(Xtest)[0]

        wu_min = min(wu_min, LB)
        wu_max = max(wu_max, UB)

    # ------------------------------------------------------------
    # 🚦 Feasibility Decision
    # ------------------------------------------------------------

    if wu_target < wu_min or wu_target > wu_max:

        st.error("❌ The requested target strength is NOT physically achievable "
                 "with this geometry and material configuration.")

        st.markdown(f"""
        ### 📉 Feasibility Summary
        - **Target wu:** {wu_target:.2f} kN/m  
        - **Minimum achievable wu:** {wu_min:.2f} kN/m  
        - **Maximum achievable wu:** {wu_max:.2f} kN/m  
        """)

        # Why infeasible?
        st.markdown("### 🛠 Why is this design infeasible?")

        reasons = []
        if wu_target < wu_min:
            reasons.append("The beam is **too strong** to reach such a low wu.")
        if wu_target > wu_max:
            reasons.append("The beam is **too weak** to reach such a high wu.")

        for r in reasons:
            st.write("- " + r)

        # Recommendations
        st.markdown("### 🔧 Engineering Recommendations")

        if wu_target < wu_min:
            st.write("""
            To **reduce strength**, consider:
            - Increase hole diameter **h0**
            - Increase spacing **s**
            - Increase number of openings **N0**
            - Reduce steel grade **fy**
            - Increase span **L**
            """)

        if wu_target > wu_max:
            st.write("""
            To **increase strength**, consider:
            - Decrease hole diameter **h0**
            - Reduce number of openings **N0**
            - Increase plate thickness (tf, tw)
            - Increase steel grade fy
            - Reduce span **L**
            """)

        st.info("The design must be adjusted before inverse design can continue.")
        return  # 🚫 STOP PIPELINE HERE

    # ============================================================
    # END OF FEASIBILITY — PIPELINE CONTINUES NORMALLY
    # ============================================================

    # Top 10 predicted sections
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
        Xtest = np.array([[H, bf, tw, tf, L, h0, s, s0, se, fy]])
        Pred_wu = fwd_p50.predict(Xtest)[0]
        LB_wu = fwd_p10.predict(Xtest)[0]
        UB_wu = fwd_p90.predict(Xtest)[0]

        if UB_wu < Pred_wu:
            quantile_cross_issue = True

        error_ratio = abs(Pred_wu - wu_target) / wu_target

        # Extract applicability flags
        app = df_full[df_full.SectionID == sec].iloc[0]
        SCI_app = app["SCI_applicable"]
        ENM_app = app["ENM_applicable"]
        AISC_app = app["AISC_applicable"]

        SCI = ENM = AISC = -1
        if SCI_app == 1:
            SCI = check_SCI(H, bf, tw, tf, h0, s0, se)
        if ENM_app == 1:
            ENM = check_ENM(H, bf, tw, tf, h0, s0)
        if AISC_app == 1:
            AISC = check_AISC(H, bf, tw, tf, h0, s)

        fm_series = df_full[df_full.SectionID == sec]["Failure_mode"]
        fm = fm_series.mode()[0] if not fm_series.mode().empty else "Unknown"

        weight = compute_weight(H, bf, tw, tf, L)

        score = multiobjective_score(
            wu_target, Pred_wu, weight, SCI, ENM, AISC, fm
        )

        results.append({
            "H (mm)": H,
            "bf (mm)": bf,
            "tw (mm)": tw,
            "tf (mm)": tf,
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

    df_res = pd.DataFrame(results).sort_values("Score").reset_index(drop=True)

    if quantile_cross_issue:
        st.info("ℹ️ In some cases, the p90 bound may be lower than p50. "
                "This can happen in quantile regression and does not affect validity.")

    exact_match = df_res.sort_values("AbsErr").iloc[0]
    st.subheader("🎯 Exact Strength-Matching Section")
    st.write(exact_match.to_frame().T)

    st.subheader("🏅 Optimal Balanced Section (Lowest Score)")
    st.write(df_res.head(1))

    st.subheader("📊 Full Ranking Table")
    st.dataframe(df_res)

    st.download_button(
        "Download Results as CSV",
        df_res.to_csv(index=False),
        file_name="inverse_design_results.csv"
    )
