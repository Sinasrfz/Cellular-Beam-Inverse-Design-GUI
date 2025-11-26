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

    # Geometric feasibility
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

    # Compute full forward space for feasibility
    all_wu_p50 = []
    for _, row in section_lookup.iterrows():
        H = row.H; bf = row.bf; tw = row.tw; tf = row.tf
        Xtest = np.array([[H, bf, tw, tf, L, h0, s, s0, se, fy]])
        val = fwd_p50.predict(Xtest)[0]
        all_wu_p50.append(val)

    wu_min = min(all_wu_p50)
    wu_max = max(all_wu_p50)

    # Check physical feasibility
    if wu_target < wu_min or wu_target > wu_max:

        st.error("❌ The requested target strength is NOT physically achievable with this geometry and material configuration.")

        st.markdown("### 📉 Feasibility Summary")
        st.write(f"**Target wu:** {wu_target:.2f} kN/m")
        st.write(f"**Minimum achievable wu:** {wu_min:.2f} kN/m")
        st.write(f"**Maximum achievable wu:** {wu_max:.2f} kN/m")

        # Engineering guidance for outside range
        if wu_target > wu_max:
            st.markdown("### 🔧 Why is this design infeasible?")
            st.write("The beam is **too weak** to reach such a high wu.")

            st.markdown("### 🔧 Engineering Recommendations")
            st.write("""
            To **increase** strength, consider:
            - Decrease hole diameter **h0**
            - Reduce number of openings **N0**
            - Increase plate thickness (tf, tw)
            - Increase steel grade **fy**
            - Reduce span **L**
            """)
        else:
            st.markdown("### 🔧 Why is this design infeasible?")
            st.write("The beam is **too strong** for such a small target wu.")

            st.markdown("### 🔧 Engineering Recommendations")
            st.write("""
            To **reduce** strength:
            - Increase hole diameter **h0**
            - Increase spacing **s**
            - Increase number of openings **N0**
            - Reduce steel grade **fy**
            - Increase span **L**
            """)
        return

    # PHASE 1: Inverse model prediction
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

        Xtest = np.array([[H, bf, tw, tf, L, h0, s, s0, se, fy]])
        Pred_wu = fwd_p50.predict(Xtest)[0]
        LB_wu = fwd_p10.predict(Xtest)[0]
        UB_wu = fwd_p90.predict(Xtest)[0]

        if UB_wu < Pred_wu:
            quantile_cross_issue = True

        error_ratio = abs(Pred_wu - wu_target) / wu_target

        # Applicability from dataset
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

    # Phase 1.5 — Achievability inside feasible range but no good match
    best_wu = df_res.iloc[0]["Predicted_wu"]
    err = abs(best_wu - wu_target)

    tolerance = max(0.15 * wu_target, 5)

    if err > tolerance:
        st.error("❗ No available section can reach your target strength for the selected geometry.")

        st.markdown("### 📉 Achievability Check")
        st.write(f"**Target wu:** {wu_target:.2f} kN/m")
        st.write(f"**Closest achievable wu:** {best_wu:.2f} kN/m")
        st.write(f"**Difference:** {err:.2f} kN/m")

        if best_wu < wu_target:
            st.markdown("### 🔧 Engineering Guidance — Increase Strength")
            st.write("""
            To **increase** strength:
            - Decrease hole diameter **h0**
            - Reduce number of openings **N0**
            - Increase plate thickness (tf, tw)
            - Increase steel grade **fy**
            - Reduce span **L**
            """)
        else:
            st.markdown("### 🔧 Engineering Guidance — Reduce Strength")
            st.write("""
            To **reduce** strength:
            - Increase hole diameter **h0**
            - Increase spacing **s**
            - Increase number of openings **N0**
            - Reduce steel grade **fy**
            - Increase span **L**
            """)
        return

    # One-time quantile warning
    if quantile_cross_issue:
        st.info("ℹ️ In some cases, the upper bound (p90) is lower than the median. "
                "This is normal in quantile regression.")

    # Exact strength-matching section
    exact_match = df_res.sort_values("AbsErr").iloc[0]
    st.subheader("🎯 Exact Strength-Matching Section")
    st.write(exact_match.to_frame().T)

    # Optimal section
    st.subheader("🏅 Optimal Balanced Section (Lowest Score)")
    st.write(df_res.head(1))

    # Full ranking table
    st.subheader("📊 Full Ranking Table")
    st.dataframe(df_res)

    st.download_button(
        "Download Results as CSV",
        df_res.to_csv(index=False),
        file_name="inverse_design_results.csv"
    )
