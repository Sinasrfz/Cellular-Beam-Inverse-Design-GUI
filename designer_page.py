# ============================================================
# designer_page.py — 3-Stage Inverse Design (Persistent Version)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

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
# INITIALIZE SESSION STATE
# ------------------------------------------------------------

def init_session():
    if "designer_inputs" not in st.session_state:
        st.session_state.designer_inputs = {}

    if "designer_results" not in st.session_state:
        st.session_state.designer_results = None

    if "designer_best" not in st.session_state:
        st.session_state.designer_best = None

    if "designer_optimal" not in st.session_state:
        st.session_state.designer_optimal = None


# ------------------------------------------------------------
# MAIN RENDER FUNCTION
# ------------------------------------------------------------

def render(inv_model, fwd_p50, fwd_p10, fwd_p90, section_lookup, df_full):

    init_session()

    st.header("🏗 Designer Tool — 3-Stage Inverse Design")

    # -------------------------
    # CLEAR BUTTON
    # -------------------------
    if st.button("🧹 Clear All"):
        for key in ["designer_inputs", "designer_results",
                    "designer_best", "designer_optimal"]:
            st.session_state.pop(key, None)
        st.experimental_rerun()

    st.sidebar.subheader("🧮 Design Inputs")

    # Load stored values or defaults
    inputs = st.session_state.designer_inputs
    wu_target = st.sidebar.number_input(
        "Target wu (kN/m)", 5.0, 300.0, inputs.get("wu_target", 30.0)
    )
    L = st.sidebar.number_input(
        "Beam span L (mm)", 5000.0, 30000.0, inputs.get("L", 12000.0)
    )
    h0 = st.sidebar.number_input(
        "Opening diameter h0 (mm)", 200.0, 800.0, inputs.get("h0", 400.0)
    )
    s = st.sidebar.number_input(
        "Centre spacing s (mm)", 300.0, 2000.0, inputs.get("s", 600.0)
    )
    fy = st.sidebar.number_input(
        "Steel fy (MPa)", 200.0, 600.0, inputs.get("fy", 355.0)
    )
    N0 = st.sidebar.number_input(
        "Number of openings N0", 1, 50, inputs.get("N0", 10)
    )

    # Save inputs persistently
    st.session_state.designer_inputs.update({
        "wu_target": wu_target,
        "L": L, "h0": h0,
        "s": s, "fy": fy, "N0": N0
    })

    # Derived geometric values
    s0 = s - h0
    se = (L - (h0 * N0 + s0 * (N0 - 1))) / 2

    st.sidebar.write(f"Computed s0 = {s0:.1f} mm")
    st.sidebar.write(f"Computed se = {se:.1f} mm")

    # Feasibility check
    if se < 0:
        st.error("❌ Selected N0 is NOT feasible for this span. Reduce N0.")
        return

    # -------------------------
    # RUN BUTTON
    # -------------------------
    if st.button("Run Inverse Design", type="primary"):
        run_inverse(
            wu_target, L, h0, s, s0, se, fy,
            inv_model, fwd_p50, fwd_p10, fwd_p90,
            section_lookup, df_full
        )

    # -------------------------
    # DISPLAY SAVED RESULTS (persistent)
    # -------------------------
    if st.session_state.designer_results is not None:
        st.subheader("🏅 Optimal Balanced Section")
        st.write(st.session_state.designer_optimal.to_frame().T)

        st.subheader("🎯 Exact Strength-Matching Section")
        st.write(st.session_state.designer_best.to_frame().T)

        st.subheader("📊 Full Ranking Table")
        st.dataframe(st.session_state.designer_results)

        # Multi-format export
        df_res = st.session_state.designer_results

        st.download_button(
            "Download CSV",
            df_res.to_csv(index=False),
            file_name="inverse_design_results.csv"
        )

        # Excel export
        buffer = io.BytesIO()
        df_res.to_excel(buffer, index=False)
        st.download_button(
            "Download Excel (.xlsx)",
            buffer.getvalue(),
            file_name="inverse_design_results.xlsx"
        )

        # JSON export
        st.download_button(
            "Download JSON",
            df_res.to_json(orient="records", indent=2),
            file_name="inverse_design_results.json"
        )

        # TXT export
        st.download_button(
            "Download TXT",
            df_res.to_string(),
            file_name="inverse_design_results.txt"
        )


# ------------------------------------------------------------
# CORE PIPELINE
# ------------------------------------------------------------

def run_inverse(wu_target, L, h0, s, s0, se, fy,
                inv_model, fwd_p50, fwd_p10, fwd_p90,
                section_lookup, df_full):

    st.subheader("🔍 Phase 1 — Inverse Model Prediction")

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

        # Applicability
        app = df_full[df_full.SectionID == sec].iloc[0]
        SCI_app = app["SCI_applicable"]
        ENM_app = app["ENM_applicable"]
        AISC_app = app["AISC_applicable"]

        SCI = ENM = AISC = -1  # N/A default
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
        st.info("ℹ️ In some cases, the upper bound is lower than the median. Normal for quantile regression.")

    # Best by strength match
    exact_match = df_res.sort_values("AbsErr").iloc[0]

    # Best by score
    optimal = df_res.iloc[0]

    # -------------------------
    # Save results persistently
    # -------------------------
    st.session_state.designer_results = df_res
    st.session_state.designer_best = exact_match
    st.session_state.designer_optimal = optimal

    # Display them immediately
    st.subheader("🎯 Exact Strength-Matching Section")
    st.write(exact_match.to_frame().T)

    st.subheader("🏅 Optimal Balanced Section (Lowest Score)")
    st.write(optimal.to_frame().T)

    st.subheader("📊 Full Ranking Table")
    st.dataframe(df_res)
