# ============================================================
# designer_page.py — 3-Stage Inverse Design (Stable Version)
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

    defaults = [
        "designer_inputs",
        "designer_results",
        "designer_best",
        "designer_optimal",
        "clear_trigger",
        "run_trigger",
    ]

    for d in defaults:
        if d not in st.session_state:
            st.session_state[d] = None


# ------------------------------------------------------------
# MAIN RENDER FUNCTION
# ------------------------------------------------------------
def render(inv_model, fwd_p50, fwd_p10, fwd_p90, section_lookup, df_full):

    init_session()

    # =====================================================
    # 1 — HANDLE TRIGGERS FIRST (NO RERUN INSIDE RENDER)
    # =====================================================

    # ---------- CLEAR ACTION ----------
    if st.session_state.clear_trigger:
        st.session_state.designer_inputs = {}
        st.session_state.designer_results = None
        st.session_state.designer_best = None
        st.session_state.designer_optimal = None
        st.session_state.clear_trigger = None
        return  # exit render safely

    # ---------- RUN ACTION ----------
    if st.session_state.run_trigger:
        data = st.session_state.run_trigger
        st.session_state.run_trigger = None  # consume trigger

        run_inverse(
            *data,
            inv_model, fwd_p50, fwd_p10, fwd_p90,
            section_lookup, df_full
        )
        return  # exit render safely

    # --------------------------------------------------------
    # 2 — NORMAL PAGE RENDERING
    # --------------------------------------------------------
    st.header("🏗 Designer Tool — 3-Stage Inverse Design")

    # ---------- CLEAR BUTTON ----------
    if st.button("🧹 Clear All"):
        st.session_state.clear_trigger = True
        return

    st.sidebar.subheader("🧮 Design Inputs")

    # Load previous values if exist
    inputs = st.session_state.designer_inputs or {}

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

    # Save
    st.session_state.designer_inputs = {
        "wu_target": wu_target,
        "L": L, "h0": h0,
        "s": s, "fy": fy, "N0": N0
    }

    # Derived
    s0 = s - h0
    se = (L - (h0 * N0 + s0 * (N0 - 1))) / 2

    st.sidebar.write(f"Computed s0 = {s0:.1f} mm")
    st.sidebar.write(f"Computed se = {se:.1f} mm")

    if se < 0:
        st.error("❌ Selected N0 is NOT feasible for this span. Reduce N0.")
        return

    # ---------- RUN BUTTON ----------
    if st.button("Run Inverse Design", type="primary"):

        # Store trigger with parameters
        st.session_state.run_trigger = [
            wu_target, L, h0, s, s0, se, fy
        ]
        return

    # ============================================================
    # SHOW RESULTS (only after run)
    # ============================================================
    if st.session_state.designer_results is not None:

        st.subheader("🎯 Exact Strength-Matching Section")
        st.write(st.session_state.designer_best.to_frame().T)

        st.subheader("🏅 Optimal Balanced Section")
        st.write(st.session_state.designer_optimal.to_frame().T)

        st.subheader("📊 Full Ranking Table")
        st.dataframe(st.session_state.designer_results)

        df_res = st.session_state.designer_results

        st.download_button("Download CSV",
            df_res.to_csv(index=False),
            file_name="inverse_design_results.csv"
        )

        buf = io.BytesIO()
        df_res.to_excel(buf, index=False)
        st.download_button(
            "Download Excel",
            buf.getvalue(),
            file_name="inverse_design_results.xlsx"
        )

        st.download_button(
            "Download JSON",
            df_res.to_json(orient="records", indent=2),
            file_name="inverse_design_results.json"
        )

        st.download_button(
            "Download TXT",
            df_res.to_string(),
            file_name="inverse_design_results.txt"
        )


# ------------------------------------------------------------
# 3 — CORE PIPELINE (NO RERUN, NO UI OUTPUT)
# ------------------------------------------------------------
def run_inverse(wu_target, L, h0, s, s0, se, fy,
                inv_model, fwd_p50, fwd_p10, fwd_p90,
                section_lookup, df_full):

    proba = inv_model.predict_proba([[wu_target, L, h0, s, fy]])[0]
    top_sections = proba.argsort()[-10:][::-1]

    results = []
    quantile_cross_issue = False

    for sec in top_sections:
        row = section_lookup[section_lookup.SectionID == sec].iloc[0]

        H, bf = int(row.H), int(row.bf)
        tw, tf = float(row.tw), float(row.tf)

        Xtest = np.array([[H, bf, tw, tf, L, h0, s, s0, se, fy]])

        Pred_wu = fwd_p50.predict(Xtest)[0]
        LB_wu = fwd_p10.predict(Xtest)[0]
        UB_wu = fwd_p90.predict(Xtest)[0]

        if UB_wu < Pred_wu:
            quantile_cross_issue = True

        error_ratio = abs(Pred_wu - wu_target) / wu_target

        app = df_full[df_full.SectionID == sec].iloc[0]

        SCI = check_SCI(H, bf, tw, tf, h0, s0, se) if app["SCI_applicable"] == 1 else -1
        ENM = check_ENM(H, bf, tw, tf, h0, s0) if app["ENM_applicable"] == 1 else -1
        AISC = check_AISC(H, bf, tw, tf, h0, s) if app["AISC_applicable"] == 1 else -1

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

    st.session_state.designer_results = df_res
    st.session_state.designer_best = df_res.sort_values("AbsErr").iloc[0]
    st.session_state.designer_optimal = df_res.iloc[0]
