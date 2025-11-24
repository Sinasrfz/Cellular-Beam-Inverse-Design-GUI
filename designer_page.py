# ============================================================
# designer_page.py — Core Inverse Design (3-Stage Pipeline)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# IMPORT YOUR STAGE-1 FUNCTIONS
# ==========================================
from stage1_functions import (
    check_SCI,
    check_ENM,
    check_AISC,
    compute_weight,
    multiobjective_score
)

# ============================================================
# MAIN RENDER FUNCTION
# ============================================================

def render(inv_model, fwd_p50, fwd_p10, fwd_p90, section_lookup, df_full):

    # Normalize both dataframes (VERY IMPORTANT)
    section_lookup.columns = (
        section_lookup.columns
        .str.replace(" ", "")
        .str.replace(",", "")
        .str.replace("×", "x")
        .str.strip()
    )

    df_full.columns = (
        df_full.columns
        .str.replace(" ", "")
        .str.replace(",", "")
        .str.replace("×", "x")
        .str.strip()
    )

    st.header("🏗 Designer Tool — 3-Stage Inverse Design")

    # --------------------- INPUTS ------------------------
    st.sidebar.subheader("🧮 Design Inputs")
    wu_target = st.sidebar.number_input("Target wu (kN/m)", 5.0, 300.0, 30.0)
    L = st.sidebar.number_input("Beam span L (mm)", 5000.0, 30000.0, 12000.0)
    h0 = st.sidebar.number_input("Opening diameter h0 (mm)", 200.0, 800.0, 400.0)
    s = st.sidebar.number_input("Centre spacing s (mm)", 300.0, 2000.0, 600.0)
    fy = st.sidebar.number_input("Steel fy (MPa)", 200.0, 600.0, 355.0)
    N0 = st.sidebar.number_input("Number of openings N0", 1, 50, 10)

    # ------------------- Compute derived -------------------
    s0 = s - h0
    se = (L - (h0 * N0) + (s0 * (N0 - 1))) / 2

    st.sidebar.write(f"Computed s0 = {s0:.1f} mm")
    st.sidebar.write(f"Computed se = {se:.1f} mm")

    if st.button("Run Inverse Design", type="primary"):
        run_inverse(wu_target, L, h0, s, s0, se, fy,
                    inv_model, fwd_p50, fwd_p10, fwd_p90,
                    section_lookup, df_full)


# ============================================================
# CORE PIPELINE
# ============================================================

def run_inverse(wu_target, L, h0, s, s0, se, fy,
                inv_model, fwd_p50, fwd_p10, fwd_p90,
                section_lookup, df_full):

    st.subheader("🔍 Phase 1 — Inverse Model Prediction (Top Sections)")

    # Predict top-5 SectionIDs
    proba = inv_model.predict_proba([[wu_target, L, h0, s, fy]])[0]
    top_sections = proba.argsort()[-5:][::-1]

    results = []

    for sec in top_sections:

        # ----------------------------
        # SECTION LOOKUP
        # ----------------------------
        row_df = section_lookup[section_lookup["SectionID"] == sec]

        if row_df.empty:
            st.warning(f"⚠ SectionID {sec} not found in lookup table.")
            continue

        row = row_df.iloc[0]

        # Ensure correct numeric types
        H  = float(row["H"])
        bf = float(row["bf"])
        tw = float(row["tw"])
        tf = float(row["tf"])

        # ----------------------------
        # FORWARD SURROGATES
        # ----------------------------
        X = np.array([[H, bf, tw, tf, L, h0, s, s0, se, fy]])
        wu50 = float(fwd_p50.predict(X)[0])
        wu10 = float(fwd_p10.predict(X)[0])
        wu90 = float(fwd_p90.predict(X)[0])

        # ----------------------------
        # CODE CHECKS
        # ----------------------------
        SCI  = check_SCI(H, bf, tw, tf, h0, s0, se)
        ENM  = check_ENM(H, bf, tw, tf, h0, s0)
        AISC = check_AISC(H, bf, tw, tf, h0, s)

        # ----------------------------
        # FAILURE MODE LOOKUP
        # ----------------------------
        fm_df = df_full[df_full["SectionID"] == sec]["Failure_mode"]
        fm = fm_df.mode()[0] if not fm_df.mode().empty else "Unknown"

        # ----------------------------
        # WEIGHT
        # ----------------------------
        weight = compute_weight(H, bf, tw, tf, L)

        # ----------------------------
        # SCORE
        # ----------------------------
        score = multiobjective_score(
            wu_target, wu50, weight,
            SCI, ENM, AISC, fm
        )

        results.append({
            "SectionID": sec,
            "H": H, "bf": bf, "tw": tw, "tf": tf,
            "Wu_pred": wu50, "Wu_10": wu10, "Wu_90": wu90,
            "SCI": SCI, "ENM": ENM, "AISC": AISC,
            "FailureMode": fm,
            "Weight_kg": weight,
            "Score": score
        })

    df_res = pd.DataFrame(results).sort_values("Score", ascending=True)

    st.subheader("🏅 Recommended Section")
    st.write(df_res.head(1))

    st.subheader("📊 Full Ranking Table")
    st.dataframe(df_res)

    st.download_button(
        "Download Results as CSV",
        df_res.to_csv(index=False),
        file_name="inverse_design_results.csv"
    )
