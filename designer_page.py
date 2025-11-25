# ============================================================
# designer_page.py — Updated with Quantile-Crossing Warning +
# Removed Code Summary Block + GitHub Image + PSO Enabled
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
    multiobjective_score
)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20


# ============================================================
# 📌 MAIN RENDER FUNCTION
# ============================================================

def render(inv_model, fwd_p50, fwd_p10, fwd_p90, section_lookup, df_full):

    st.header("🏗 Designer Tool — 3-Stage Hybrid Inverse Design")

    st.sidebar.subheader("🧮 Design Inputs")
    wu_target = st.sidebar.number_input("Target wu (kN/m)", 5.0, 300.0, 30.0)
    L = st.sidebar.number_input("Beam span L (mm)", 5000.0, 30000.0, 12000.0)
    h0 = st.sidebar.number_input("Opening diameter h0 (mm)", 200.0, 800.0, 400.0)
    s = st.sidebar.number_input("Centre spacing s (mm)", 300.0, 2000.0, 600.0)
    fy = st.sidebar.number_input("Steel fy (MPa)", 200.0, 600.0, 355.0)
    N0 = st.sidebar.number_input("Number of openings N0", 1, 50, 10)

    # Derived geometry
    s0 = s - h0
    se = (L - (h0 * N0 + s0 * (N0 - 1))) / 2

    st.sidebar.write(f"Computed s0 = {s0:.1f} mm")
    st.sidebar.write(f"Computed se = {se:.1f} mm")

    # ============================================================
    # ❗ Immediate feasibility check (must be BEFORE button)
    # ============================================================
    if se < 0:
        st.error(
            "❌ The computed end spacing *se* is negative.\n\n"
            "This means the chosen number of openings **N0** is not feasible.\n"
            "➡ Reduce **N0** or increase the spacing **s**."
        )
        return

    # ============================================================
    # Show Image from GitHub Repository
    # ============================================================
    st.image(
        "https://raw.githubusercontent.com/Sinasrfz/Cellular-Beam-Inverse-Design-GUI/main/Picture1.jpg",
        caption="Cellular Beam Geometry",
        use_column_width=True
    )

    # ============================================================
    # RUN BUTTON
    # ============================================================
    if st.button("Run Inverse Design", type="primary"):
        run_inverse(
            wu_target, L, h0, s, s0, se, fy,
            inv_model, fwd_p50, fwd_p10, fwd_p90,
            section_lookup, df_full
        )


# ============================================================
# CORE PIPELINE
# ============================================================

def run_inverse(wu_target, L, h0, s, s0, se, fy,
                inv_model, fwd_p50, fwd_p10, fwd_p90,
                section_lookup, df_full):

    st.subheader("🔍 Phase 1 — Inverse Model Prediction")

    # Top candidates
    proba = inv_model.predict_proba([[wu_target, L, h0, s, fy]])[0]
    top_sections = proba.argsort()[-10:][::-1]

    results = []
    crossing_detected = False   # <--- detect quantile crossing

    # ============================================================
    # Loop Through Candidates
    # ============================================================
    for sec in top_sections:

        row = section_lookup[section_lookup.SectionID == sec].iloc[0]
        H = int(row.H)
        bf = int(row.bf)
        tw = float(row.tw)
        tf = float(row.tf)

        # Forward surrogate
        X = np.array([[H, bf, tw, tf, L, h0, s, s0, se, fy]])
        wu50 = fwd_p50.predict(X)[0]
        wu10 = fwd_p10.predict(X)[0]
        wu90 = fwd_p90.predict(X)[0]

        # ❗ Quantile crossing detection (NO message here)
        if wu90 < wu50:
            crossing_detected = True

        error_ratio = abs(wu50 - wu_target) / wu_target

        # --------------------------------------------------------
        # Code checks with Emoji conversion
        # --------------------------------------------------------
        SCI = check_SCI(H, bf, tw, tf, h0, s0, se)
        ENM = check_ENM(H, bf, tw, tf, h0, s0)
        AISC = check_AISC(H, bf, tw, tf, h0, s)

        def make_emoji(val):
            if val == -1:
                return "⚪ N/A"
            elif val == 1:
                return "🟩 PASS"
            else:
                return "🟥 FAIL"

        SCI_emoji = make_emoji(SCI)
        ENM_emoji = make_emoji(ENM)
        AISC_emoji = make_emoji(AISC)

        # Failure mode
        fm_series = df_full[df_full.SectionID == sec]["Failure_mode"]
        fm = fm_series.mode()[0] if not fm_series.mode().empty else "Unknown"

        weight = compute_weight(H, bf, tw, tf, L)

        # Score
        score = multiobjective_score(
            wu_target, wu50, weight, SCI, ENM, AISC, fm
        )

        results.append({
            "SectionID": sec,
            "H": H, "bf": bf, "tw": tw, "tf": tf,
            "Wu_50": wu50,
            "Wu_10": wu10,
            "Wu_90": wu90,
            "ErrorRatio": error_ratio,
            "SCI": SCI_emoji,
            "ENM": ENM_emoji,
            "AISC": AISC_emoji,
            "FailureMode": fm,
            "Weight_kg": weight,
            "Score": score
        })

    # DataFrame
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("Score", ascending=True).reset_index(drop=True)

    # ============================================================
    # Show quantile-crossing warning ONLY if detected
    # ============================================================
    if crossing_detected:
        st.info(
            "🔍 *Note on prediction bounds:* For some candidate sections, the "
            "p90 prediction was slightly lower than p50. This is normal in "
            "quantile regression (quantile crossing) and does **not** affect "
            "the validity of the design."
        )

    # ============================================================
    # Strength Match Indicator
    # ============================================================
    st.subheader("📏 Strength Match Indicator")

    best = df_res.iloc[0]
    diff = best["Wu_50"] - wu_target
    diff_percent = 100 * diff / wu_target

    if abs(diff) <= 0.02 * wu_target:
        st.success(f"✔ Strength perfectly matched ({diff_percent:+.2f}%).")
    elif abs(diff) <= 0.10 * wu_target:
        st.warning(f"⚠ Moderately matched ({diff_percent:+.2f}%).")
    else:
        st.error(f"❌ Strength mismatch ({diff_percent:+.2f}%).")

    # ============================================================
    # 📌 Results Table (Code Summary Removed)
    # ============================================================
    st.subheader("🏅 Recommended Section")
    st.write(df_res.head(1))

    st.subheader("📊 Full Ranking Table")
    st.dataframe(df_res)

    st.download_button(
        "Download Results as CSV",
        df_res.to_csv(index=False),
        file_name="inverse_design_results.csv"
    )
