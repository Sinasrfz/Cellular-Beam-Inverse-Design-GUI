# ============================================================
# designer_page.py — Core Inverse Design (3-Stage Pipeline)
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

plt.rcParams["font.family"] = "Times New Roman"]
plt.rcParams["font.size"] = 20

# ============================================================
# MAIN RENDER FUNCTION
# ============================================================

def render(inv_model, fwd_p50, fwd_p10, fwd_p90, section_lookup, df_full):

    st.header("🏗 Designer Tool — 3-Stage Inverse Design")

    st.sidebar.subheader("🧮 Design Inputs")
    wu_target = st.sidebar.number_input("Target wu (kN/m)", 5.0, 300.0, 30.0)
    L = st.sidebar.number_input("Beam span L (mm)", 5000.0, 30000.0, 12000.0)
    h0 = st.sidebar.number_input("Opening diameter h0 (mm)", 200.0, 800.0, 400.0)
    s = st.sidebar.number_input("Centre spacing s (mm)", 300.0, 2000.0, 600.0)
    fy = st.sidebar.number_input("Steel fy (MPa)", 200.0, 600.0, 355.0)
    N0 = st.sidebar.number_input("Number of openings N0", 1, 50, 10)

    # Derived geometry
    s0 = s - h0
    se = (L - ((h0 * N0) + (s0 * (N0 - 1)))) / 2

    st.sidebar.write(f"Computed s0 = {s0:.1f} mm")
    st.sidebar.write(f"Computed se = {se:.1f} mm")

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

    # Top 10 predicted sections
    proba = inv_model.predict_proba([[wu_target, L, h0, s, fy]])[0]
    top_sections = proba.argsort()[-10:][::-1]

    strict_results = []
    relaxed_results = []
    all_results = []

    for sec in top_sections:

        row = section_lookup[section_lookup.SectionID == sec].iloc[0]
        H = int(row.H)
        bf = int(row.bf)
        tw = float(row.tw)
        tf = float(row.tf)

        # Forward surrogate prediction
        X = np.array([[H, bf, tw, tf, L, h0, s, s0, se, fy]])
        wu50 = fwd_p50.predict(X)[0]
        wu10 = fwd_p10.predict(X)[0]
        wu90 = fwd_p90.predict(X)[0]

        error_ratio = abs(wu50 - wu_target) / wu_target

        # Code checks
        SCI = check_SCI(H, bf, tw, tf, h0, s0, se)
        ENM = check_ENM(H, bf, tw, tf, h0, s0)
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

        row_entry = {
            "SectionID": sec,
            "H": H, "bf": bf, "tw": tw, "tf": tf,
            "Wu_pred": wu50, "Wu_10": wu10, "Wu_90": wu90,
            "ErrorRatio": error_ratio,
            "SCI": SCI, "ENM": ENM, "AISC": AISC,
            "FailureMode": fm,
            "Weight_kg": weight,
            "Score": score
        }

        # Feasibility levels
        if error_ratio <= 0.02:
            strict_results.append(row_entry)
        if error_ratio <= 0.10:
            relaxed_results.append(row_entry)
        all_results.append(row_entry)

    # ============================================================
    # PROPER OUTPUT PRIORITY
    # ============================================================

    if strict_results:
        df_res = pd.DataFrame(strict_results)
        st.success("✔ Found designs within ±2% accuracy.")

    elif relaxed_results:
        df_res = pd.DataFrame(relaxed_results)
        st.warning("⚠ No ±2% match. Showing ±10% feasible designs.")

    else:
        df_res = pd.DataFrame(all_results)
        st.error("⚠ No feasible match. Showing closest available design.")

    # 🔥 FINAL MANDATORY SORT — ALWAYS BY SCORE
    df_res = df_res.sort_values("Score", ascending=True).reset_index(drop=True)

    # ============================================================
    # Strength Match Indicator
    # ============================================================

    st.subheader("📏 Strength Match Indicator")

    best = df_res.iloc[0]
    diff = best["Wu_pred"] - wu_target
    diff_percent = 100 * diff / wu_target

    if abs(diff) <= 0.02 * wu_target:
        st.success(f"✔ Strength perfectly matched ({diff_percent:+.2f}%).")
    elif abs(diff) <= 0.10 * wu_target:
        st.warning(f"⚠ Strength moderately matched ({diff_percent:+.2f}%).")
    else:
        st.error(f"❌ Strength mismatch ({diff_percent:+.2f}%).")

    # ============================================================
    # Code Check Color Indicators
    # ============================================================

    def check_color(val):
        return "🟩 PASS" if val == 1 else "🟥 FAIL"

    st.markdown("### 📘 Code Check Summary")
    st.write({
        "SCI": check_color(best["SCI"]),
        "ENM": check_color(best["ENM"]),
        "AISC": check_color(best["AISC"])
    })

    # ============================================================
    # Geometry Plot
    # ============================================================

    st.markdown("### 📐 Recommended Geometry")

    fig, ax = plt.subplots(figsize=(6, 4))

    H = best["H"]
    bf = best["bf"]
    tw = best["tw"]
    tf = best["tf"]

    ax.add_patch(plt.Rectangle((-bf/2, H/2 - tf), bf, tf, color="gray"))
    ax.add_patch(plt.Rectangle((-bf/2, -H/2), bf, tf, color="gray"))
    ax.add_patch(plt.Rectangle((-tw/2, -H/2), tw, H, color="lightgray"))

    ax.set_xlim(-bf, bf)
    ax.set_ylim(-H/1.2, H/1.2)
    ax.set_aspect("equal")
    ax.set_title(f"Section: H={H} mm, bf={bf} mm, tw={tw} mm, tf={tf} mm")
    ax.axis("off")
    st.pyplot(fig)

    # ============================================================
    # Final Tables
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
