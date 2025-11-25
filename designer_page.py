# ============================================================
# designer_page.py — Practical Inverse Design (Hybrid PSO + Unique Designs Only)
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
# MAIN RENDER FUNCTION
# ============================================================

def render(inv_model, fwd_p50, fwd_p10, fwd_p90, section_lookup, df_full):

    st.header("🏗 Designer Tool — 3-Stage Inverse Design (Hybrid PSO Enabled)")

    st.sidebar.subheader("🧮 Design Inputs")
    wu_target = st.sidebar.number_input("Target wu (kN/m)", 5.0, 300.0, 30.0)
    L = st.sidebar.number_input("Beam span L (mm)", 5000.0, 30000.0, 12000.0)
    h0 = st.sidebar.number_input("Opening diameter h0 (mm)", 200.0, 800.0, 400.0)
    s = st.sidebar.number_input("Centre spacing s (mm)", 300.0, 2000.0, 600.0)
    fy = st.sidebar.number_input("Steel fy (MPa)", 200.0, 600.0, 355.0)
    N0 = st.sidebar.number_input("Number of openings N0", 1, 50, 10)

    # derived geometry
    s0 = s - h0
    se = (L - (h0 * N0 + s0 * (N0 - 1))) / 2

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

    # Top 10 candidates
    proba = inv_model.predict_proba([[wu_target, L, h0, s, fy]])[0]
    top_sections = proba.argsort()[-10:][::-1]

    strict_results = []
    relaxed_results = []
    all_results = []

    # Preload manufacturable options
    H_values  = sorted(df_full["H"].unique())
    bf_values = sorted(df_full["bf"].unique())
    tw_values = sorted(df_full["tw"].unique())
    tf_values = sorted(df_full["tf"].unique())

    def nearest_index(values, target):
        return int(np.argmin([abs(v - target) for v in values]))

    # ============================================================
    # For each predicted section
    # ============================================================

    for sec in top_sections:

        row = section_lookup[section_lookup.SectionID == sec].iloc[0]
        H_base = int(row.H)
        bf_base = int(row.bf)
        tw_base = float(row.tw)
        tf_base = float(row.tf)

        # -------------------------------------------------------
        # Create LOCAL NEIGHBORHOOD BOUNDS (±1 catalog step)
        # -------------------------------------------------------
        H_idx0  = nearest_index(H_values,  H_base)
        bf_idx0 = nearest_index(bf_values, bf_base)
        tw_idx0 = nearest_index(tw_values, tw_base)
        tf_idx0 = nearest_index(tf_values, tf_base)

        def local_range(values, idx):
            lo = max(idx - 1, 0)
            hi = min(idx + 1, len(values)-1)
            return list(range(lo, hi+1))

        H_idx_list  = local_range(H_values,  H_idx0)
        bf_idx_list = local_range(bf_values, bf_idx0)
        tw_idx_list = local_range(tw_values, tw_idx0)
        tf_idx_list = local_range(tf_values, tf_idx0)

        # ======================================================
        # PSO REFINEMENT (DISCRETE + RANDOM INITIALIZATION)
        # ======================================================

        num_particles = 12
        num_iters = 25
        w = 0.7
        c1 = 1.5
        c2 = 1.5

        pos = np.zeros((num_particles, 4))
        vel = np.zeros((num_particles, 4))

        def rand_choice(lst):
            return np.random.choice(lst)

        for i in range(num_particles):
            pos[i] = [
                rand_choice(H_idx_list),
                rand_choice(bf_idx_list),
                rand_choice(tw_idx_list),
                rand_choice(tf_idx_list)
            ]

        # Fitness
        def fitness(idx_vec):
            hi  = int(np.clip(round(idx_vec[0]), 0, len(H_values)-1))
            bfi = int(np.clip(round(idx_vec[1]), 0, len(bf_values)-1))
            twi = int(np.clip(round(idx_vec[2]), 0, len(tw_values)-1))
            tfi = int(np.clip(round(idx_vec[3]), 0, len(tf_values)-1))

            Hc  = H_values[hi]
            bfc = bf_values[bfi]
            twc = tw_values[twi]
            tfc = tf_values[tfi]

            Xp  = np.array([[Hc, bfc, twc, tfc, L, h0, s, s0, se, fy]])
            wu  = fwd_p50.predict(Xp)[0]
            return abs(wu - wu_target)

        pbest = pos.copy()
        pbest_fit = np.array([fitness(p) for p in pos])

        gbest = pbest[np.argmin(pbest_fit)]
        gbest_fit = np.min(pbest_fit)

        # PSO loop
        for _ in range(num_iters):
            for i in range(num_particles):

                vel[i] = (
                    w * vel[i]
                    + c1 * np.random.rand() * (pbest[i] - pos[i])
                    + c2 * np.random.rand() * (gbest    - pos[i])
                )

                pos[i] += vel[i]
                fit = fitness(pos[i])

                if fit < pbest_fit[i]:
                    pbest[i] = pos[i].copy()
                    pbest_fit[i] = fit

                    if fit < gbest_fit:
                        gbest = pos[i].copy()
                        gbest_fit = fit

        # Final PSO geometry
        final_hi  = int(np.clip(round(gbest[0]), 0, len(H_values)-1))
        final_bfi = int(np.clip(round(gbest[1]), 0, len(bf_values)-1))
        final_twi = int(np.clip(round(gbest[2]), 0, len(tw_values)-1))
        final_tfi = int(np.clip(round(gbest[3]), 0, len(tf_values)-1))

        H  = H_values[final_hi]
        bf = bf_values[final_bfi]
        tw = tw_values[final_twi]
        tf = tf_values[final_tfi]

        # ------------------------------------------------------
        # Forward surrogate for optimized geometry
        # ------------------------------------------------------
        X = np.array([[H, bf, tw, tf, L, h0, s, s0, se, fy]])
        wu50 = fwd_p50.predict(X)[0]
        wu10 = fwd_p10.predict(X)[0]
        wu90 = fwd_p90.predict(X)[0]

        error_ratio = abs(wu50 - wu_target) / wu_target

        SCI = check_SCI(H, bf, tw, tf, h0, s0, se)
        ENM = check_ENM(H, bf, tw, tf, h0, s0)
        AISC = check_AISC(H, bf, tw, tf, h0, s)

        fm_series = df_full[df_full.SectionID == sec]["Failure_mode"]
        fm = fm_series.mode()[0] if not fm_series.mode().empty else "Unknown"

        weight = compute_weight(H, bf, tw, tf, L)
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

        if error_ratio <= 0.02:
            strict_results.append(row_entry)
        if error_ratio <= 0.10:
            relaxed_results.append(row_entry)

        all_results.append(row_entry)

    # ============================================================
    # Build final result DataFrame
    # ============================================================

    if strict_results:
        df_res = pd.DataFrame(strict_results)
        st.success("✔ Found designs within ±2% accuracy.")
    elif relaxed_results:
        df_res = pd.DataFrame(relaxed_results)
        st.warning("⚠ Showing ±10% feasible designs.")
    else:
        df_res = pd.DataFrame(all_results)
        st.error("⚠ No match within ±10%. Showing closest designs.")

    # ============================================================
    # REMOVE DUPLICATE GEOMETRIES
    # ============================================================
    df_res = df_res.drop_duplicates(subset=["H","bf","tw","tf"]).reset_index(drop=True)

    # Sort by score
    df_res = df_res.sort_values("Score", ascending=True).reset_index(drop=True)

    # ============================================================
    # Strength Summary
    # ============================================================

    st.subheader("📏 Strength Match Indicator")

    best = df_res.iloc[0]
    diff = best["Wu_pred"] - wu_target
    diff_percent = 100 * diff / wu_target

    if abs(diff) <= 0.02 * wu_target:
        st.success(f"✔ Strength perfectly matched ({diff_percent:+.2f}%).")
    elif abs(diff) <= 0.10 * wu_target:
        st.warning(f"⚠ Moderately matched ({diff_percent:+.2f}%).")
    else:
        st.error(f"❌ Strength mismatch ({diff_percent:+.2f}%).")

    # ============================================================
    # Code Check Summary
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
