# ============================================================
# explorer_page.py — Full Dataset Explorer (10 Feature Version)
# (NO seaborn — Streamlit Cloud safe)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18


# ============================================================
# MAIN RENDER FUNCTION
# ============================================================

def render(df_full):

    st.header("📁 Dataset Explorer — Full Statistical & Structural Insights")

    st.write("""
    This page provides a comprehensive exploration of the full dataset used 
    for training the inverse model, forward surrogate, and code classification.
    """)

    # ============================================================
    # 1 — INTERACTIVE FILTERING
    # ============================================================

    st.subheader("🔎 1. Interactive Filtering Panel")

    with st.expander("Open Filters"):
        col1, col2, col3 = st.columns(3)

        H_min, H_max = int(df_full["H"].min()), int(df_full["H"].max())
        bf_min, bf_max = int(df_full["bf"].min()), int(df_full["bf"].max())
        tw_min, tw_max = float(df_full["tw"].min()), float(df_full["tw"].max())
        tf_min, tf_max = float(df_full["tf"].min()), float(df_full["tf"].max())
        s_min, s_max = float(df_full["s"].min()), float(df_full["s"].max())
        fy_min, fy_max = int(df_full["fy"].min()), int(df_full["fy"].max())

        with col1:
            H_range = st.slider("H (mm)", H_min, H_max, (H_min, H_max))
            bf_range = st.slider("bf (mm)", bf_min, bf_max, (bf_min, bf_max))

        with col2:
            tw_range = st.slider("tw (mm)", tw_min, tw_max, (tw_min, tw_max))
            tf_range = st.slider("tf (mm)", tf_min, tf_max, (tf_min, tf_max))

        with col3:
            s_range  = st.slider("s (mm)", s_min, s_max, (s_min, s_max))
            fy_range = st.slider("fy (MPa)", fy_min, fy_max, (fy_min, fy_max))

        filtered = df_full[
            (df_full["H"].between(*H_range)) &
            (df_full["bf"].between(*bf_range)) &
            (df_full["tw"].between(*tw_range)) &
            (df_full["tf"].between(*tf_range)) &
            (df_full["s"].between(*s_range)) &
            (df_full["fy"].between(*fy_range))
        ]

        st.write(f"Filtered samples: **{len(filtered)}**")
        st.dataframe(filtered)

    st.markdown("---")

    # ============================================================
    # 2 — SUMMARY STATISTICS
    # ============================================================

    st.subheader("📊 2. Summary Statistics")
    st.dataframe(df_full.describe())

    st.markdown("---")

    # ============================================================
    # 3 — CORRELATION HEATMAP (Matplotlib)
    # ============================================================

    st.subheader("🧩 3. Correlation Matrix Heatmap")

    corr_cols = ["wu_FEA", "H", "bf", "tw", "tf", "h0", "s", "s0", "se", "L", "fy"]
    corr = df_full[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, cmap="viridis")
    fig.colorbar(cax)

    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticklabels(corr_cols)

    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", color="white")

    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 4 — PAIR PLOTS (Key Relationships)
    # ============================================================

    st.subheader("📐 4. Key Scatter Plots (FEA vs Parameters)")

    plot_cols = ["H", "tw", "h0", "s", "fy"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    axes = axes.flatten()

    for ax, col in zip(axes, plot_cols):
        ax.scatter(df_full[col], df_full["wu_FEA"], alpha=0.4)
        ax.set_xlabel(col)
        ax.set_ylabel("wu_FEA")
        ax.set_title(f"wu_FEA vs {col}")

    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 5 — SECTION BROWSER
    # ============================================================

    st.subheader("📘 5. Section Browser")

    unique_sections = df_full[["H", "bf", "tw", "tf"]].drop_duplicates()

    index = st.selectbox(
        "Select a section:", 
        unique_sections.index,
        format_func=lambda i: str(tuple(unique_sections.loc[i]))
    )

    sec_data = unique_sections.loc[index]
    Hc, bfc, twc, tfc = sec_data["H"], sec_data["bf"], sec_data["tw"], sec_data["tf"]

    subset = df_full[
        (df_full["H"] == Hc) &
        (df_full["bf"] == bfc) &
        (df_full["tw"] == twc) &
        (df_full["tf"] == tfc)
    ]

    st.write(f"Samples for this section: **{len(subset)}**")
    st.dataframe(subset)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(subset["wu_FEA"], bins=20)
    ax.set_title("Strength Distribution (wu_FEA)")
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 6 — OPENING GEOMETRY EXPLORER
    # ============================================================

    st.subheader("🕳 6. Opening Geometry Explorer")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].scatter(df_full["h0"], df_full["wu_FEA"], alpha=0.4)
    ax[0].set_xlabel("h0 (mm)")
    ax[0].set_ylabel("wu_FEA")

    ax[1].scatter(df_full["s"]/df_full["h0"], df_full["wu_FEA"], alpha=0.4)
    ax[1].set_xlabel("s / h0")
    ax[1].set_ylabel("wu_FEA")

    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 7 — CODE APPLICABILITY SUMMARY
    # ============================================================

    st.subheader("📙 7. Code Applicability Summary")

    appl_cols = ["SCI_applicable", "ENM_applicable", "AISC_applicable"]

    st.write(df_full[appl_cols].mean())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(appl_cols, df_full[appl_cols].mean())
    ax.set_title("Code Applicability Rates")
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 8 — FAILURE MODE EXPLORER
    # ============================================================

    st.subheader("⚠ 8. Failure Mode Explorer")

    fm_counts = df_full["Failure_mode"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(fm_counts.index, fm_counts.values)
    ax.set_title("Failure Mode Distribution")
    ax.set_xlabel("Failure Mode")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 9 — VARIABLE DISTRIBUTIONS
    # ============================================================

    st.subheader("📦 9. Variable Distributions")

    dist_cols = ["wu_FEA","H","bf","tw","tf","h0","s","s0","se","L","fy"]

    fig, axes = plt.subplots(len(dist_cols)//3 + 1, 3, figsize=(14, 14))
    axes = axes.flatten()

    for ax, col in zip(axes, dist_cols):
        ax.hist(df_full[col], bins=20, alpha=0.7)
        ax.set_title(col)

    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 10 — OUTLIER DETECTION
    # ============================================================

    st.subheader("🚨 10. Outlier Detection")

    wu_mean = df_full["wu_FEA"].mean()
    wu_std  = df_full["wu_FEA"].std()

    outliers = df_full[
        (df_full["wu_FEA"] > wu_mean + 3*wu_std) |
        (df_full["wu_FEA"] < wu_mean - 3*wu_std)
    ]

    st.write(f"Outliers detected: **{len(outliers)}**")
    st.dataframe(outliers)

    st.info("Outliers may represent extreme geometries or special failure mechanisms. They are not necessarily errors.")

