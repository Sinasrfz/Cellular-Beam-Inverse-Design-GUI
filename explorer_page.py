# ============================================================
# explorer_page.py — Dataset Explorer & Visual Analytics
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
    # 1 — INTERACTIVE FILTERING PANEL
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
    # 2 — Summary Statistics
    # ============================================================
    st.subheader("📊 2. Summary Statistics")
    st.dataframe(df_full.describe())

    st.markdown("---")

    # ============================================================
    # 3 — Correlation Heatmap
    # ============================================================
    st.subheader("🧩 3. Correlation Matrix Heatmap")

    corr_cols = ["wu_FEA","H","bf","tw","tf","h0","s","s0","se","L","fy"]
    corr = df_full[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(corr, cmap="viridis")
    fig.colorbar(cax, shrink=0.8)

    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(corr_cols, fontsize=10)

    # Annotate correlation values
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(
                j, i, f"{corr.iloc[i,j]:.2f}",
                ha="center", va="center",
                color="white", fontsize=8
            )

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 4 — Scatter Plots (Fixed spacing)
    # ============================================================
    st.subheader("📐 4. Key Scatter Plots (wu_FEA vs Parameters)")

    plot_cols = ["H", "tw", "h0", "s", "fy"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    axes = axes.flatten()

    for ax, col in zip(axes[:5], plot_cols):
        ax.scatter(df_full[col], df_full["wu_FEA"], alpha=0.45, s=18)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("wu_FEA", fontsize=12)
        ax.set_title(f"wu_FEA vs {col}", fontsize=13)

    fig.delaxes(axes[5])
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 5 — Opening Geometry Explorer (fixed spacing)
    # ============================================================
    st.subheader("🕳 5. Opening Geometry Explorer")

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(df_full["h0"], df_full["wu_FEA"], alpha=0.4)
    ax[0].set_xlabel("h0 (mm)")
    ax[0].set_ylabel("wu_FEA")
    ax[0].set_title("Strength vs Opening Diameter")

    ax[1].scatter(df_full["s"] / df_full["h0"], df_full["wu_FEA"], alpha=0.4)
    ax[1].set_xlabel("s / h0")
    ax[1].set_ylabel("wu_FEA")
    ax[1].set_title("Strength vs s/h0 Ratio")

    plt.subplots_adjust(wspace=0.35)
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 6 — Code Applicability Summary (fixed label overlap)
    # ============================================================
    st.subheader("📙 6. Code Applicability Summary")

    appl_cols = ["SCI_applicable", "ENM_applicable", "AISC_applicable"]
    vals = df_full[appl_cols].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(appl_cols, vals, color="#2f7ed8")
    ax.set_ylabel("Applicability Ratio", fontsize=14)
    ax.set_title("Code Applicability Rates", fontsize=16)

    plt.xticks(rotation=20, ha="right", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 7 — Failure Mode Distribution
    # ============================================================
    st.subheader("⚠ 7. Failure Mode Distribution")

    fig, ax = plt.subplots(figsize=(7, 6))
    df_full["Failure_mode"].value_counts().plot.pie(
        autopct="%1.1f%%", ax=ax, textprops={"fontsize": 12}
    )
    ax.set_ylabel("")
    ax.set_title("Failure Mode Breakdown", fontsize=16)
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 8 — Variable Distributions (fixed spacing)
    # ============================================================
    st.subheader("📦 8. Variable Distributions")

    dist_cols = ["wu_FEA","H","bf","tw","tf","h0","s","s0","se","L","fy"]

    rows, cols = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 18))
    axes = axes.flatten()

    for ax, col in zip(axes[:len(dist_cols)], dist_cols):
        ax.hist(df_full[col], bins=20, alpha=0.7, edgecolor="black")
        ax.set_title(col, fontsize=13)
        ax.tick_params(labelsize=11)

    # Remove unused subplot slots
    for i in range(len(dist_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.subplots_adjust(hspace=0.6, wspace=0.4)
    st.pyplot(fig)

    st.markdown("---")

    st.success("✔ Dataset Explorer Loaded Successfully")
