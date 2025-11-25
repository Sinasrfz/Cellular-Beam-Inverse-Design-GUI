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

    st.header("📁 Dataset Explorer — Visual Analytics")

    st.write("""
    This page provides interactive tools to explore the full FEA dataset:
    histograms, scatter plots, applicability maps, distributions, 
    geometric ratios, and correlations.
    """)

    # ============================================================
    # 1 — Dataset Preview
    # ============================================================
    st.subheader("📄 1. Dataset Preview")
    st.dataframe(df_full.head(30))

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

    fig, ax = plt.subplots(figsize=(12, 12))
    cax = ax.imshow(corr, cmap="viridis")
    fig.colorbar(cax, shrink=0.8)

    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(corr_cols, fontsize=12)

    # Annotate correlation values
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(
                j, i, f"{corr.iloc[i,j]:.2f}",
                ha="center", va="center",
                color="white", fontsize=10
            )

    plt.subplots_adjust(top=0.92, bottom=0.1, left=0.15, right=0.98)
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 4 — Scatter Plots
    # ============================================================
    st.subheader("📐 4. Key Scatter Plots (wu_FEA vs Parameters)")

    plot_cols = ["H", "tw", "h0", "s", "fy"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    axes = axes.flatten()

    for ax, col in zip(axes[:5], plot_cols):
        ax.scatter(df_full[col], df_full["wu_FEA"], alpha=0.45, s=18)
        ax.set_xlabel(col, fontsize=14)
        ax.set_ylabel("wu_FEA", fontsize=14)
        ax.set_title(f"wu_FEA vs {col}", fontsize=16)

    # Remove empty subplot
    fig.delaxes(axes[5])

    plt.subplots_adjust(hspace=0.45, wspace=0.30)
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 5 — Opening Geometry Explorer
    # ============================================================
    st.subheader("🕳 5. Opening Geometry Explorer")

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].scatter(df_full["h0"], df_full["wu_FEA"], alpha=0.4)
    ax[0].set_xlabel("h0 (mm)")
    ax[0].set_ylabel("wu_FEA")
    ax[0].set_title("Strength vs Opening Diameter")

    ax[1].scatter(df_full["s"] / df_full["h0"], df_full["wu_FEA"], alpha=0.4)
    ax[1].set_xlabel("s / h0")
    ax[1].set_ylabel("wu_FEA")
    ax[1].set_title("Strength vs s/h0 Ratio")

    plt.subplots_adjust(wspace=0.30)
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 6 — Code Applicability Summary
    # ============================================================
    st.subheader("📙 6. Code Applicability Summary")

    appl_cols = ["SCI_applicable", "ENM_applicable", "AISC_applicable"]

    fig, ax = plt.subplots(figsize=(10, 6))
    vals = df_full[appl_cols].mean()

    ax.bar(appl_cols, vals, color="#2f7ed8")
    ax.set_ylabel("Applicability Ratio", fontsize=16)
    ax.set_title("Code Applicability Rates", fontsize=18)

    plt.xticks(rotation=30, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 7 — Failure Mode Distribution
    # ============================================================
    st.subheader("⚠ 7. Failure Mode Distribution")

    fig, ax = plt.subplots(figsize=(7, 6))
    df_full["Failure_mode"].value_counts().plot.pie(
        autopct="%1.1f%%", ax=ax, textprops={"fontsize": 13}
    )
    ax.set_ylabel("")
    ax.set_title("Failure Mode Breakdown", fontsize=18)

    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # 8 — Variable Distributions
    # ============================================================
    st.subheader("📦 8. Variable Distributions")

    dist_cols = ["wu_FEA","H","bf","tw","tf","h0","s","s0","se","L","fy"]

    rows, cols = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 16))
    axes = axes.flatten()

    for ax, col in zip(axes[:len(dist_cols)], dist_cols):
        ax.hist(df_full[col], bins=20, alpha=0.7, edgecolor="black")
        ax.set_title(col, fontsize=14)
        ax.tick_params(labelsize=12)

    # Remove unused subplot slots
    for i in range(len(dist_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.subplots_adjust(hspace=0.55, wspace=0.35)
    st.pyplot(fig)

    st.markdown("---")

    st.success("✔ Dataset Explorer Loaded Successfully")
