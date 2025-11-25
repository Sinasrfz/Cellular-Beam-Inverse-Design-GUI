# ============================================================
# explorer_page.py — Full Dataset Explorer (10 Feature Version)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18


# ============================================================
# MAIN RENDER FUNCTION
# ============================================================

def render(df_full):

    st.header("📁 Dataset Explorer — Full Statistical & Structural Insights")

    st.write("""
    This page provides a comprehensive exploration of the full dataset used 
    to train the inverse model, forward surrogate, and structural code classifiers.
    """)

    # ============================================================
    # FEATURE 1 — INTERACTIVE FILTER PANEL
    # ============================================================

    st.subheader("🔎 1. Interactive Filtering")

    with st.expander("Click to open filtering panel"):
        col1, col2, col3 = st.columns(3)

        H_min, H_max = int(df_full["H"].min()), int(df_full["H"].max())
        bf_min, bf_max = int(df_full["bf"].min()), int(df_full["bf"].max())
        tw_min, tw_max = float(df_full["tw"].min()), float(df_full["tw"].max())
        tf_min, tf_max = float(df_full["tf"].min()), float(df_full["tf"].max())

        with col1:
            H_range = st.slider("H (mm)", H_min, H_max, (H_min, H_max))
            bf_range = st.slider("bf (mm)", bf_min, bf_max, (bf_min, bf_max))
        with col2:
            tw_range = st.slider("tw (mm)", tw_min, tw_max, (tw_min, tw_max))
            tf_range = st.slider("tf (mm)", tf_min, tf_max, (tf_min, tf_max))
        with col3:
            s_range = st.slider("s (mm)", float(df_full["s"].min()), float(df_full["s"].max()),
                                (float(df_full["s"].min()), float(df_full["s"].max())))
            fy_range = st.slider("fy (MPa)", int(df_full["fy"].min()), int(df_full["fy"].max()),
                                 (int(df_full["fy"].min()), int(df_full["fy"].max())))

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
    # FEATURE 2 — SUMMARY STATISTICS
    # ============================================================

    st.subheader("📊 2. Summary Statistics")
    st.write(df_full.describe())

    st.markdown("---")

    # ============================================================
    # FEATURE 3 — CORRELATION HEATMAP
    # ============================================================

    st.subheader("🧩 3. Correlation Matrix Heatmap")

    corr_cols = [
        "wu_FEA", "H", "bf", "tw", "tf", "h0", "s", "s0", "se", "L", "fy"
    ]

    corr = df_full[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="viridis", annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # FEATURE 4 — PAIR PLOTS (KEY RELATIONSHIPS)
    # ============================================================

    st.subheader("📐 4. Key Scatter Relationships")

    plot_cols = ["wu_FEA", "H", "tw", "h0", "s", "fy"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))

    for ax, col in zip(axes.flatten(), plot_cols[1:]):
        ax.scatter(df_full[col], df_full["wu_FEA"], alpha=0.4)
        ax.set_xlabel(col)
        ax.set_ylabel("wu_FEA")
        ax.set_title(f"wu_FEA vs {col}")

    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # FEATURE 5 — SECTION BROWSER
    # ============================================================

    st.subheader("📘 5. Section Browser (Catalog View)")

    unique_sections = df_full[["H", "bf", "tw", "tf"]].drop_duplicates()

    sec = st.selectbox("Select a section:", unique_sections.index,
                       format_func=lambda i: str(tuple(unique_sections.loc[i])))

    chosen = unique_sections.loc[sec]
    Hc, bfc, twc, tfc = chosen["H"], chosen["bf"], chosen["tw"], chosen["tf"]

    subset = df_full[
        (df_full["H"] == Hc) &
        (df_full["bf"] == bfc) &
        (df_full["tw"] == twc) &
        (df_full["tf"] == tfc)
    ]

    st.write(f"Samples found: **{len(subset)}**")
    st.dataframe(subset)

    # Strength distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(subset["wu_FEA"], bins=20)
    ax.set_title("Strength distribution for this section")
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # FEATURE 6 — OPENING GEOMETRY EXPLORER
    # ============================================================

    st.subheader("🕳 6. Opening Geometry Explorer")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(df_full["h0"], df_full["wu_FEA"], alpha=0.4)
    ax[0].set_xlabel("h0 (mm)")
    ax[0].set_ylabel("wu_FEA")

    ax[1].scatter(df_full["s"] / df_full["h0"], df_full["wu_FEA"], alpha=0.4)
    ax[1].set_xlabel("s / h0")
    ax[1].set_ylabel("wu_FEA")

    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # FEATURE 7 — CODE APPLICABILITY SUMMARY
    # ============================================================

    st.subheader("📙 7. Code Applicability Summary")

    appl_cols = ["SCI_applicable", "ENM_applicable", "AISC_applicable"]
    st.write(df_full[appl_cols].mean())

    fig, ax = plt.subplots(figsize=(6, 4))
    df_full[appl_cols].mean().plot(kind="bar", ax=ax)
    ax.set_title("Fraction of Samples where Code is Applicable")
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # FEATURE 8 — FAILURE MODE EXPLORER
    # ============================================================

    st.subheader("⚠ 8. Failure Mode Explorer")

    fig, ax = plt.subplots(figsize=(6, 4))
    df_full["Failure_mode"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Failure Mode Distribution")
    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # FEATURE 9 — DISTRIBUTION PLOTS FOR EACH VARIABLE
    # ============================================================

    st.subheader("📦 9. Variable Distributions")

    dist_cols = ["wu_FEA", "H", "bf", "tw", "tf", "h0", "s", "s0", "se", "L", "fy"]

    fig, axes = plt.subplots(len(dist_cols) // 3 + 1, 3, figsize=(15, 15))

    for ax, col in zip(axes.flatten(), dist_cols):
        ax.hist(df_full[col], bins=20, alpha=0.7)
        ax.set_title(col)

    st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # FEATURE 10 — OUTLIER DETECTION TOOL
    # ============================================================

    st.subheader("🚨 10. Outlier Detection")

    wu_mean = df_full["wu_FEA"].mean()
    wu_std  = df_full["wu_FEA"].std()

    outliers = df_full[
        (df_full["wu_FEA"] > wu_mean + 3 * wu_std) |
        (df_full["wu_FEA"] < wu_mean - 3 * wu_std)
    ]

    st.write(f"Outliers detected: **{len(outliers)}**")
    st.dataframe(outliers)

    st.write("""
    Outliers are not errors — they may represent extreme geometries or uncommon failure 
    mechanisms. Engineers should examine these cases for design insights.
    """)

