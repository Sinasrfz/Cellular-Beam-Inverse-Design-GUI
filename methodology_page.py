# ============================================================
# methodology_page.py — Full Methodology Description (Safari-Safe)
# ============================================================

import streamlit as st

def safe_markdown(text):
    """Render markdown without Safari/Regex autolink crashes."""
    st.markdown(f"<div style='white-space:pre-wrap'>{text}</div>", unsafe_allow_html=True)

def render():

    st.header("🧬 Methodology — 3-Stage Inverse Design Framework")

    safe_markdown("""
This page explains the complete methodology used in this tool.
The workflow combines inverse classification, forward surrogate modelling,
and multi-criteria structural filtering to identify feasible and
code-compliant cellular beam designs.
""")

    st.markdown("---")

    # ============================================================
    # 1 — OVERVIEW
    # ============================================================
    st.subheader("📌 1. Overview of the Complete Pipeline")

    safe_markdown("""
The full design engine consists of three coordinated stages:

Stage 1 — Inverse Model (Classification)
Predicts the most suitable steel section for a target strength input.

Stage 2 — Surrogate Forward Prediction (Regression, Quantile Models)
Computes Predicted_wu, LowerBound_wu, and UpperBound_wu for each candidate.

Stage 3 — Structural & Code-Based Evaluation
Applies simplified versions of SCI rules, EN1993-1-13 rules, AISC rules,
weight comparisons, failure mode filtering, and multi-objective scoring.

This ensures recommendations are both numerically accurate and mechanically meaningful.
""")

    st.markdown("### 🔄 High-Level Flow")
    safe_markdown("""
User Inputs → Stage 1 → Top Sections → Stage 2 → Strength Checks →
Structural Evaluation → Ranking → Final Recommendation
""")

    st.markdown("---")

    # ============================================================
    # 2 — STAGE 1
    # ============================================================
    st.subheader("🎯 2. Stage 1 — Inverse Prediction Model (Classification)")

    safe_markdown("""
The inverse model is a CatBoost multiclass classifier trained to map:

Inputs:
- Target strength (wu)
- Span (L)
- Opening diameter (h0)
- Spacing (s)
- Steel grade (fy)

Output:
- Probabilities over all candidate steel sections (200+ classes)

The model returns the Top-10 most probable sections, reducing search space
while keeping very high accuracy.
""")

    st.markdown("---")

    # ============================================================
    # 3 — STAGE 2
    # ============================================================
    st.subheader("📈 3. Stage 2 — Forward Surrogate Modelling (P50 / P10 / P90)")

    safe_markdown("""
Each candidate from Stage 1 is evaluated using three regression models:

P50 → median resistance  
P10 → safe lower-bound  
P90 → upper bound  

Inputs to regression:
- H, bf, tw, tf
- L
- h0, s
- Derived: s0, se
- fy

Quantile interpretation:
- If P10 > target → high reliability
- If P50 ≈ target → strong match
- If P90 < target → target is physically impossible
""")

    st.markdown("---")

    # ============================================================
    # 4 — STAGE 3
    # ============================================================
    st.subheader("🏗 4. Stage 3 — Structural & Code Evaluation")

    safe_markdown("""
Each Top-10 section is filtered through simplified:

SCI rules
EN 1993-1-13 rules
AISC rules
Weight calculation
Failure mode reference

A combined score is generated from:
- Strength matching
- Code compliance
- Weight
- Failure mode
""")

    st.markdown("### Ranking")
    safe_markdown("""
Outputs include:
- Exact strength-matching section
- Best balanced optimal section
- Full ranking table
""")

    st.markdown("---")

    # ============================================================
    # 5 — FEASIBILITY
    # ============================================================
    st.subheader("🧭 5. Feasibility & Validation")

    safe_markdown("""
Validation includes:

1. Geometry feasibility  
   - N0  
   - s0  
   - se  

2. Dataset domain via Global Validity Map  
   - Safe  
   - Caution  
   - Outside  

3. Achievable strength range  
   - If target > max achievable → raise engineering warning  
""")

    st.markdown("---")

    # ============================================================
    # 6 — DATASET
    # ============================================================
    st.subheader("📚 6. Dataset Basis")

    safe_markdown("""
Dataset includes:
- Geometry
- fy
- Derived opening metrics
- FEA resistance (wu_FEA)
- Failure modes
- Code applicability

All models are trained directly from this dataset.
""")

    st.markdown("---")

    # ============================================================
    # 7 — ASCII PIPELINE
    # ============================================================
    st.subheader("📐 7. Pipeline Diagram (ASCII)")

    safe_markdown("""
User Inputs
     │
     ▼
[ Stage 1: Inverse Classifier ]
     │
     ▼
[ Stage 2: Forward Surrogates (P10 / P50 / P90) ]
     │
     ▼
[ Stage 3: Structural Filters + Code Checks ]
     │
     ▼
Final Recommended Design
""")

    st.markdown("---")

    # ============================================================
    # 8 — NOTES
    # ============================================================
    st.subheader("📝 8. Notes & Limitations")

    safe_markdown("""
- Surrogate does not extrapolate beyond dataset  
- Code checks simplified  
- FEA data governs reliability  
- Quantile uncertainties realistic  
""")

    st.success("✔ Methodology summary loaded successfully.")
