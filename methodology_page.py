# ============================================================
# methodology_page.py — Full Methodology Description
# ============================================================

import streamlit as st

def render():

    st.header("🧬 Methodology — 3-Stage Inverse Design Framework")

    st.write("""
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

    st.write("""
    The full design engine consists of three coordinated stages:

    **Stage 1 — Inverse Model (Classification)**  
    Predicts the *most suitable steel section* for a target strength input.

    **Stage 2 — Surrogate Forward Prediction (Regression, Quantile Models)**  
    Computes **Predicted_wu**, **LowerBound_wu**, and **UpperBound_wu** for each candidate.

    **Stage 3 — Structural & Code-Based Evaluation**  
    Applies simplified versions of:  
    • SCI design checks  
    • EN1993-1-13 (ENM) opening rules  
    • AISC rules for web-post stability  
    • Weight comparison  
    • Failure-mode filtering  
    • Multi-objective score ranking

    This guarantees that recommendations are both:
    - **numerically accurate** (ML-based), and
    - **mechanically meaningful** (code-aware).
    """)

    st.markdown("### 🔄 High-Level Flow")
    st.code("""
User Inputs → Stage 1 → Top Sections → Stage 2 → Strength Checks →
Structural Evaluation → Ranking → Final Recommendation
""")

    st.markdown("---")

    # ============================================================
    # 2 — STAGE 1: INVERSE MODEL
    # ============================================================
    st.subheader("🎯 2. Stage 1 — Inverse Prediction Model (Classification)")

    st.write("""
    The inverse model is a **CatBoost multiclass classifier** trained to map:

    **Inputs:**  
    - Target strength (wu)
    - Span (L)
    - Opening diameter (h0)
    - Spacing (s)
    - Steel grade (fy)

    **Output:**  
    - Probabilities over all candidate steel sections (≈200+ classes)

    The model produces the **Top-10 most probable sections**, which are then
    passed to Stage 2.  

    This drastically reduces search space while keeping high accuracy.
    """)

    st.markdown("---")

    # ============================================================
    # 3 — STAGE 2: FORWARD SURROGATE
    # ============================================================
    st.subheader("📈 3. Stage 2 — Forward Surrogate Modelling (P50 / P10 / P90)")

    st.write("""
    For each candidate section from Stage 1, the tool evaluates the predicted 
    resistance using three independent gradient-boosted regression models:

    - **P50 model:** median resistance  
    - **P10 model:** lower-bound safety estimate  
    - **P90 model:** upper-bound estimate  

    **Inputs to the surrogate models:**
    - H, bf, tw, tf  
    - L  
    - h0, s  
    - Derived: s0, se  
    - fy  

    These predictions form the basis for feasibility validation and
    code-based evaluation in Stage 3.
    """)

    st.markdown("### Quantile-based Safety Interpretation")
    st.write("""
    - **If P10 > target wu:** high reliability  
    - **If P50 ≈ target wu:** strong match  
    - **If P90 < target wu:** target is physically impossible  
    """)

    st.markdown("---")

    # ============================================================
    # 4 — STAGE 3: STRUCTURAL EVALUATION
    # ============================================================
    st.subheader("🏗 4. Stage 3 — Structural & Code Evaluation")

    st.write("""
    Each of the Top-10 sections is evaluated using simplified 
    structural checks:

    **✔ SCI rules**  
    – Web-post stability  
    – Vierendeel bending  
    – Hole interaction limits  

    **✔ EN 1993-1-13 (ENM rules)**  
    – Opening geometry  
    – Plate slenderness  
    – Spacing and ligament checks  

    **✔ AISC rules**  
    – Web-post shear  
    – Web crippling  

    **✔ Weight calculation**  
    Used in multi-objective ranking.

    **✔ Failure mode reference**  
    From the dataset labels.

    Each section receives a **multi-objective score** combining:
    - Strength matching  
    - Code compliance  
    - Weight  
    - Failure mode favourability  
    """)

    st.markdown("### Ranking")
    st.write("""
    The final output includes:

    • **Exact strength-matching section**  
    • **Optimal balanced section**  
    • **Full ranked table with bounds and code checks**
    """)

    st.markdown("---")

    # ============================================================
    # 5 — FEASIBILITY & VALIDATION
    # ============================================================
    st.subheader("🧭 5. Feasibility & Physical-Domain Validation")

    st.write("""
    Before any design is accepted, the model validates:

    **1. Geometry feasibility**
    - Opening count N0  
    - Edge distances se  
    - Clear spacing s0  

    **2. Dataset domain validity**
    Using the Global Validity Map:
    - Safe  
    - Caution (±10% from limits)  
    - Outside dataset  

    **3. Achievable strength range**
    If the user's target is **higher than any achievable section**, 
    the system produces an engineering recommendation.

    Likewise, if the target is **too low**, the system warns about 
    unnecessary oversizing.
    """)

    st.markdown("---")

    # ============================================================
    # 6 — DATASET FOUNDATION
    # ============================================================
    st.subheader("📚 6. Dataset Basis")

    st.write("""
    The backbone of the methodology is a large high-fidelity dataset containing:

    - Geometric parameters  
    - Tensile properties (fy)  
    - Derived opening metrics  
    - FEA-computed ultimate resistance (wu_FEA)  
    - Failure modes  
    - Code applicability flags  

    The surrogate and inverse models were trained purely on this dataset,
    ensuring data-driven consistency across the entire methodology.
    """)

    st.markdown("---")

    # ============================================================
    # 7 — VISUAL PIPELINE (ASCII DIAGRAM)
    # ============================================================
    st.subheader("📐 7. Pipeline Diagram (ASCII)")

    st.code("""
User Inputs
     │
     ▼
[ Stage 1: Inverse Classifier ]
     │  → Top-10 sections
     ▼
[ Stage 2: Forward Surrogates (P10 / P50 / P90) ]
     │  → Predicted strengths + bounds
     ▼
[ Stage 3: Structural Filters + Code Checks ]
     │  → Score / Ranking
     ▼
Final Recommended Design
""")

    st.markdown("---")

    # ============================================================
    # 8 — FINAL NOTES
    # ============================================================
    st.subheader("📝 8. Notes & Limitations")

    st.write("""
    • The surrogate cannot predict outside the dataset domain.  
    • Code checks are simplified but consistent with SCI/EN/AISC logic.  
    • FEA data governs reliability — predictions inherit its limits.  
    • Quantile models provide realistic uncertainty bounds.  
    """)

    st.success("✔ Methodology summary loaded successfully.")
