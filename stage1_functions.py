# ============================================================
# Stage 1 – Utility Functions for Inverse Design GUI
# (Corrected scoring, stable ranking)
# ============================================================

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# CODE CHECK FUNCTIONS
# ------------------------------------------------------------

def check_SCI(H, bf, tw, tf, h0, s0, se):
    h = H + tf
    hT = (h - h0) / 2
    return int(h0 <= 0.8*h and hT >= tf + 30 and s0 >= 0.3*h0 and se >= 0.5*h0)

def check_ENM(H, bf, tw, tf, h0, s0):
    h = H + tf
    hT = (h - h0) / 2
    return int(h0 <= 0.8*h and hT >= tf + 30 and s0 >= 0.1*h0)

def check_AISC(H, bf, tw, tf, h0, s):
    return int(1.25 <= H/h0 <= 1.75 and 1.08 <= s/h0 <= 1.50)

# ------------------------------------------------------------
# WEIGHT FUNCTION
# ------------------------------------------------------------

def compute_weight(H, bf, tw, tf, L, density=7850/1e9):
    A = 2*(bf*tf) + tw*(H - 2*tf)
    return A * L * density

# ------------------------------------------------------------
# UPDATED MULTI-OBJECTIVE SCORE  (MUCH BETTER)
# ------------------------------------------------------------

def multiobjective_score(wu_target, wu_pred, weight, sci, en, aisc, failure_mode):
    """
    LOWER SCORE = BETTER DESIGN.
    Dominant: strength match.
    Secondary: weight, failure mode.
    Hard penalty: code failures.
    """

    # --- 1. Strength accuracy (dominant factor)
    error_ratio = abs(wu_pred - wu_target) / wu_target
    f_strength = error_ratio * 10  # strong weighting

    # --- 2. Weight term
    f_weight = weight / 1000       # scale to ~0.6–1.3

    # --- 3. Code safety
    penalty_code = 0
    if sci == 0: penalty_code += 50
    if en == 0:  penalty_code += 50
    if aisc == 0: penalty_code += 50

    # --- 4. Failure mode priority
    failure_priority = {
        "BGB": 0,
        "WPL": 1,
        "WPS": 2,
        "LGB": 3,
        "WPB": 4,
        "Unknown": 5
    }
    f_failure = failure_priority.get(failure_mode, 5)

    # FINAL SCORE
    score = f_strength + f_weight + penalty_code + f_failure

    return score
