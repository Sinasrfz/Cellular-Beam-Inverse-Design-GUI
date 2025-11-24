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
    LOWER SCORE = BETTER.
    Strength → dominant
    Weight → moderate
    Code fail → very large penalty
    Failure mode → small penalty
    """

    # --- 1. Strength accuracy (dominant)
    error_ratio = abs(wu_pred - wu_target) / wu_target
    f_strength = error_ratio * 20   # increased weight

    # --- 2. Weight (now stronger)
    f_weight = weight / 500         # was 1000 → now weight matters 2× more

    # --- 3. Code penalties (very large)
    penalty_code = 0
    if sci == 0: penalty_code += 100
    if en == 0:  penalty_code += 100
    if aisc == 0: penalty_code += 100

    # --- 4. Failure mode penalties
    # better modes = lower score
    failure_priority = {
        "BGB": 0,
        "WPL": 2,
        "WPS": 4,
        "LGB": 6,
        "WPB": 8,
        "Unknown": 10
    }
    f_failure = failure_priority.get(failure_mode, 10)

    # --- final combined score
    score = f_strength + f_weight + penalty_code + f_failure
    return score


