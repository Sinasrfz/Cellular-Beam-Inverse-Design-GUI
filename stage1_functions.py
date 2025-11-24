# ============================================================
# Stage 1 – Utility Functions for Inverse Design GUI
# (Extracted directly from your scientific Stage-1 pipeline)
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
# MULTI-OBJECTIVE SCORE
# ------------------------------------------------------------

def multiobjective_score(wu_target, wu_pred, weight, sci, en, aisc, failure_mode):
    strength_penalty = abs(wu_pred - wu_target) / wu_target
    code_penalty = (1 - sci) + (1 - en) + (1 - aisc)

    # Failure mode penalty exactly as your Stage-1 logic
    failure_penalty = 0
    if failure_mode == "WPB":
        failure_penalty = 3
    elif failure_mode in ["WPS", "VBT"]:
        failure_penalty = 1

    return (
        2.0 * strength_penalty +
        1.5 * code_penalty +
        0.5 * (weight / 200.0) +
        1.5 * failure_penalty
    )
