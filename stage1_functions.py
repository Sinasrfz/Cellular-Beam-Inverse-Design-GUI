# ============================================================
# Stage 1 – Utility Functions for Inverse Design GUI
# ============================================================

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# CODE CHECK FUNCTIONS
# ------------------------------------------------------------
def check_SCI(H, bf, tw, tf, h0, s0, se):
    """SCI limit checks."""
    h = H + tf
    hT = (h - h0) / 2

    return int(
        h0 <= 0.8 * h and
        hT >= tf + 30 and
        s0 >= 0.3 * h0 and
        se >= 0.5 * h0
    )


def check_ENM(H, bf, tw, tf, h0, s0):
    """EN1993-1-13 limit checks."""
    h = H + tf
    hT = (h - h0) / 2

    return int(
        h0 <= 0.8 * h and
        hT >= tf + 30 and
        s0 >= 0.1 * h0
    )


def check_AISC(H, bf, tw, tf, h0, s):
    """AISC web-post and spacing limits."""
    return int(
        1.25 <= H / h0 <= 1.75 and
        1.08 <= s / h0 <= 1.50
    )


# ------------------------------------------------------------
# WEIGHT FUNCTION
# ------------------------------------------------------------
def compute_weight(H, bf, tw, tf, L, density=7850 / 1e9):
    """
    Compute steel weight (kg) from cross-section geometry.
    Area (mm²) × length (mm) × density (kg/mm³).
    """
    A = 2 * (bf * tf) + tw * (H - 2 * tf)
    return A * L * density


# ------------------------------------------------------------
# EMOJI CONVERSION
# ------------------------------------------------------------
def code_to_emoji(val):
    """Convert applicability (1,0,-1) into emoji string."""
    if val == 1:
        return "🟩 PASS"
    if val == 0:
        return "🟥 FAIL"
    if val == -1:
        return "⚪ N/A"
    return "⚪ N/A"


# ------------------------------------------------------------
# MULTI-OBJECTIVE SCORE (Applicability-aware)
# ------------------------------------------------------------
def multiobjective_score(wu_target, wu_pred, weight,
                         sci, enm, aisc, failure_mode):
    """
    Score = strength mismatch + weight + code penalties + failure priority.
    Lower score = better.
    """

    # Strength mismatch
    error_ratio = abs(wu_pred - wu_target) / wu_target
    f_strength = error_ratio * 20

    # Weight contribution
    f_weight = weight / 500

    # Apply code penalties (FAIL = +100, N/A = 0)
    penalty_code = 0
    if sci == 0:
        penalty_code += 100
    if enm == 0:
        penalty_code += 100
    if aisc == 0:
        penalty_code += 100

    # Failure mode ranking
    failure_priority = {
        "BGB": 0,
        "WPL": 2,
        "WPS": 4,
        "LGB": 6,
        "WPB": 8,
        "Unknown": 10
    }

    f_failure = failure_priority.get(failure_mode, 10)

    return f_strength + f_weight + penalty_code + f_failure
