# ============================================================
# Stage 1 – Utility Functions for Inverse Design GUI
# ============================================================

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# ⚠️ REMOVED: CODE CHECK FUNCTIONS
# These are no longer used because the GUI now uses dataset
# applicability values directly. Keeping them would cause
# inconsistency, so they are intentionally removed.
# ------------------------------------------------------------

# def check_SCI(...):
# def check_ENM(...):
# def check_AISC(...):
# (Removed intentionally — dataset values are used instead)


# ------------------------------------------------------------
# WEIGHT FUNCTION  (unchanged)
# ------------------------------------------------------------

def compute_weight(H, bf, tw, tf, L, density=7850/1e9):
    A = 2 * (bf * tf) + tw * (H - 2 * tf)
    return A * L * density


# ------------------------------------------------------------
# EMOJI CONVERSION (unchanged)
# ------------------------------------------------------------

def code_to_emoji(val):
    if val == 1:
        return "🟩 PASS"
    if val == 0:
        return "🟥 FAIL"
    if val == -1:
        return "⚪ N/A"
    return "⚪ N/A"


# ------------------------------------------------------------
# MULTI-OBJECTIVE SCORE (unchanged)
# ------------------------------------------------------------

def multiobjective_score(wu_target, wu_pred, weight, sci, enm, aisc, failure_mode):

    # Strength mismatch
    error_ratio = abs(wu_pred - wu_target) / wu_target
    f_strength = error_ratio * 20

    # Weight
    f_weight = weight / 500

    # Code penalties (ONLY FAIL = +100, N/A = 0)
    penalty_code = 0
    if sci == 0: penalty_code += 100
    if enm == 0: penalty_code += 100
    if aisc == 0: penalty_code += 100

    # Failure mode priority
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
