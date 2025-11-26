# ============================================================
# stage1_functions.py — Helper Functions (Final ML + Emoji Version)
# ============================================================

import numpy as np

# ------------------------------------------------------------
# 1) RULE-BASED CODE CHECKS (UNCHANGED)
# ------------------------------------------------------------

def check_SCI(H, bf, tw, tf, h0, s0, se):
    """SCI geometric applicability check. (original formulas)"""
    h = H + tf
    hT = (h - h0) / 2
    return int(h0 <= 0.8*h and hT >= tf + 30 and s0 >= 0.3*h0 and se >= 0.5*h0)


def check_ENM(H, bf, tw, tf, h0, s0):
    """ENM geometric applicability check. (original formulas)"""
    h = H + tf
    hT = (h - h0) / 2
    return int(h0 <= 0.8*h and hT >= tf + 30 and s0 >= 0.1*h0)


def check_AISC(H, bf, tw, tf, h0, s):
    """AISC geometric applicability check. (original formulas)"""
    return int(1.25 <= H/h0 <= 1.75 and 1.08 <= s/h0 <= 1.50)


# ------------------------------------------------------------
# 2) WEIGHT FUNCTION (UNCHANGED)
# ------------------------------------------------------------

def compute_weight(H, bf, tw, tf, L, density=7850/1e9):
    """Compute steel weight in kg in mm-unit system."""
    A = 2*(bf*tf) + tw*(H - 2*tf)
    return A * L * density


# ------------------------------------------------------------
# 3) MULTIOBJECTIVE SCORE  — UPDATED FOR N/A HANDLING
# ------------------------------------------------------------

def multiobjective_score(wu_target, wu_pred, weight,
                         sci, enm, aisc, failure_mode):
    """
    ✔ If sci/enm/aisc = -1 (N/A), it contributes *zero penalty*.
    ✔ If sci/enm/aisc = 0  (Fail), it adds penalty +1.
    ✔ If sci/enm/aisc = 1  (Pass), it adds 0 penalty.

    Failure penalty & strength penalty remain unchanged.
    """

    # Strength penalty
    strength_penalty = abs(wu_pred - wu_target) / max(wu_target, 1e-6)

    # Applicability penalty (N/A ignored)
    def code_pen(x):
        if x == -1:  # N/A
            return 0
        return 1 - x  # Fail=1, Pass=0

    code_penalty = code_pen(sci) + code_pen(enm) + code_pen(aisc)

    # Failure mode penalty (unchanged)
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


# ------------------------------------------------------------
# 4) RULE-BASED EMOJI (kept for debugging only)
# ------------------------------------------------------------

def code_to_emoji(val):
    if val == 1:
        return "✔️"
    if val == 0:
        return "❌"
    return "➖"


# ------------------------------------------------------------
# 5) ML APPLICABILITY LABELS (text) — for internal use
# ------------------------------------------------------------

def applicability_to_label(val):
    if val == -1:
        return "N/A"
    elif val == 1:
        return "Pass"
    elif val == 0:
        return "Fail"
    return "N/A"


# ------------------------------------------------------------
# 6) ML APPLICABILITY — FINAL EMOJI OUTPUT (USED IN GUI)
# ------------------------------------------------------------

def applicability_to_emoji(val):
    """
    Emoji-based ML applicability (final output for GUI):
        -1 -> ⬜ N/A
         1 -> 🟩 Pass
         0 -> 🟥 Fail
    """
    if val == -1:
        return "⬜ N/A"
    elif val == 1:
        return "🟩 Pass"
    elif val == 0:
        return "🟥 Fail"
    return "⬜ N/A"
