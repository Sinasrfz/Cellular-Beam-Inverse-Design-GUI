# ============================================================
# stage1_functions.py — Helper Functions (Updated, Non-Breaking)
# ============================================================

import numpy as np

# ------------------------------------------------------------
# RULE-BASED CODE CHECKS (UNCHANGED)
# ------------------------------------------------------------

def check_SCI(H, bf, tw, tf, h0, s0, se):
    """
    SCI geometric applicability check.
    These remain EXACTLY as your original formulas.
    """
    h = H + tf
    hT = (h - h0) / 2
    return int(h0 <= 0.8*h and hT >= tf + 30 and s0 >= 0.3*h0 and se >= 0.5*h0)


def check_ENM(H, bf, tw, tf, h0, s0):
    """
    ENM geometric applicability check.
    Unchanged.
    """
    h = H + tf
    hT = (h - h0) / 2
    return int(h0 <= 0.8*h and hT >= tf + 30 and s0 >= 0.1*h0)


def check_AISC(H, bf, tw, tf, h0, s):
    """
    AISC geometric applicability check.
    Unchanged.
    """
    return int(1.25 <= H/h0 <= 1.75 and 1.08 <= s/h0 <= 1.50)


# ------------------------------------------------------------
# WEIGHT FUNCTION (UNCHANGED)
# ------------------------------------------------------------

def compute_weight(H, bf, tw, tf, L, density=7850/1e9):
    """
    Compute steel beam weight using mm units.
    Unchanged.
    """
    A = 2*(bf*tf) + tw*(H - 2*tf)
    return A * L * density


# ------------------------------------------------------------
# MULTIOBJECTIVE SCORE (UPDATED BUT NON-BREAKING)
# ------------------------------------------------------------

def multiobjective_score(wu_target, wu_pred, weight,
                         sci, enm, aisc, failure_mode):
    """
    Scoring function used to rank candidate sections.

    ❗ Your original scoring logic is 100% preserved.
    ❗ Only ML inputs are now passed from designer_page.py,
       so this function needs NO modifications in structure.
    """

    strength_penalty = abs(wu_pred - wu_target) / max(wu_target, 1e-6)

    code_penalty = (1 - sci) + (1 - enm) + (1 - aisc)

    # failure mode weight
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
# EMOJI MAPPING FOR RULE-BASED CHECKS (UNCHANGED)
# ------------------------------------------------------------

def code_to_emoji(val):
    """
    Convert 1/0/-1 into emoji for GUI display.
    Unchanged.
    """
    if val == 1:
        return "✔️"
    if val == 0:
        return "❌"
    return "➖"    # Not applicable or unknown


# ------------------------------------------------------------
# NEW — ML EMOJI MAPPING (ADDED, NON-BREAKING)
# ------------------------------------------------------------

def ml_to_emoji(val):
    """
    New helper used in designer_page.py if needed.
    Does NOT replace old function.
    """
    return "✔️" if val == 1 else "❌"
