# ============================================================
# stage1_functions.py — Helper Functions (Deterministic Version)
# ============================================================

import numpy as np

# ------------------------------------------------------------
# 1) RULE-BASED CODE CHECKS (UNCHANGED, still available if needed)
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
# 2) WEIGHT FUNCTION (UNCHANGED)
# ------------------------------------------------------------

def compute_weight(H, bf, tw, tf, L, density=7850/1e9):
    A = 2*(bf*tf) + tw*(H - 2*tf)
    return A * L * density


# ------------------------------------------------------------
# 3) MULTIOBJECTIVE SCORE (DETERMINISTIC VERSION)
# ------------------------------------------------------------
def multiobjective_score(wu_target, wu_pred, weight,
                         sci, enm, aisc, failure_mode):
    """
    Deterministic scoring:
      N/A = -1 → ignored (penalty = 0)
      Pass = +1 → penalty 0
      Fail = 0 → penalty +1

    Failure penalty + strength penalty unchanged.
    """

    # Strength penalty
    strength_penalty = abs(wu_pred - wu_target) / max(wu_target, 1e-6)

    # Applicability penalty (N/A ignored)
    def code_pen(val):
        if val == -1:
            return 0      # N/A → ignore
        return 1 - val    # Fail=1, Pass=0

    code_penalty = (
        code_pen(sci) +
        code_pen(enm) +
        code_pen(aisc)
    )

    # Failure mode penalty (same as your original logic)
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
# 4) RULE-BASED EMOJI (kept for debugging)
# ------------------------------------------------------------
def code_to_emoji(val):
    if val == 1:
        return "✔️"
    if val == 0:
        return "❌"
    return "➖"


# ------------------------------------------------------------
# 5) APPLICABILITY LABEL (for internal use if needed)
# ------------------------------------------------------------
def applicability_to_label(val):
    """
    Deterministic text label:
        -1 → N/A
         1 → Pass
         0 → Fail
    """
    if val == -1:
        return "N/A"
    elif val == 1:
        return "Pass"
    elif val == 0:
        return "Fail"
    return "N/A"


# ------------------------------------------------------------
# 6) APPLICABILITY EMOJI (USED IN GUI)
# ------------------------------------------------------------
def applicability_to_emoji(val):
    """
    Final GUI representation:
        -1 → ⬜
         1 → 🟩
         0 → 🟥
    """
    if val == -1:
        return "⬜"
    elif val == 1:
        return "🟩"
    elif val == 0:
        return "🟥"
    return "⬜"
