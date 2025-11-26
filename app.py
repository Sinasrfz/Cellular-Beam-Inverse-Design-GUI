# ============================================================
# app.py — MAIN STREAMLIT APPLICATION (3-Stage Pipeline Version)
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import requests
import io

# ------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Cellular Beam Inverse Design Tool",
    layout="wide"
)
st.title("🧠 Cellular Beam Inverse Design Tool")


# ============================================================
# 1 — INVERSE MODEL (GitHub Releases URL)
# ============================================================

INVERSE_MODEL_URL = (
    "https://github.com/Sinasrfz/Cellular-Beam-Inverse-Design-GUI/"
    "releases/download/v1.0/inverse_model_catboost.joblib"
)

def load_joblib_from_url(url):
    """Download .joblib file from GitHub Releases."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return joblib.load(io.BytesIO(response.content))
    except Exception as e:
        st.error("❌ Failed to load inverse model.")
        st.code(str(e))
        raise e


# ============================================================
# 2 — LOCAL FILE PATHS (stored in GitHub repo root)
# ============================================================

FWD_P50        = "forward_p50.joblib"
FWD_P10        = "forward_p10.joblib"
FWD_P90        = "forward_p90.joblib"
SECTION_LOOKUP = "section_lookup.csv"
DATA_FILE      = "21.xlsx"


# ============================================================
# 3 — CACHE LOADERS
# ============================================================

@st.cache_resource
def load_inverse():
    return load_joblib_from_url(INVERSE_MODEL_URL)

@st.cache_resource
def load_forward():
    return (
        joblib.load(FWD_P50),
        joblib.load(FWD_P10),
        joblib.load(FWD_P90)
    )

@st.cache_resource
def load_lookup():
    df = pd.read_csv(SECTION_LOOKUP)

    df.columns = (
        df.columns
        .str.replace(" ", "")
        .str.replace(",", "")
        .str.replace("×", "x")
        .str.strip()
    )
    return df

@st.cache_resource
def load_full_data():
    df = pd.read_excel(DATA_FILE)

    df.columns = (
        df.columns
        .str.replace(" ", "")
        .str.replace(",", "")
        .str.replace("×", "x")
        .str.strip()
    )
    return df


# ============================================================
# 4 — INITIAL LOAD
# ============================================================

try:
    inv_model = load_inverse()
    fwd_p50, fwd_p10, fwd_p90 = load_forward()
    section_lookup = load_lookup()
    df_full = load_full_data()

    if "SectionID" not in df_full.columns:
        st.error("❌ ERROR: Your dataset does NOT contain SectionID.")
        st.stop()

    st.success("✔ All models and data loaded successfully.")

    # Store for use in pages
    st.session_state["inv_model"] = inv_model
    st.session_state["fwd_p50"] = fwd_p50
    st.session_state["fwd_p10"] = fwd_p10
    st.session_state["fwd_p90"] = fwd_p90
    st.session_state["section_lookup"] = section_lookup
    st.session_state["df_full"] = df_full

except Exception as e:
    st.error("❌ Failed to load required assets.")
    st.code(str(e))
    st.stop()


# ============================================================
# 5 — SIDEBAR NAVIGATION
# ============================================================

st.sidebar.header("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "🏗 Designer Tool",
        "📊 Diagnostics",
        "📁 Dataset Explorer",
        "📘 Methodology"
    ]
)


# ============================================================
# 6 — PAGE ROUTING
# ============================================================

if page == "🏗 Designer Tool":
    import designer_page
    designer_page.render(
        st.session_state["inv_model"],
        st.session_state["fwd_p50"],
        st.session_state["fwd_p10"],
        st.session_state["fwd_p90"],
        st.session_state["section_lookup"],
        st.session_state["df_full"]
    )

elif page == "📊 Diagnostics":
    import diagnostics_page
    diagnostics_page.render()

elif page == "📁 Dataset Explorer":
    import explorer_page
    explorer_page.render(st.session_state["df_full"])

elif page == "📘 Methodology":
    import methodology_page
    methodology_page.render()
