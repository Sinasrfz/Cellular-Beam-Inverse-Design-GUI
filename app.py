# ============================================================
# app.py — MAIN STREAMLIT APPLICATION (3-Stage Pipeline Version)
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import requests
import io

# Streamlit Layout
st.set_page_config(page_title="Cellular Beam Inverse Design Tool", layout="wide")
st.title("🧠 Cellular Beam Inverse Design Tool")

# ============================================================
# GITHUB RELEASES DOWNLOAD URL FOR INVERSE MODEL
# ============================================================

INVERSE_MODEL_URL = (
    "https://github.com/Sinasrfz/Cellular-Beam-Inverse-Design-GUI/"
    "releases/download/v1.0/inverse_model_catboost.joblib"
)

def load_joblib_from_url(url):
    """Download .joblib file from GitHub Releases or other direct link."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        file_obj = io.BytesIO(response.content)
        return joblib.load(file_obj)
    except Exception as e:
        st.error("❌ Failed to load model from URL.")
        st.code(str(e))
        raise e

# ============================================================
# LOCAL FILE PATHS (GitHub-hosted)
# ============================================================

FWD_P50        = "forward_p50.joblib"
FWD_P10        = "forward_p10.joblib"
FWD_P90        = "forward_p90.joblib"
SECTION_LOOKUP = "section_lookup.csv"
DATA_FILE      = "21.xlsx"   # full dataset

# ============================================================
# CACHE LOADERS
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
    return pd.read_csv(SECTION_LOOKUP)

@st.cache_resource
def load_full_data():
    return pd.read_excel(DATA_FILE)

# ============================================================
# INITIAL LOAD
# ============================================================

try:
    inv_model = load_inverse()    # Load the heavy inverse model from GitHub Releases
    fwd_p50, fwd_p10, fwd_p90 = load_forward()
    section_lookup = load_lookup()
    df_full = load_full_data()
    st.success("✔ All models and data loaded successfully.")
except Exception as e:
    st.error("❌ Failed to load required assets.")
    st.code(str(e))
    st.stop()

# ============================================================
# SIDEBAR NAVIGATION
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
# PAGE ROUTING
# ============================================================

if page == "🏗 Designer Tool":
    import designer_page
    designer_page.render(
        inv_model, fwd_p50, fwd_p10, fwd_p90,
        section_lookup, df_full
    )

elif page == "📊 Diagnostics":
    import diagnostics_page
    diagnostics_page.render()

elif page == "📁 Dataset Explorer":
    import explorer_page
    explorer_page.render(df_full)

elif page == "📘 Methodology":
    import methodology_page
    methodology_page.render()
