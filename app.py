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
# GOOGLE DRIVE DIRECT DOWNLOAD URL FOR INVERSE MODEL
# ============================================================

INVERSE_MODEL_URL = (
    "https://drive.google.com/uc?export=download&id=1_ETS6bVRiKq8YuwWRLOMwEREYcxblM6k"
)

def load_joblib_from_drive(url):
    """Download .joblib file from a Google Drive direct link."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return joblib.load(io.BytesIO(response.content))
    except Exception as e:
        st.error("❌ Failed to load model from Google Drive.")
        st.code(str(e))
        raise e


# ============================================================
# LOCAL FILE PATHS (GitHub-hosted)
# ============================================================

FWD_P50       = "forward_p50.joblib"
FWD_P10       = "forward_p10.joblib"
FWD_P90       = "forward_p90.joblib"
SECTION_LOOKUP = "section_lookup.csv"
DATA_FILE     = "21.xlsx"   # full dataset

# ============================================================
# CACHE LOADERS
# ============================================================

@st.cache_resource
def load_inverse():
    return load_joblib_from_drive(INVERSE_MODEL_URL)

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
    inv_model = load_inverse()   # Load from Google Drive
    fwd_p50, fwd_p10, fwd_p90 = load_forward()  # Load from GitHub
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
    designer_page.render(inv_model, fwd_p50, fwd_p10, fwd_p90, section_lookup, df_full)

elif page == "📊 Diagnostics":
    import diagnostics_page
    diagnostics_page.render()

elif page == "📁 Dataset Explorer":
    import explorer_page
    explorer_page.render(df_full)

elif page == "📘 Methodology":
    import methodology_page
    methodology_page.render()
