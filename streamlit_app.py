import os
import sys
import json
import io
from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st

# Windows MSVC runtime DLL path helper (if running locally on Windows)
_msvc_dir = r"C:\Program Files\Microsoft Office\root\Client"
if os.path.exists(_msvc_dir) and _msvc_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _msvc_dir + ";" + os.environ.get("PATH", "")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Workspace Directories
WORKSPACE_ROOT = Path(__file__).resolve().parent

_candidate_model_dirs = [
    Path(os.environ["MODELS_DIR"]) if "MODELS_DIR" in os.environ else None,
    WORKSPACE_ROOT / "models",
    WORKSPACE_ROOT / "backend" / "models",
    Path("models").resolve(),
    Path("/app/models")
]
MODELS_DIR = next((d for d in _candidate_model_dirs if d and d.exists()), WORKSPACE_ROOT / "models")

DISEASE_MODEL_PATH = MODELS_DIR / "potato_model.keras"
VALIDATOR_MODEL_PATH = MODELS_DIR / "potato_validator.keras"
CLASS_MAPPING_PATH = MODELS_DIR / "class_mapping.json"
VALIDATOR_CONFIG_PATH = MODELS_DIR / "validator_config.json"
KNOWLEDGE_BASE_PATH = WORKSPACE_ROOT / "backend" / "knowledge_base.json"

_candidate_sample_dirs = [
    WORKSPACE_ROOT / "samples",
    WORKSPACE_ROOT / "frontend" / "public" / "samples",
    Path("samples").resolve()
]
SAMPLES_DIR = next((d for d in _candidate_sample_dirs if d and d.exists()), WORKSPACE_ROOT / "samples")

IMG_WIDTH = 224
IMG_HEIGHT = 224

# Page Configuration
st.set_page_config(
    page_title="Potato Doctor AI — Precision Leaf Pathology",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State Initialization
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "selected_crop" not in st.session_state:
    st.session_state.selected_crop = "potato"

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

if "scans_count" not in st.session_state:
    st.session_state.scans_count = 0

is_dark = (st.session_state.theme == "dark")

# ==================== DYNAMIC THEME STYLESHEET ====================
if not is_dark:
    # COMPREHENSIVE LIGHT THEME RULES (Pure White/Slate, Black Text, White Buttons & Navbar)
    theme_css = """
    :root {
        --bg-page: #f8fafc;
        --bg-surface: #ffffff;
        --bg-card: #ffffff;
        --bg-card-hover: #f1f5f9;
        --bg-card-active: #ecfdf5;
        --text-primary: #000000;
        --text-secondary: #1e293b;
        --text-muted: #475569;
        --border-color: #cbd5e1;
        --border-accent: #059669;
        --navbar-bg: #ffffff;
        --navbar-border: #cbd5e1;
        --hero-bg: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 40%, #ffffff 100%);
        --stat-bg: #ffffff;
        --tab-bg: #e2e8f0;
        --box-shadow: 0 4px 18px -2px rgba(0, 0, 0, 0.07);
        --result-verified-bg: #ecfdf5;
        --result-rejected-bg: #fef2f2;
        --result-uncertain-bg: #fffbeb;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #f8fafc !important;
        color: #000000 !important;
    }

    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }

    /* Force all text in Light Theme to black / dark charcoal */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp li, .stApp a,
    .stApp strong, .stApp b, .stMarkdown, .stText {
        color: #000000 !important;
    }

    /* Top Navbar Row */
    div[data-testid="stHorizontalBlock"]:has(#navbar-marker) {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 16px !important;
        padding: 14px 22px !important;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.06) !important;
        margin-bottom: 24px !important;
        align-items: center !important;
    }
    .nav-link {
        color: #000000 !important;
        background: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        text-decoration: none !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 6px 14px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    .nav-link:hover {
        color: #047857 !important;
        background: #ecfdf5 !important;
        border-color: #10b981 !important;
    }

    /* Sidebar in Light Mode */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #ffffff !important;
        border-right: 1px solid #cbd5e1 !important;
        color: #000000 !important;
    }
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] caption {
        color: #334155 !important;
    }

    /* ALL BUTTONS IN LIGHT MODE: PURE WHITE WITH CRISP BORDER AND BLACK TEXT */
    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1.5px solid #94a3b8 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
    }
    .stButton > button * {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    .stButton > button:hover {
        background-color: #ecfdf5 !important;
        border-color: #10b981 !important;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.2) !important;
    }
    .stButton > button:hover * {
        color: #047857 !important;
    }
    .stButton > button:active {
        background-color: #d1fae5 !important;
    }

    /* TABS IN LIGHT MODE */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #e2e8f0 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 6px !important;
        gap: 8px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #000000 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
    }
    .stTabs [data-baseweb="tab"] * {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #047857 !important;
        border-bottom: none !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    }
    .stTabs [aria-selected="true"] * {
        color: #047857 !important;
        font-weight: 800 !important;
    }

    /* FILE UPLOADER IN LIGHT MODE */
    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 2px dashed #94a3b8 !important;
        border-radius: 14px !important;
        padding: 12px !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #f8fafc !important;
    }
    [data-testid="stFileUploader"] * {
        color: #000000 !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #94a3b8 !important;
    }

    /* CAMERA INPUT IN LIGHT MODE */
    [data-testid="stCameraInput"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 14px !important;
        padding: 12px !important;
    }
    [data-testid="stCameraInput"] * {
        color: #000000 !important;
    }

    /* RADIO BUTTONS IN LIGHT MODE */
    div[data-testid="stRadio"] div[role="radiogroup"] {
        background-color: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 4px !important;
    }
    div[data-testid="stRadio"] label, div[data-testid="stRadio"] span, div[data-testid="stRadio"] p {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* EXPANDERS IN LIGHT MODE */
    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
    }
    div[data-testid="stExpander"] details summary {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    div[data-testid="stExpander"] details summary * {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
        color: #000000 !important;
    }

    /* CARDS IN LIGHT MODE */
    .crop-card {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.06) !important;
    }
    .crop-card:hover {
        background: #f8fafc !important;
        border-color: #10b981 !important;
    }
    .crop-card.active {
        background: #ecfdf5 !important;
        border: 2px solid #10b981 !important;
    }
    .crop-card h3 {
        color: #000000 !important;
    }
    .crop-card p, .crop-card ul, .crop-card li {
        color: #1e293b !important;
    }

    /* HERO BANNER IN LIGHT MODE */
    .hero-banner {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 40%, #ffffff 100%) !important;
        border: 1px solid #a7f3d0 !important;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.05) !important;
    }
    .hero-banner h1, .hero-banner p {
        color: #000000 !important;
    }

    /* STAT BOX IN LIGHT MODE */
    .stat-box {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }
    .stat-box h4, .stat-box p, .stat-box div {
        color: #000000 !important;
    }

    /* RESULT PANELS IN LIGHT MODE */
    .result-verified {
        background: #ecfdf5 !important;
        border: 2px solid #10b981 !important;
    }
    .result-verified h1, .result-verified h2, .result-verified div {
        color: #000000 !important;
    }
    .result-rejected {
        background: #fef2f2 !important;
        border: 2px solid #ef4444 !important;
    }
    .result-rejected h1, .result-rejected h2, .result-rejected div, .result-rejected p {
        color: #000000 !important;
    }
    .result-uncertain {
        background: #fffbeb !important;
        border: 2px solid #f59e0b !important;
    }
    .result-uncertain h1, .result-uncertain h2, .result-uncertain div, .result-uncertain p {
        color: #000000 !important;
    }

    /* BADGES */
    .badge-online {
        background: #ecfdf5 !important;
        color: #047857 !important;
        border: 1px solid #10b981 !important;
    }
    .badge-guard {
        background: #eff6ff !important;
        color: #1d4ed8 !important;
        border: 1px solid #3b82f6 !important;
    }
    .badge-cloud {
        background: #fbf5ff !important;
        color: #7e22ce !important;
        border: 1px solid #a855f7 !important;
    }
    .status-active {
        background: #ecfdf5 !important;
        color: #047857 !important;
        border: 1px solid #10b981 !important;
    }
    .status-guarded {
        background: #fffbeb !important;
        color: #b45309 !important;
        border: 1px solid #f59e0b !important;
    }

    /* PROGRESS METERS IN LIGHT MODE */
    [data-testid="stProgress"] > div {
        background-color: #cbd5e1 !important;
    }
    [data-testid="stProgress"] > div > div > div > div {
        background-color: #10b981 !important;
    }
    """
else:
    # COMPREHENSIVE DARK THEME RULES (Obsidian & Emerald Glassmorphism)
    theme_css = """
    :root {
        --bg-page: #080c0a;
        --bg-surface: #0f1712;
        --bg-card: linear-gradient(160deg, rgba(20, 30, 24, 0.85) 0%, rgba(12, 18, 14, 0.85) 100%);
        --bg-card-hover: linear-gradient(160deg, rgba(24, 38, 30, 0.95) 0%, rgba(16, 24, 19, 0.95) 100%);
        --bg-card-active: linear-gradient(160deg, rgba(16, 185, 129, 0.16) 0%, rgba(6, 78, 59, 0.22) 100%);
        --text-primary: #ffffff;
        --text-secondary: #d1d5db;
        --text-muted: #9ca3af;
        --border-color: rgba(255, 255, 255, 0.08);
        --border-accent: rgba(52, 211, 153, 0.45);
        --navbar-bg: rgba(13, 20, 16, 0.9);
        --navbar-border: rgba(52, 211, 153, 0.25);
        --hero-bg: radial-gradient(120% 120% at 50% 0%, rgba(16, 185, 129, 0.22) 0%, rgba(6, 78, 59, 0.12) 50%, rgba(8, 12, 10, 0.8) 100%);
        --stat-bg: rgba(15, 23, 20, 0.7);
        --tab-bg: rgba(17, 24, 20, 0.6);
        --box-shadow: 0 12px 36px -10px rgba(0, 0, 0, 0.7);
        --result-verified-bg: linear-gradient(145deg, rgba(16, 185, 129, 0.14) 0%, rgba(6, 78, 59, 0.2) 100%);
        --result-rejected-bg: linear-gradient(145deg, rgba(239, 68, 68, 0.15) 0%, rgba(153, 27, 27, 0.18) 100%);
        --result-uncertain-bg: linear-gradient(145deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.18) 100%);
    }

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #080c0a !important;
        color: #f3f4f6 !important;
    }

    header[data-testid="stHeader"] {
        background-color: #080c0a !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Top Navbar Row */
    div[data-testid="stHorizontalBlock"]:has(#navbar-marker) {
        background: rgba(13, 20, 16, 0.9) !important;
        border: 1px solid rgba(52, 211, 153, 0.25) !important;
        border-radius: 16px !important;
        padding: 14px 22px !important;
        box-shadow: 0 12px 36px -10px rgba(0, 0, 0, 0.7) !important;
        margin-bottom: 24px !important;
        align-items: center !important;
    }
    .nav-link {
        color: #9ca3af !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        text-decoration: none !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 6px 14px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    .nav-link:hover {
        color: #34d399 !important;
        background: rgba(16, 185, 129, 0.15) !important;
        border-color: #10b981 !important;
    }

    /* Sidebar in Dark Mode */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #0b110e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #f3f4f6 !important;
    }

    /* BUTTONS IN DARK MODE */
    .stButton > button {
        background-color: #111915 !important;
        color: #f3f4f6 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4) !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
    }
    .stButton > button * {
        color: #f3f4f6 !important;
    }
    .stButton > button:hover {
        background-color: rgba(16, 185, 129, 0.2) !important;
        border-color: #10b981 !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25) !important;
    }
    .stButton > button:hover * {
        color: #34d399 !important;
    }

    /* TABS IN DARK MODE */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(17, 24, 20, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        padding: 6px !important;
        gap: 8px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #9ca3af !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
    }
    .stTabs [data-baseweb="tab"] * {
        color: #9ca3af !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(16, 185, 129, 0.25) !important;
        color: #34d399 !important;
        border-bottom: none !important;
    }
    .stTabs [aria-selected="true"] * {
        color: #34d399 !important;
    }

    /* CARDS IN DARK MODE */
    .crop-card {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
    }
    .crop-card:hover {
        background: var(--bg-card-hover) !important;
        border-color: rgba(16, 185, 129, 0.45) !important;
    }
    .crop-card.active {
        background: var(--bg-card-active) !important;
        border: 2px solid #10b981 !important;
    }
    """

# Injected Modern Stylesheet
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    {theme_css}

    html, body, [class*="css"] {{
        font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .block-container {{
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }}

    .nav-links {{
        display: flex;
        gap: 12px;
        align-items: center;
        flex-wrap: wrap;
    }}
    .nav-divider {{
        height: 1px;
        background: var(--border-color);
        margin: 14px 0 24px 0;
    }}

    /* Hero Banner */
    .hero-banner {{
        border-radius: 20px;
        padding: 30px 36px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }}

    .badge-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }}

    /* Section Subheadings */
    .section-title {{
        font-size: 1.35rem;
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .section-subtitle {{
        font-size: 0.92rem;
        color: var(--text-muted);
        margin-bottom: 18px;
    }}

    /* Crop Card Components */
    .crop-card {{
        border-radius: 16px;
        padding: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(8px);
    }}
    .crop-card:hover {{
        transform: translateY(-4px);
    }}

    .crop-status-badge {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 3px 10px;
        border-radius: 999px;
        display: inline-block;
    }}

    /* Diagnosis Result Panels */
    .result-panel {{
        border-radius: 18px;
        padding: 26px 30px;
        margin: 16px 0 24px 0;
        box-shadow: var(--box-shadow);
        backdrop-filter: blur(12px);
    }}

    .stat-box {{
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# Model & Metadata Loading with Caching
@st.cache_resource(show_spinner="Initializing TensorFlow Neural Vision Pipeline...")
def load_models():
    import tensorflow as tf
    disease_model = None
    validator_model = None

    if DISEASE_MODEL_PATH.exists():
        try:
            disease_model = tf.keras.models.load_model(str(DISEASE_MODEL_PATH))
        except Exception as e:
            st.error(f"Error loading disease model: {e}")

    if VALIDATOR_MODEL_PATH.exists():
        try:
            validator_model = tf.keras.models.load_model(str(VALIDATOR_MODEL_PATH))
        except Exception as e:
            st.error(f"Error loading leaf validator model: {e}")

    return disease_model, validator_model

@st.cache_data
def load_metadata():
    if CLASS_MAPPING_PATH.exists():
        with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
            class_mapping = json.load(f)
    else:
        class_mapping = {
            "indices_to_classes": {"0": "Potato___Early_blight", "1": "Potato___Late_blight", "2": "Potato___healthy"},
            "classes_to_indices": {"Potato___Early_blight": 0, "Potato___Late_blight": 1, "Potato___healthy": 2},
            "friendly_names": {
                "Potato___Early_blight": "Potato Early Blight",
                "Potato___Late_blight": "Potato Late Blight",
                "Potato___healthy": "Potato Healthy"
            }
        }

    if VALIDATOR_CONFIG_PATH.exists():
        with open(VALIDATOR_CONFIG_PATH, "r", encoding="utf-8") as f:
            val_config = json.load(f)
    else:
        val_config = {"thresholds": {"high_confidence_potato": 0.80, "low_confidence_rejection": 0.35}}

    if KNOWLEDGE_BASE_PATH.exists():
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)
    else:
        kb = {}

    return class_mapping, val_config, kb

disease_model, validator_model = load_models()
class_mapping, val_config, knowledge_base = load_metadata()

# Image Preprocessing
def preprocess_image(pil_img: Image.Image):
    img_rgb = pil_img.convert("RGB")

    # Guard 1: Minimum Resolution
    if img_rgb.size[0] < 32 or img_rgb.size[1] < 32:
        return None, "Image resolution is too low. Please upload an image with at least 100x100 pixels."

    # Guard 2: Blank or Solid Color Image Detection
    img_thumb = np.array(img_rgb.resize((64, 64)), dtype=np.float32)
    if float(np.std(img_thumb)) < 8.0:
        return None, "The uploaded image appears to be blank, solid, or uninformative. Please upload a clear leaf photo."

    # Normalization [-1.0, 1.0] for MobileNetV2
    img_resized = img_rgb.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.float32)
    img_norm = (img_arr / 127.5) - 1.0
    img_batch = np.expand_dims(img_norm, axis=0)

    return img_batch, None

# Inference Pipeline with Two-Stage Botanical Verification
def analyze_leaf(pil_img: Image.Image):
    img_batch, err = preprocess_image(pil_img)
    if err:
        return {"error": err}

    thresholds = val_config.get("thresholds", {})
    high_thresh = float(thresholds.get("high_confidence_potato", 0.80))
    low_thresh = float(thresholds.get("low_confidence_rejection", 0.35))

    # STAGE 1: Independent Botanical Species Validation
    if validator_model is None:
        return {"error": "Leaf validator model is not available."}

    val_pred = float(validator_model.predict(img_batch, verbose=0)[0][0])

    if val_pred < low_thresh:
        return {
            "status": "REJECTED_NOT_POTATO",
            "title": "Potato Leaf Not Detected",
            "message": "This image does not appear to show a potato leaf. Potato Doctor AI currently supports potato leaves only. Please upload a clear image of a potato leaf.",
            "confidence": val_pred,
            "confidence_pct": round(val_pred * 100, 2)
        }

    if val_pred < high_thresh:
        return {
            "status": "UNCERTAIN_LEAF",
            "title": "Unable to Verify Potato Leaf",
            "message": "We could not confidently verify that this is a potato leaf. Please upload a clear, well-lit image showing the leaf.",
            "confidence": val_pred,
            "confidence_pct": round(val_pred * 100, 2)
        }

    # STAGE 2: 3-Class Potato Disease Pathology
    if disease_model is None:
        return {"error": "Disease classification model is not available."}

    preds = disease_model.predict(img_batch, verbose=0)[0]
    pred_idx = int(np.argmax(preds))

    idx_to_class = {int(k): v for k, v in class_mapping["indices_to_classes"].items()}
    friendly_names = class_mapping.get("friendly_names", {})

    predicted_class = idx_to_class.get(pred_idx, "Potato___Early_blight")
    disease_conf = float(preds[pred_idx])

    probs = {}
    for i, p in enumerate(preds):
        c_id = idx_to_class.get(i, f"Class_{i}")
        f_name = friendly_names.get(c_id, c_id)
        probs[f_name] = round(float(p) * 100, 2)

    return {
        "status": "VERIFIED_POTATO_LEAF",
        "predicted_class": predicted_class,
        "disease_name": friendly_names.get(predicted_class, predicted_class),
        "confidence": disease_conf,
        "confidence_pct": round(disease_conf * 100, 2),
        "validation_conf_pct": round(val_pred * 100, 2),
        "probabilities": probs,
        "guidance": knowledge_base.get(predicted_class, {})
    }

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1518977676601-b53f82aba655?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.markdown("## 🥔 Potato Doctor AI")
    st.caption("Precision Botanical Pathology & Agronomic Defense")
    
    st.markdown("---")
    
    # Theme Switcher in Sidebar
    st.markdown("### 🎨 Interface Theme")
    theme_btn_text = "☀️ Switch to Light Mode" if is_dark else "🌙 Switch to Dark Mode"
    st.button(theme_btn_text, key="btn_sidebar_theme_toggle", on_click=toggle_theme, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📡 System Diagnostics")
    
    both_ready = (disease_model is not None) and (validator_model is not None)
    if both_ready:
        st.success("🟢 AI Neural Models Online")
    else:
        st.error("🔴 Models Offline or Initializing")

    st.markdown(f"**Session Scans Completed:** `{st.session_state.scans_count}`")

    with st.expander("🔬 Model Specs & Telemetry", expanded=False):
        st.markdown(f"""
        - **Disease Pathology Model**:
          - Architecture: MobileNetV2 Transfer Learning
          - Size: `{round(DISEASE_MODEL_PATH.stat().st_size / (1024*1024), 2) if DISEASE_MODEL_PATH.exists() else 0} MB`
          - Held-out Test Accuracy: **97.22%** (Macro F1: 0.958)
          - Scope: Early Blight, Late Blight, Healthy
        - **Botanical Species Validator**:
          - Architecture: Binary MobileNetV2
          - Size: `{round(VALIDATOR_MODEL_PATH.stat().st_size / (1024*1024), 2) if VALIDATOR_MODEL_PATH.exists() else 0} MB`
          - ROC AUC: **0.9971**
          - High Confidence Gate: `≥ 80.0%`
          - Low Rejection Gate: `< 35.0%`
        """)

    with st.expander("📜 Agricultural References", expanded=False):
        st.markdown("""
        - **Cornell University**: Vegetable MD Extension
        - **University of California**: UC IPM Guidelines
        - **FRAC Code Guidelines**: Fungicide Resistance Action Committee
        - **USDA Agricultural Research Service (ARS)**
        """)

    st.markdown("---")
    st.markdown("### 👨‍💻 Developer & Project Credits")
    st.markdown("""
    **Muhammad Raheel Chishti**  
    *SMIT Final Year Project (FYP)*  
    *AI & Deep Learning Botanical Vision*
    """)
    st.caption("Native Streamlit Cloud Architecture")

# ==================== TOP NAVIGATION BAR ====================
navbar_container = st.container()
with navbar_container:
    nav_col1, nav_col2, nav_col3 = st.columns([4.2, 4.4, 2.4], vertical_alignment="center")
    
    with nav_col1:
        st.markdown(f"""
        <span id="navbar-marker" style="display:none;"></span>
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 2.1rem;">🥔</span>
            <div>
                <div style="font-size: 1.35rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em; line-height: 1.15;">
                    Potato Doctor AI
                </div>
                <div style="font-size: 0.74rem; color: #10b981; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;">
                    Precision Leaf Pathology System
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with nav_col2:
        st.markdown("""
        <div class="nav-links">
            <a href="#crops" class="nav-link">🌾 Crops</a>
            <a href="#studio" class="nav-link">🔬 Diagnostic Studio</a>
            <a href="#pipeline" class="nav-link">🧬 Pipeline</a>
            <a href="#protocols" class="nav-link">📋 Protocols</a>
        </div>
        """, unsafe_allow_html=True)
        
    with nav_col3:
        btn_c1, btn_c2 = st.columns([1.1, 1.1], vertical_alignment="center")
        with btn_c1:
            st.markdown('<span class="badge-pill badge-online">● Online</span>', unsafe_allow_html=True)
        with btn_c2:
            toggle_label = "☀️ Light" if is_dark else "🌙 Dark"
            st.button(toggle_label, key="btn_navbar_theme_toggle", on_click=toggle_theme, help="Toggle between Light and Dark interface modes")

st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

# ==================== MAIN HERO BANNER ====================
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
        <div style="max-width: 820px;">
            <div style="display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap;">
                <span class="badge-pill badge-online">● Neural Models Online</span>
                <span class="badge-pill badge-guard">🛡️ 2-Stage Botanical Guard Active</span>
                <span class="badge-pill badge-cloud">☁️ Streamlit Cloud Native</span>
            </div>
            <h1 style="margin: 0; font-size: 2.35rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em;">
                Potato Doctor AI <span style="font-weight: 300; color: #10b981;">| Leaf Pathology Studio</span>
            </h1>
            <p style="margin: 10px 0 0 0; color: var(--text-secondary); font-size: 1.05rem; line-height: 1.5;">
                State-of-the-art Deep Vision system trained on native potato crops. Deploys an independent botanical validator to verify <em>Solanum tuberosum</em> foliage before executing disease classification and FRAC-compliant recovery plans.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== FOUR CROP PORTFOLIO CARDS ====================
st.markdown('<div id="crops" class="section-title">🌾 Agricultural Crop Portfolio (4 Channels)</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Choose a crop to activate its diagnostic studio or inspect its botanical safety guard:</div>', unsafe_allow_html=True)

crop_cols = st.columns(4)

crops_data = [
    {
        "id": "potato",
        "icon": "🥔",
        "name": "Potato",
        "species": "Solanum tuberosum",
        "status": "Active & Supported",
        "status_class": "status-active",
        "badge_text": "ACTIVE STUDIO",
        "desc": "Full 2-Stage Botanical Screening & 3-Class Deep Pathology Engine.",
        "pathologies": ["Early Blight", "Late Blight", "Healthy Foliage"],
        "ready": True
    },
    {
        "id": "tomato",
        "icon": "🍅",
        "name": "Tomato",
        "species": "Solanum lycopersicum",
        "status": "Guarded / Coming Soon",
        "status_class": "status-guarded",
        "badge_text": "BOTANICAL GUARD",
        "desc": "Cross-species rejection active. Screened by validator to prevent false inferences.",
        "pathologies": ["Early Blight", "Late Blight", "Leaf Mold"],
        "ready": False
    },
    {
        "id": "corn",
        "icon": "🌽",
        "name": "Corn / Maize",
        "species": "Zea mays",
        "status": "Guarded / Coming Soon",
        "status_class": "status-guarded",
        "badge_text": "BOTANICAL GUARD",
        "desc": "Gramineae grain pathology engine in research. Screened by botanical validator.",
        "pathologies": ["Common Rust", "Northern Blight", "Gray Spot"],
        "ready": False
    },
    {
        "id": "pepper",
        "icon": "🫑",
        "name": "Bell Pepper",
        "species": "Capsicum annuum",
        "status": "Guarded / Coming Soon",
        "status_class": "status-guarded",
        "badge_text": "BOTANICAL GUARD",
        "desc": "Capsicum pathology module undergoing validation. Guarded against misdiagnosis.",
        "pathologies": ["Bacterial Spot", "Anthracnose", "Healthy"],
        "ready": False
    }
]

for idx, crop in enumerate(crops_data):
    with crop_cols[idx]:
        is_active = (st.session_state.selected_crop == crop["id"])
        card_class = "crop-card active" if is_active else "crop-card"
        
        st.markdown(f"""
        <div class="{card_class}">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                    <span style="font-size: 2.2rem;">{crop['icon']}</span>
                    <span class="crop-status-badge {crop['status_class']}">{crop['badge_text']}</span>
                </div>
                <h3 style="margin: 0 0 2px 0; font-size: 1.25rem; font-weight: 800; color: var(--text-primary);">{crop['name']}</h3>
                <div style="font-size: 0.8rem; font-style: italic; color: #10b981; margin-bottom: 8px; font-weight: 700;">{crop['species']}</div>
                <p style="font-size: 0.84rem; color: var(--text-muted); line-height: 1.4; margin-bottom: 12px;">{crop['desc']}</p>
                <div style="font-size: 0.78rem; color: #10b981; font-weight: 800; margin-bottom: 6px;">Target Pathologies:</div>
                <ul style="font-size: 0.78rem; color: var(--text-secondary); padding-left: 18px; margin: 0 0 16px 0;">
                    {''.join([f"<li>{p}</li>" for p in crop['pathologies']])}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        btn_label = f"✓ Selected ({crop['name']})" if is_active else f"Select {crop['name']}"
        if st.button(btn_label, key=f"btn_crop_{crop['id']}", use_container_width=True):
            st.session_state.selected_crop = crop["id"]
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Dynamic Crop View Handling
if st.session_state.selected_crop != "potato":
    guarded_crop = next((c for c in crops_data if c["id"] == st.session_state.selected_crop), None)
    st.markdown(f"""
    <div style="background: var(--result-uncertain-bg); border: 2px solid #f59e0b; border-radius: 16px; padding: 24px 28px; margin-bottom: 24px; box-shadow: var(--box-shadow);">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span style="font-size: 2.4rem;">{guarded_crop['icon']}</span>
                <div>
                    <h2 style="margin: 0; color: var(--text-primary); font-size: 1.4rem; font-weight: 800;">
                        {guarded_crop['name']} ({guarded_crop['species']}) — Guarded by Botanical Validator
                    </h2>
                    <p style="margin: 4px 0 0 0; color: var(--text-secondary); font-size: 0.95rem;">
                        To safeguard against false positives, Potato Doctor AI enforces strict botanical gates. Uploads of {guarded_crop['name']} foliage are safely halted by the Two-Stage Validator.
                    </p>
                </div>
            </div>
            <span style="background: rgba(245, 158, 11, 0.25); color: #b45309; border: 1px solid #f59e0b; padding: 6px 14px; border-radius: 999px; font-weight: 800; font-size: 0.85rem;">
                🛡️ Cross-Crop False Inference Defense
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_guard_1, col_guard_2 = st.columns([1, 1])
    with col_guard_1:
        st.markdown(f"""
        #### 🔒 Why is {guarded_crop['name']} Guarded?
        1. **Cross-Crop Ambiguity Risk**: Disease classifiers trained only on potato leaves can mistake tomato early blight or pepper spots for potato diseases with high false confidence.
        2. **Botanical Validation Shield**: Our standalone `potato_validator.keras` model screens foliage first (**ROC AUC = 0.9971**).
        3. **Demonstrated Accuracy**: In held-out testing, non-potato crop leaves are successfully halted with **89.5%** strictly rejected below threshold **τ = 0.35**.
        """)
        
    with col_guard_2:
        st.markdown("#### 🧪 Test Cross-Crop Rejection")
        st.write("You can test our Two-Stage Validator right now by loading a sample non-potato leaf to observe the safety rejection:")
        
        col_btn_test, col_btn_back = st.columns(2)
        with col_btn_test:
            if st.button("🧪 Test Rejection (Tomato Leaf)", use_container_width=True):
                st.session_state.selected_crop = "potato"
                st.session_state.selected_sample = "tomato_leaf.JPG"
                st.rerun()
        with col_btn_back:
            if st.button("🥔 Return to Potato Studio", use_container_width=True):
                st.session_state.selected_crop = "potato"
                st.rerun()

else:
    # Active Potato Diagnostic Studio
    st.markdown('<div id="studio" class="section-title">🔬 Potato Leaf Diagnostic Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Upload a potato leaf photo, choose a calibrated test preset, or use your live camera:</div>', unsafe_allow_html=True)

    tab_upload, tab_sample, tab_camera = st.tabs([
        "📤 Upload Leaf Specimen", 
        "🧪 1-Click Test Presets", 
        "📷 Live Camera Capture"
    ])

    selected_image = None
    sample_label = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload leaf image (PNG, JPG, JPEG, WEBP)", 
            type=["jpg", "jpeg", "png", "webp"],
            key="file_uploader_input"
        )
        if uploaded_file:
            try:
                selected_image = Image.open(uploaded_file)
                sample_label = f"Uploaded File: {uploaded_file.name}"
            except Exception as e:
                st.error(f"Cannot read image file: {e}")

    with tab_sample:
        st.markdown("Select any pre-calibrated sample to test both potato leaf pathology and cross-species rejection defense:")
        sample_cols = st.columns(6)

        samples_config = [
            {"name": "🍂 Early Blight", "file": "early_blight.JPG", "label": "Sample: Potato Early Blight", "col": 0},
            {"name": "🌧️ Late Blight", "file": "late_blight.JPG", "label": "Sample: Potato Late Blight", "col": 1},
            {"name": "🌿 Healthy Leaf", "file": "healthy.JPG", "label": "Sample: Potato Healthy Foliage", "col": 2},
            {"name": "🍅 Tomato (Non-Potato)", "file": "tomato_leaf.JPG", "label": "Sample: Tomato Leaf (Cross-Species Rejection)", "col": 3},
            {"name": "🔍 Unclear / Blurry", "file": "unclear_leaf.JPG", "label": "Sample: Inconclusive Foliage", "col": 4},
            {"name": "🚗 Non-Plant Object", "file": "unrelated_object.JPEG", "label": "Sample: Unrelated Non-Plant Object", "col": 5}
        ]

        for s in samples_config:
            with sample_cols[s["col"]]:
                if st.button(s["name"], key=f"btn_sample_{s['file']}", use_container_width=True):
                    st.session_state.selected_sample = s["file"]

        if st.session_state.selected_sample:
            sample_path = SAMPLES_DIR / st.session_state.selected_sample
            if sample_path.exists():
                selected_image = Image.open(sample_path)
                matching_cfg = next((c for c in samples_config if c["file"] == st.session_state.selected_sample), None)
                sample_label = matching_cfg["label"] if matching_cfg else st.session_state.selected_sample

    with tab_camera:
        cam_file = st.camera_input("Capture leaf photo with device camera")
        if cam_file:
            try:
                selected_image = Image.open(cam_file)
                sample_label = "Live Camera Capture"
            except Exception as e:
                st.error(f"Cannot access camera image: {e}")

    # ==================== ANALYSIS & RESULTS ====================
    if selected_image is not None:
        st.markdown("---")
        col_img, col_diag = st.columns([1, 1.6], gap="large")

        with col_img:
            st.markdown(f"#### Specimen: {sample_label or 'Analyzed Leaf'}")
            st.image(selected_image, use_container_width=True)
            
            st.markdown(f"""
            <div class="stat-box">
                <div style="font-size: 0.85rem; color: var(--text-muted); font-weight: 600;">Specimen Dimensions</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: var(--text-primary);">
                    {selected_image.size[0]} × {selected_image.size[1]} px
                </div>
                <div style="font-size: 0.82rem; color: #10b981; margin-top: 4px; font-weight: 700;">
                    Color Mode: {selected_image.mode} | Format: {selected_image.format or 'Memory Image'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_diag:
            with st.spinner("Executing Two-Stage Deep Vision Diagnostics..."):
                result = analyze_leaf(selected_image)
                st.session_state.scans_count += 1

            if "error" in result:
                st.error(f"⚠️ {result['error']}")

            # CASE 1: Rejected (Not a Potato Leaf)
            elif result["status"] == "REJECTED_NOT_POTATO":
                st.markdown(f"""
                <div class="result-panel result-rejected">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                        <span style="color: #ef4444; font-weight: 800; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">
                            🛑 Botanical Guard Intercepted
                        </span>
                        <span style="color: #ef4444; font-size: 0.88rem; background: rgba(239, 68, 68, 0.15); padding: 4px 12px; border-radius: 999px; font-weight: 800;">
                            Potato Probability: <strong>{result['confidence_pct']}%</strong> (&lt; 35%)
                        </span>
                    </div>
                    <h2 style="color: var(--text-primary); margin: 14px 0 10px 0; font-size: 1.85rem; font-weight: 800;">
                        {result['title']}
                    </h2>
                    <div style="background: rgba(239, 68, 68, 0.1); padding: 16px 20px; border-left: 5px solid #ef4444; border-radius: 8px; font-size: 1.05rem; color: var(--text-primary); margin-bottom: 16px; line-height: 1.5; font-weight: 600;">
                        "{result['message']}"
                    </div>
                    <div style="font-size: 0.92rem; color: var(--text-secondary); line-height: 1.5;">
                        <strong>Why was disease prediction halted?</strong><br>
                        To prevent dangerous false inferences, Potato Doctor AI utilizes an independent botanical binary classifier. When leaves from other plant species (like tomato, corn, pepper) or non-plant objects are uploaded, disease diagnosis is halted immediately without generating misleading recommendations.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # CASE 2: Inconclusive / Uncertain Leaf
            elif result["status"] == "UNCERTAIN_LEAF":
                st.markdown(f"""
                <div class="result-panel result-uncertain">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                        <span style="color: #d97706; font-weight: 800; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">
                            ⚠️ Validation: Inconclusive Morphology
                        </span>
                        <span style="color: #d97706; font-size: 0.88rem; background: rgba(245, 158, 11, 0.15); padding: 4px 12px; border-radius: 999px; font-weight: 800;">
                            Potato Probability: <strong>{result['confidence_pct']}%</strong> (35% - 80%)
                        </span>
                    </div>
                    <h2 style="color: var(--text-primary); margin: 14px 0 10px 0; font-size: 1.85rem; font-weight: 800;">
                        {result['title']}
                    </h2>
                    <div style="background: rgba(245, 158, 11, 0.1); padding: 16px 20px; border-left: 5px solid #f59e0b; border-radius: 8px; font-size: 1.05rem; color: var(--text-primary); margin-bottom: 16px; line-height: 1.5; font-weight: 600;">
                        "{result['message']}"
                    </div>
                    <div style="font-size: 0.92rem; color: var(--text-secondary); line-height: 1.5;">
                        <strong>Photography tips for reliable diagnosis:</strong><br>
                        • Ensure ample natural sunlight without strong harsh shadows.<br>
                        • Place the leaf flat on a neutral background or hold steady.<br>
                        • Avoid blurry focus, distant views, or images showing multiple plant types.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # CASE 3: Verified Potato Leaf Pathology
            elif result["status"] == "VERIFIED_POTATO_LEAF":
                guidance = result.get("guidance", {})
                severity = guidance.get("severity", "Verified")
                is_healthy = "healthy" in result["predicted_class"].lower()
                is_late = "late" in result["predicted_class"].lower()

                accent_color = "#10b981" if is_healthy else ("#ef4444" if is_late else "#f59e0b")

                st.markdown(f"""
                <div class="result-panel result-verified" style="border-color: {accent_color};">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                        <span style="color: #10b981; font-weight: 800; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">
                            ✅ Verified Potato Leaf (Solanum tuberosum) — {result['validation_conf_pct']}%
                        </span>
                        <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid {accent_color}; color: {accent_color}; font-size: 0.82rem; font-weight: 800; padding: 4px 12px; border-radius: 999px;">
                            Severity: {severity}
                        </span>
                    </div>
                    <h1 style="color: var(--text-primary); margin: 12px 0 4px 0; font-size: 2.25rem; font-weight: 800;">
                        {result['disease_name']}
                    </h1>
                    <div style="color: #10b981; font-style: italic; margin-bottom: 16px; font-size: 0.95rem; font-weight: 700;">
                        Scientific pathogen: <strong>{guidance.get('scientific_name', 'Solanum tuberosum')}</strong>
                    </div>
                    <div style="font-size: 1.15rem; color: var(--text-primary); margin-bottom: 8px;">
                        Pathology Diagnostic Confidence: <strong style="color: {accent_color}; font-size: 1.55rem;">{result['confidence_pct']}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Probability Distribution Gauges
                st.markdown("##### 📊 Class Probability Breakdown")
                for c_name, c_prob in result["probabilities"].items():
                    col_pname, col_pbar = st.columns([1.5, 3])
                    with col_pname:
                        st.write(f"**{c_name}**: `{c_prob}%`")
                    with col_pbar:
                        st.progress(c_prob / 100.0)

                # Comprehensive Agronomic Protocols
                st.markdown("---")
                st.markdown('<div id="protocols" class="section-title">📋 Evidence-Based Agronomic Management Protocol</div>', unsafe_allow_html=True)

                tab_act, tab_treat, tab_prev, tab_sym = st.tabs([
                    "⚡ Immediate Actions (24-48h)", 
                    "💊 Chemical Treatments (FRAC)", 
                    "🛡️ Prevention & Cultural", 
                    "🔬 Symptoms & Identification"
                ])

                with tab_act:
                    for act in guidance.get("immediate_actions", []):
                        st.markdown(f"• 🔴 **Action**: {act}")

                with tab_treat:
                    treatments = guidance.get("treatments", {})
                    st.markdown("##### Fungicide & Protectant Formulations")
                    for fg in treatments.get("fungicides", []):
                        st.markdown(f"• 💊 {fg}")
                    if treatments.get("precautions"):
                        st.warning(f"⚠️ **Precautionary Safety**: {treatments['precautions']}")

                with tab_prev:
                    for pr in guidance.get("prevention", []):
                        st.markdown(f"• 🛡️ {pr}")

                with tab_sym:
                    for sm in guidance.get("symptoms", []):
                        st.markdown(f"• 🔬 {sm}")

                if guidance.get("sources"):
                    st.caption(f"Curated Agricultural Extension Sources: {', '.join(guidance['sources'])}")

    else:
        # Default Instructional Guide when no image is loaded
        st.info("👆 Upload an image, choose a calibrated 1-click test preset, or use your camera to begin instant diagnosis.")

        st.markdown('<div id="pipeline" class="section-title">🧬 How Potato Doctor AI\'s Two-Stage Pipeline Works</div>', unsafe_allow_html=True)
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.markdown("""
            <div class="stat-box">
                <h4 style="color: #10b981; margin: 0 0 6px 0; font-weight: 800;">1. Botanical Validation</h4>
                <p style="color: var(--text-secondary); font-size: 0.88rem; margin: 0; line-height: 1.45;">
                    The upload is first evaluated by a standalone binary MobileNetV2 classifier. Non-potato foliage (tomato, pepper, corn) and non-plant items are safely halted.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_w2:
            st.markdown("""
            <div class="stat-box">
                <h4 style="color: #2563eb; margin: 0 0 6px 0; font-weight: 800;">2. Disease Pathology</h4>
                <p style="color: var(--text-secondary); font-size: 0.88rem; margin: 0; line-height: 1.45;">
                    Once verified as <em>Solanum tuberosum</em> with high confidence (≥ 80%), the 3-class pathology model diagnoses Early Blight, Late Blight, or Healthy foliage.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_w3:
            st.markdown("""
            <div class="stat-box">
                <h4 style="color: #7c3aed; margin: 0 0 6px 0; font-weight: 800;">3. FRAC Agronomy Guidance</h4>
                <p style="color: var(--text-secondary); font-size: 0.88rem; margin: 0; line-height: 1.45;">
                    Provides immediate 24-48h quarantine measures, FRAC-compliant fungicide rotation, biological controls (Bacillus subtilis), and cultural prevention.
                </p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-muted); font-size: 0.85rem; padding: 12px 0;">
    Potato Doctor AI • SMIT Final Year Project • Developed by <strong>Muhammad Raheel Chishti</strong><br>
    <em>Solanum tuberosum</em> Deep Leaf Pathology & Botanical Safeguard System • Streamlit Community Cloud Native
</div>
""", unsafe_allow_html=True)
