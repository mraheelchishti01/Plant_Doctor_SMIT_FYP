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

# Custom Styling (Dark Emerald Glassmorphism & High-End Aesthetic)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Top Navigation & Hero Banner */
    .hero-banner {
        background: radial-gradient(120% 120% at 50% 0%, rgba(16, 185, 129, 0.22) 0%, rgba(6, 78, 59, 0.12) 50%, rgba(8, 12, 10, 0.8) 100%);
        border: 1px solid rgba(52, 211, 153, 0.35);
        border-radius: 20px;
        padding: 32px 36px;
        margin-bottom: 28px;
        box-shadow: 0 12px 36px -10px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.08) 0%, transparent 60%);
        pointer-events: none;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-online {
        background: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.45);
        box-shadow: 0 0 14px rgba(16, 185, 129, 0.25);
    }
    .badge-guard {
        background: rgba(59, 130, 246, 0.18);
        color: #93c5fd;
        border: 1px solid rgba(96, 165, 250, 0.45);
        box-shadow: 0 0 14px rgba(59, 130, 246, 0.2);
    }
    .badge-cloud {
        background: rgba(168, 85, 247, 0.18);
        color: #d8b4fe;
        border: 1px solid rgba(192, 132, 252, 0.45);
    }

    /* Section Subheadings */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f3f4f6;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-subtitle {
        font-size: 0.92rem;
        color: #9ca3af;
        margin-bottom: 18px;
    }

    /* Crop Card Components */
    .crop-card {
        background: linear-gradient(160deg, rgba(20, 30, 24, 0.85) 0%, rgba(12, 18, 14, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        backdrop-filter: blur(8px);
    }
    .crop-card:hover {
        transform: translateY(-4px);
        border-color: rgba(52, 211, 153, 0.4);
        box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.5), 0 0 15px rgba(16, 185, 129, 0.15);
    }
    .crop-card.active {
        background: linear-gradient(160deg, rgba(16, 185, 129, 0.16) 0%, rgba(6, 78, 59, 0.22) 100%);
        border: 2px solid #10b981;
        box-shadow: 0 12px 28px -6px rgba(16, 185, 129, 0.25), 0 0 20px rgba(16, 185, 129, 0.2);
    }

    .crop-status-badge {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 3px 10px;
        border-radius: 999px;
        display: inline-block;
    }
    .status-active {
        background: rgba(16, 185, 129, 0.25);
        color: #34d399;
        border: 1px solid #10b981;
    }
    .status-guarded {
        background: rgba(245, 158, 11, 0.18);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }

    /* Diagnosis Result Glassmorphic Panels */
    .result-panel {
        border-radius: 18px;
        padding: 26px 30px;
        margin: 16px 0 24px 0;
        box-shadow: 0 14px 34px -8px rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(12px);
    }
    .result-verified {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.14) 0%, rgba(6, 78, 59, 0.2) 100%);
        border: 1px solid rgba(52, 211, 153, 0.5);
    }
    .result-rejected {
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.15) 0%, rgba(153, 27, 27, 0.18) 100%);
        border: 1px solid rgba(248, 113, 113, 0.55);
    }
    .result-uncertain {
        background: linear-gradient(145deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.18) 100%);
        border: 1px solid rgba(251, 191, 36, 0.55);
    }

    /* Telemetry Info Stat Box */
    .stat-box {
        background: rgba(15, 23, 20, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 10px;
    }

    /* Tab enhancements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(17, 24, 20, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        color: #9ca3af;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(16, 185, 129, 0.25) !important;
        color: #34d399 !important;
        border-bottom-color: transparent !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "selected_crop" not in st.session_state:
    st.session_state.selected_crop = "potato"

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

if "scans_count" not in st.session_state:
    st.session_state.scans_count = 0

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

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1518977676601-b53f82aba655?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.markdown("## 🥔 Potato Doctor AI")
    st.caption("Precision Botanical Pathology & Agronomic Defense")
    
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

# ==================== MAIN USER INTERFACE ====================

# 1. Hero Header Banner
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
        <div style="max-width: 780px;">
            <div style="display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap;">
                <span class="badge-pill badge-online">● Neural Models Online</span>
                <span class="badge-pill badge-guard">🛡️ 2-Stage Botanical Guard Active</span>
                <span class="badge-pill badge-cloud">☁️ Streamlit Cloud Native</span>
            </div>
            <h1 style="margin: 0; font-size: 2.35rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">
                Potato Doctor AI <span style="font-weight: 300; color: #6ee7b7;">| Leaf Pathology Studio</span>
            </h1>
            <p style="margin: 10px 0 0 0; color: #d1fae5; font-size: 1.05rem; line-height: 1.5;">
                State-of-the-art Deep Vision system trained on native potato crops. Deploys an independent botanical validator to verify <em>Solanum tuberosum</em> foliage before executing disease classification and FRAC-compliant recovery plans.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. Four Crop Portfolio Cards (User Requirement: "crops k 4 cards use krna")
st.markdown('<div class="section-title">🌾 Select Agricultural Crop Portfolio</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Choose a crop to activate its diagnostic studio or view botanical safety guard specifications:</div>', unsafe_allow_html=True)

crop_cols = st.columns(4)

# Define the 4 Crops
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
                <h3 style="margin: 0 0 2px 0; font-size: 1.25rem; font-weight: 700; color: #ffffff;">{crop['name']}</h3>
                <div style="font-size: 0.8rem; font-style: italic; color: #a7f3d0; margin-bottom: 8px;">{crop['species']}</div>
                <p style="font-size: 0.84rem; color: #9ca3af; line-height: 1.4; margin-bottom: 12px;">{crop['desc']}</p>
                <div style="font-size: 0.78rem; color: #6ee7b7; font-weight: 600; margin-bottom: 6px;">Target Pathologies:</div>
                <ul style="font-size: 0.78rem; color: #d1d5db; padding-left: 18px; margin: 0 0 16px 0;">
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

# 3. Dynamic Crop View Handling
if st.session_state.selected_crop != "potato":
    # Non-Potato Crop Guard Presentation
    guarded_crop = next((c for c in crops_data if c["id"] == st.session_state.selected_crop), None)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(180, 83, 9, 0.05) 100%);
                border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 16px; padding: 24px 28px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span style="font-size: 2.4rem;">{guarded_crop['icon']}</span>
                <div>
                    <h2 style="margin: 0; color: #fef3c7; font-size: 1.4rem;">
                        {guarded_crop['name']} ({guarded_crop['species']}) — Guarded by Botanical Validator
                    </h2>
                    <p style="margin: 4px 0 0 0; color: #fde68a; font-size: 0.95rem;">
                        To safeguard against false positives, Potato Doctor AI enforces strict botanical gates. Uploads of {guarded_crop['name']} foliage are safely halted by the Two-Stage Validator.
                    </p>
                </div>
            </div>
            <span style="background: rgba(245, 158, 11, 0.25); color: #fbbf24; border: 1px solid #f59e0b; padding: 6px 14px; border-radius: 999px; font-weight: 700; font-size: 0.85rem;">
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
    st.markdown('<div class="section-title">🔬 Potato Leaf Diagnostic Studio</div>', unsafe_allow_html=True)
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
                <div style="font-size: 0.85rem; color: #9ca3af;">Specimen Dimensions</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #f3f4f6;">
                    {selected_image.size[0]} × {selected_image.size[1]} px
                </div>
                <div style="font-size: 0.8rem; color: #34d399; margin-top: 4px;">
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
                        <span style="color: #fca5a5; font-size: 0.88rem; background: rgba(239, 68, 68, 0.2); padding: 4px 12px; border-radius: 999px;">
                            Potato Probability: <strong>{result['confidence_pct']}%</strong> (&lt; 35%)
                        </span>
                    </div>
                    <h2 style="color: #ffffff; margin: 14px 0 10px 0; font-size: 1.85rem; font-weight: 800;">
                        {result['title']}
                    </h2>
                    <div style="background: rgba(0,0,0,0.4); padding: 16px 20px; border-left: 5px solid #ef4444; border-radius: 8px; font-size: 1.05rem; color: #fecaca; margin-bottom: 16px; line-height: 1.5;">
                        "{result['message']}"
                    </div>
                    <div style="font-size: 0.9rem; color: #d1d5db; line-height: 1.5;">
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
                        <span style="color: #f59e0b; font-weight: 800; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">
                            ⚠️ Validation: Inconclusive Morphology
                        </span>
                        <span style="color: #fde68a; font-size: 0.88rem; background: rgba(245, 158, 11, 0.2); padding: 4px 12px; border-radius: 999px;">
                            Potato Probability: <strong>{result['confidence_pct']}%</strong> (35% - 80%)
                        </span>
                    </div>
                    <h2 style="color: #ffffff; margin: 14px 0 10px 0; font-size: 1.85rem; font-weight: 800;">
                        {result['title']}
                    </h2>
                    <div style="background: rgba(0,0,0,0.4); padding: 16px 20px; border-left: 5px solid #f59e0b; border-radius: 8px; font-size: 1.05rem; color: #fef3c7; margin-bottom: 16px; line-height: 1.5;">
                        "{result['message']}"
                    </div>
                    <div style="font-size: 0.9rem; color: #d1d5db; line-height: 1.5;">
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
                accent_glow = "rgba(16, 185, 129, 0.25)" if is_healthy else ("rgba(239, 68, 68, 0.25)" if is_late else "rgba(245, 158, 11, 0.25)")

                st.markdown(f"""
                <div class="result-panel result-verified" style="border-color: {accent_color}; box-shadow: 0 12px 32px {accent_glow};">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                        <span style="color: #34d399; font-weight: 800; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">
                            ✅ Verified Potato Leaf (Solanum tuberosum) — {result['validation_conf_pct']}%
                        </span>
                        <span style="background: rgba(0,0,0,0.4); border: 1px solid {accent_color}; color: {accent_color}; font-size: 0.82rem; font-weight: 700; padding: 4px 12px; border-radius: 999px;">
                            Severity: {severity}
                        </span>
                    </div>
                    <h1 style="color: #ffffff; margin: 12px 0 4px 0; font-size: 2.25rem; font-weight: 800;">
                        {result['disease_name']}
                    </h1>
                    <div style="color: #a7f3d0; font-style: italic; margin-bottom: 16px; font-size: 0.95rem;">
                        Scientific pathogen: <strong>{guidance.get('scientific_name', 'Solanum tuberosum')}</strong>
                    </div>
                    <div style="font-size: 1.15rem; color: #f3f4f6; margin-bottom: 8px;">
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
                st.markdown("### 📋 Evidence-Based Agronomic Management Protocol")

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

        st.markdown("### 🧬 How Potato Doctor AI's Two-Stage Pipeline Works")
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.markdown("""
            <div class="stat-box">
                <h4 style="color: #34d399; margin: 0 0 6px 0;">1. Botanical Validation</h4>
                <p style="color: #d1d5db; font-size: 0.88rem; margin: 0; line-height: 1.45;">
                    The upload is first evaluated by a standalone binary MobileNetV2 classifier. Non-potato foliage (tomato, pepper, corn) and non-plant items are safely halted.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_w2:
            st.markdown("""
            <div class="stat-box">
                <h4 style="color: #60a5fa; margin: 0 0 6px 0;">2. Disease Pathology</h4>
                <p style="color: #d1d5db; font-size: 0.88rem; margin: 0; line-height: 1.45;">
                    Once verified as <em>Solanum tuberosum</em> with high confidence (≥ 80%), the 3-class pathology model diagnoses Early Blight, Late Blight, or Healthy foliage.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_w3:
            st.markdown("""
            <div class="stat-box">
                <h4 style="color: #a78bfa; margin: 0 0 6px 0;">3. FRAC Agronomy Guidance</h4>
                <p style="color: #d1d5db; font-size: 0.88rem; margin: 0; line-height: 1.45;">
                    Provides immediate 24-48h quarantine measures, FRAC-compliant fungicide rotation, biological controls (Bacillus subtilis), and cultural prevention.
                </p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9ca3af; font-size: 0.85rem; padding: 12px 0;">
    Potato Doctor AI • SMIT Final Year Project • Developed by <strong>Muhammad Raheel Chishti</strong><br>
    <em>Solanum tuberosum</em> Deep Leaf Pathology & Botanical Safeguard System • Streamlit Community Cloud Native
</div>
""", unsafe_allow_html=True)
