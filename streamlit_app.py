import os
import sys
import json
import io
from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st

# Windows MSVC runtime DLL fix if running locally on Windows
_msvc_dir = r"C:\Program Files\Microsoft Office\root\Client"
if os.path.exists(_msvc_dir) and _msvc_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _msvc_dir + ";" + os.environ.get("PATH", "")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Setup Paths
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
SAMPLES_DIR = WORKSPACE_ROOT / "frontend" / "public" / "samples"

IMG_WIDTH = 224
IMG_HEIGHT = 224

# Page configuration
st.set_page_config(
    page_title="Potato Doctor AI — Precision Leaf Pathology",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    .badge-online {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-twostage {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(59, 130, 246, 0.15);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.35);
        margin-left: 8px;
    }
    .metric-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
    }
    .rejection-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(185, 28, 28, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 14px;
        padding: 24px;
        margin: 20px 0;
    }
    .uncertain-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(217, 119, 6, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 14px;
        padding: 24px;
        margin: 20px 0;
    }
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 14px;
        padding: 24px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Resource Caching
@st.cache_resource(show_spinner="Loading deep learning models into memory...")
def load_models():
    import tensorflow as tf
    
    disease_model = None
    validator_model = None
    
    if DISEASE_MODEL_PATH.exists():
        try:
            disease_model = tf.keras.models.load_model(str(DISEASE_MODEL_PATH))
        except Exception as e:
            st.error(f"Failed to load disease model: {e}")
            
    if VALIDATOR_MODEL_PATH.exists():
        try:
            validator_model = tf.keras.models.load_model(str(VALIDATOR_MODEL_PATH))
        except Exception as e:
            st.error(f"Failed to load leaf validator model: {e}")
            
    return disease_model, validator_model

@st.cache_data
def load_metadata():
    # Class mapping
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
        
    # Validator config
    if VALIDATOR_CONFIG_PATH.exists():
        with open(VALIDATOR_CONFIG_PATH, "r", encoding="utf-8") as f:
            val_config = json.load(f)
    else:
        val_config = {"thresholds": {"high_confidence_potato": 0.80, "low_confidence_rejection": 0.35}}
        
    # Knowledge base
    if KNOWLEDGE_BASE_PATH.exists():
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)
    else:
        kb = {}
        
    return class_mapping, val_config, kb

# Load resources
disease_model, validator_model = load_models()
class_mapping, val_config, knowledge_base = load_metadata()

# Preprocessing function
def preprocess_image(pil_img: Image.Image):
    img_rgb = pil_img.convert("RGB")
    
    # Check minimum resolution
    if img_rgb.size[0] < 32 or img_rgb.size[1] < 32:
        return None, "Image resolution is too low. Please upload an image with at least 100x100 pixels."
        
    # Check blank or solid color
    img_thumb = np.array(img_rgb.resize((64, 64)), dtype=np.float32)
    if float(np.std(img_thumb)) < 8.0:
        return None, "The uploaded image appears to be blank, solid, or uninformative. Please upload a clear leaf photo."
        
    # Resize to 224x224
    img_resized = img_rgb.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.float32)
    # Standard MobileNetV2 normalization [-1.0, 1.0]
    img_norm = (img_arr / 127.5) - 1.0
    img_batch = np.expand_dims(img_norm, axis=0)
    
    return img_batch, None

# Run Prediction Pipeline
def analyze_leaf(pil_img: Image.Image):
    img_batch, err = preprocess_image(pil_img)
    if err:
        return {"error": err}
        
    thresholds = val_config.get("thresholds", {})
    high_thresh = float(thresholds.get("high_confidence_potato", 0.80))
    low_thresh = float(thresholds.get("low_confidence_rejection", 0.35))
    
    # STAGE 1: Botanical Potato-Leaf Validation
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
        
    # STAGE 2: Potato Disease Prediction
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

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1518977676601-b53f82aba655?w=500&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("🥔 Potato Doctor AI")
    st.caption("Deep Learning Leaf Pathology & Agricultural Guidance")
    
    st.markdown("---")
    st.markdown("### 📡 System Diagnostics")
    
    both_ready = (disease_model is not None) and (validator_model is not None)
    if both_ready:
        st.success("🟢 AI Models Online (TensorFlow / Keras)")
    else:
        st.error("🔴 Models Offline or degraded")
        
    with st.expander("Model Specifications", expanded=False):
        st.markdown(f"""
        - **Disease Model**: `{DISEASE_MODEL_PATH.name}`
          - Architecture: MobileNetV2 Transfer Learning
          - Size: `{round(DISEASE_MODEL_PATH.stat().st_size / (1024*1024), 2) if DISEASE_MODEL_PATH.exists() else 0} MB`
          - Classes: Early Blight, Late Blight, Healthy
        - **Botanical Validator**: `{VALIDATOR_MODEL_PATH.name}`
          - Architecture: Binary MobileNetV2
          - Size: `{round(VALIDATOR_MODEL_PATH.stat().st_size / (1024*1024), 2) if VALIDATOR_MODEL_PATH.exists() else 0} MB`
          - Threshold: `>= 80%` potato confidence
        """)
        
    st.markdown("---")
    st.markdown("### 🌾 Crop Coverage")
    st.markdown("""
    - 🥔 **Potato** *(Solanum tuberosum)* — `Active`
    - 🍅 **Tomato** — *Coming Soon*
    - 🌽 **Corn (Maize)** — *Coming Soon*
    - 🫑 **Pepper** — *Coming Soon*
    """)
    
    st.markdown("---")
    st.caption("SMIT Final Year Project — Agricultural Deep Vision")

# ----------------- MAIN UI -----------------

# Header Banner
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; color: #ffffff;">Potato Doctor AI</h1>
            <p style="margin: 6px 0 0 0; color: #a7f3d0; font-size: 1.05rem;">
                Autonomous Botanical Species Validation & Precision Disease Pathology
            </p>
        </div>
        <div>
            <span class="badge-online">● AI Models Online</span>
            <span class="badge-twostage">🛡️ Two-Stage Filter Active</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Diagnostic Studio Tabs
tab_upload, tab_sample, tab_camera = st.tabs(["📤 Upload Specimen", "🧪 Test Samples (1-Click)", "📷 Camera Capture"])

selected_image = None
sample_label = None

with tab_upload:
    uploaded_file = st.file_uploader("Upload a photograph of a potato leaf (JPG, JPEG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        try:
            selected_image = Image.open(uploaded_file)
            sample_label = uploaded_file.name
        except Exception as e:
            st.error(f"Cannot read image file: {e}")

with tab_sample:
    st.markdown("Select any pre-configured specimen to test both potato diagnosis and negative-class rejection:")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🍂 Early Blight", use_container_width=True):
            p = SAMPLES_DIR / "early_blight.JPG"
            if p.exists(): selected_image = Image.open(p); sample_label = "Sample: Early Blight Specimen"
    with col2:
        if st.button("🌧️ Late Blight", use_container_width=True):
            p = SAMPLES_DIR / "late_blight.JPG"
            if p.exists(): selected_image = Image.open(p); sample_label = "Sample: Late Blight Specimen"
    with col3:
        if st.button("🌿 Healthy Leaf", use_container_width=True):
            p = SAMPLES_DIR / "healthy.JPG"
            if p.exists(): selected_image = Image.open(p); sample_label = "Sample: Healthy Potato Specimen"
    with col4:
        if st.button("🍅 Tomato (Negative)", use_container_width=True):
            p = SAMPLES_DIR / "tomato_leaf.JPG"
            if p.exists(): selected_image = Image.open(p); sample_label = "Sample: Non-Potato Leaf (Tomato)"
    with col5:
        if st.button("🔍 Ambiguous Leaf", use_container_width=True):
            p = SAMPLES_DIR / "unclear_leaf.JPG"
            if p.exists(): selected_image = Image.open(p); sample_label = "Sample: Inconclusive Leaf"

with tab_camera:
    cam_file = st.camera_input("Take a photo of a potato leaf with your device camera")
    if cam_file:
        selected_image = Image.open(cam_file)
        sample_label = "Live Camera Capture"

# Analysis Execution
if selected_image is not None:
    st.markdown("---")
    col_img, col_diag = st.columns([1, 1.6], gap="large")
    
    with col_img:
        st.markdown(f"#### Specimen: {sample_label or 'Uploaded Leaf'}")
        st.image(selected_image, use_container_width=True)
        st.caption(f"Original Dimensions: {selected_image.size[0]} × {selected_image.size[1]} px | Format: {selected_image.format or 'RGB'}")
        
    with col_diag:
        with st.spinner("Executing Two-Stage Deep Pathology Pipeline..."):
            result = analyze_leaf(selected_image)
            
        if "error" in result:
            st.error(f"⚠️ {result['error']}")
            
        elif result["status"] == "REJECTED_NOT_POTATO":
            st.markdown(f"""
            <div class="rejection-box">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #ef4444; font-weight: 700; text-transform: uppercase; font-size: 0.85rem;">
                        🛑 Species Validation Filtered
                    </span>
                    <span style="color: #fca5a5; font-size: 0.85rem;">
                        Potato Probability: <strong>{result['confidence_pct']}%</strong>
                    </span>
                </div>
                <h2 style="color: #fecaca; margin: 12px 0 8px 0;">{result['title']}</h2>
                <p style="color: #ffffff; background: rgba(0,0,0,0.3); padding: 12px 16px; border-left: 4px solid #ef4444; border-radius: 6px; font-size: 1rem;">
                    "{result['message']}"
                </p>
                <div style="margin-top: 14px; font-size: 0.88rem; color: #d1d5db;">
                    <strong>Why was prediction halted?</strong><br>
                    To ensure diagnostic safety, Potato Doctor AI deploys an independent botanical validator before disease models. Non-potato foliage (such as tomato, corn, or pepper) is safely rejected to avoid false diagnoses.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        elif result["status"] == "UNCERTAIN_LEAF":
            st.markdown(f"""
            <div class="uncertain-box">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #f59e0b; font-weight: 700; text-transform: uppercase; font-size: 0.85rem;">
                        ⚠️ Validation: Inconclusive
                    </span>
                    <span style="color: #fde68a; font-size: 0.85rem;">
                        Potato Probability: <strong>{result['confidence_pct']}%</strong>
                    </span>
                </div>
                <h2 style="color: #fef3c7; margin: 12px 0 8px 0;">{result['title']}</h2>
                <p style="color: #ffffff; background: rgba(0,0,0,0.3); padding: 12px 16px; border-left: 4px solid #f59e0b; border-radius: 6px; font-size: 1rem;">
                    "{result['message']}"
                </p>
                <div style="margin-top: 14px; font-size: 0.88rem; color: #d1d5db;">
                    <strong>Diagnostic safety notice:</strong> The visual characteristics of this leaf were borderline. Please capture a clear, single leaf against a neutral background in natural light.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        elif result["status"] == "VERIFIED_POTATO_LEAF":
            guidance = result.get("guidance", {})
            severity = guidance.get("severity", "Verified")
            is_healthy = "healthy" in result["predicted_class"].lower()
            
            accent_color = "#10b981" if is_healthy else ("#ef4444" if "late" in result["predicted_class"].lower() else "#f59e0b")
            
            st.markdown(f"""
            <div class="success-box" style="border-color: {accent_color};">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                    <span style="color: #34d399; font-weight: 700; text-transform: uppercase; font-size: 0.82rem;">
                        ✅ Verified Potato Leaf (Solanum tuberosum) — {result['validation_conf_pct']}%
                    </span>
                    <span style="color: #d1d5db; font-size: 0.85rem;">
                        Severity: <strong style="color: {accent_color};">{severity}</strong>
                    </span>
                </div>
                <h1 style="color: #ffffff; margin: 12px 0 4px 0; font-size: 2.1rem;">
                    {result['disease_name']}
                </h1>
                <p style="color: #a7f3d0; margin-bottom: 16px; font-style: italic;">
                    Scientific agent: {guidance.get('scientific_name', 'Solanum tuberosum')}
                </p>
                <div style="font-size: 1.15rem; color: #ffffff; margin-bottom: 16px;">
                    Diagnostic Confidence: <strong style="color: {accent_color}; font-size: 1.4rem;">{result['confidence_pct']}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Breakdown
            st.markdown("##### Class Probability Distribution")
            for c_name, c_prob in result["probabilities"].items():
                col_n, col_p = st.columns([3, 1])
                with col_n:
                    st.progress(c_prob / 100.0)
                with col_p:
                    st.write(f"**{c_name}**: `{c_prob}%`")
                    
            # Detailed Guidance Accordions
            st.markdown("---")
            st.markdown("### 📋 Agronomic Treatment & Management Protocol")
            
            tab_act, tab_treat, tab_prev, tab_sym = st.tabs([
                "⚡ Immediate Actions", "🧪 Treatments", "🛡️ Prevention", "🔍 Symptoms"
            ])
            
            with tab_act:
                for act in guidance.get("immediate_actions", []):
                    st.markdown(f"- 🔴 **Action**: {act}")
                    
            with tab_treat:
                treatments = guidance.get("treatments", {})
                st.markdown("##### Fungicide & Protectant Options")
                for fg in treatments.get("fungicides", []):
                    st.markdown(f"- 💊 {fg}")
                if treatments.get("precautions"):
                    st.warning(f"⚠️ **Precautions**: {treatments['precautions']}")
                    
            with tab_prev:
                for pr in guidance.get("prevention", []):
                    st.markdown(f"- 🛡️ {pr}")
                    
            with tab_sym:
                for sm in guidance.get("symptoms", []):
                    st.markdown(f"- 🔬 {sm}")
                    
            if guidance.get("sources"):
                st.caption(f"Sources: {', '.join(guidance['sources'])}")

else:
    # Empty State Guide
    st.info("👆 Upload an image, choose a test sample, or snap a photo with your camera above to begin real-time diagnosis.")
    
    st.markdown("### How the Two-Stage AI Diagnostic Pipeline Works")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        #### 1. Botanical Validation
        The uploaded leaf is evaluated by an independent binary vision model. If the specimen is a tomato, pepper, corn, or non-potato leaf, the system immediately halts to prevent false predictions.
        """)
    with col_b:
        st.markdown("""
        #### 2. Disease Pathology
        Once botanical identity as *Solanum tuberosum* is confirmed with high confidence (≥ 80%), the 3-class pathology model diagnoses Early Blight, Late Blight, or Optimal Health.
        """)
    with col_c:
        st.markdown("""
        #### 3. Treatment Guidance
        Actionable field protocols, fungicide active ingredients adhering to FRAC guidelines, organic alternatives, and cultural controls are delivered instantly.
        """)

st.markdown("---")
st.caption("Potato Doctor AI • Certified Plant Pathology Research System • Solanum tuberosum Specialist")
