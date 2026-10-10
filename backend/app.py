import os
import sys
import json
import io
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure Windows finds 64-bit MSVC runtime DLLs
_msvc_dir = r"C:\Program Files\Microsoft Office\root\Client"
if os.path.exists(_msvc_dir) and _msvc_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _msvc_dir + ";" + os.environ.get("PATH", "")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Resolve models directory across diverse cloud/container environments (Docker, Render, Railway, local)
_candidate_model_dirs = [
    Path(os.environ["MODELS_DIR"]) if "MODELS_DIR" in os.environ else None,
    WORKSPACE_ROOT / "models",
    Path(__file__).resolve().parent / "models",
    Path("models").resolve(),
    Path("/app/models")
]
MODELS_DIR = next((d for d in _candidate_model_dirs if d and d.exists()), WORKSPACE_ROOT / "models")
REPORTS_DIR = WORKSPACE_ROOT / "reports"
KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent / "knowledge_base.json"

# Separate Disease and Validation Models
MODEL_PATH = MODELS_DIR / "potato_model.keras"
CLASS_MAPPING_PATH = MODELS_DIR / "class_mapping.json"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

VALIDATOR_MODEL_PATH = MODELS_DIR / "potato_validator.keras"
VALIDATOR_CONFIG_PATH = MODELS_DIR / "validator_config.json"

IMG_HEIGHT = 224
IMG_WIDTH = 224

app = FastAPI(
    title="Potato Doctor AI API",
    description="Precision Potato Leaf Pathology & Crop Disease Diagnostic System with Strict Species Validation",
    version="1.1.0"
)

# CORS configuration - Allow all origins by default so any Vercel preview or production deployment can connect
origins_env = os.getenv("CORS_ORIGINS", "*")
if origins_env == "*":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_origin_regex=r"https://.*\.vercel\.app",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Global model caches
_tf_model = None
_validator_model = None
_class_mapping = None
_knowledge_base = None
_validator_config = None

def get_knowledge_base() -> Dict[str, Any]:
    global _knowledge_base
    if _knowledge_base is None:
        if KNOWLEDGE_BASE_PATH.exists():
            with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
                _knowledge_base = json.load(f)
        else:
            _knowledge_base = {}
    return _knowledge_base

def get_class_mapping() -> Dict[str, Any]:
    global _class_mapping
    if _class_mapping is None:
        if CLASS_MAPPING_PATH.exists():
            with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
                _class_mapping = json.load(f)
        else:
            _class_mapping = {
                "indices_to_classes": {
                    "0": "Potato___Early_blight",
                    "1": "Potato___Late_blight",
                    "2": "Potato___healthy"
                },
                "classes_to_indices": {
                    "Potato___Early_blight": 0,
                    "Potato___Late_blight": 1,
                    "Potato___healthy": 2
                },
                "friendly_names": {
                    "Potato___Early_blight": "Potato Early Blight",
                    "Potato___Late_blight": "Potato Late Blight",
                    "Potato___healthy": "Potato Healthy"
                }
            }
    return _class_mapping

def get_validator_config() -> Dict[str, Any]:
    global _validator_config
    if _validator_config is None:
        if VALIDATOR_CONFIG_PATH.exists():
            with open(VALIDATOR_CONFIG_PATH, "r", encoding="utf-8") as f:
                _validator_config = json.load(f)
        else:
            _validator_config = {
                "thresholds": {
                    "high_confidence_potato": 0.80,
                    "low_confidence_rejection": 0.35
                }
            }
    return _validator_config

def get_model():
    """Load and cache the 3-class Potato Disease Classifier model"""
    global _tf_model
    if _tf_model is None and MODEL_PATH.exists():
        try:
            import tensorflow as tf
            print(f"[Backend] Loading Disease Classifier from {MODEL_PATH}...")
            _tf_model = tf.keras.models.load_model(str(MODEL_PATH))
            print("[Backend] Disease Classifier loaded successfully.")
        except Exception as e:
            print(f"[Backend Warning] Failed to load disease classifier: {e}")
            _tf_model = None
    return _tf_model

def get_validator_model():
    """Load and cache the separate Binary Potato-Leaf Validator model"""
    global _validator_model
    if _validator_model is None and VALIDATOR_MODEL_PATH.exists():
        try:
            import tensorflow as tf
            print(f"[Backend] Loading Binary Potato Validator from {VALIDATOR_MODEL_PATH}...")
            _validator_model = tf.keras.models.load_model(str(VALIDATOR_MODEL_PATH))
            print("[Backend] Binary Potato Validator loaded successfully.")
        except Exception as e:
            print(f"[Backend Warning] Failed to load validator model: {e}")
            _validator_model = None
    return _validator_model

@app.on_event("startup")
def startup_event():
    get_class_mapping()
    get_knowledge_base()
    get_validator_config()
    # Preload both independent models
    get_validator_model()
    get_model()

@app.get("/")
def root():
    return {
        "service": "Potato Doctor AI Inference API",
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/health",
        "model_classes": "/classes"
    }

@app.get("/health")
def health_check():
    model = get_model()
    validator = get_validator_model()
    config = get_validator_config()

    disease_exists = MODEL_PATH.exists()
    disease_size_mb = round(MODEL_PATH.stat().st_size / (1024 * 1024), 2) if disease_exists else 0

    val_exists = VALIDATOR_MODEL_PATH.exists()
    val_size_mb = round(VALIDATOR_MODEL_PATH.stat().st_size / (1024 * 1024), 2) if val_exists else 0

    disease_ready = model is not None
    validator_ready = validator is not None
    both_ready = disease_ready and validator_ready

    return {
        "status": "healthy" if both_ready else "degraded",
        "service": "Potato Doctor AI Inference API",
        "version": "1.1.0",
        "api_online": True,
        "model_loaded": both_ready,
        "models_loaded": both_ready,
        "target_crop": "Potato",
        "disease_classifier": {
            "loaded": disease_ready,
            "file_exists": disease_exists,
            "size_mb": disease_size_mb,
            "supported_classes": list(get_class_mapping().get("friendly_names", {}).values())
        },
        "potato_validator": {
            "loaded": validator_ready,
            "file_exists": val_exists,
            "size_mb": val_size_mb,
            "thresholds": config.get("thresholds", {
                "high_confidence_potato": 0.80,
                "low_confidence_rejection": 0.35
            })
        }
    }

@app.get("/classes")
def list_classes():
    mapping = get_class_mapping()
    friendly = mapping.get("friendly_names", {})

    crops_dashboard = [
        {
            "crop": "Potato",
            "scientific_name": "Solanum tuberosum",
            "status": "Available",
            "is_active": True,
            "badge": "Active Model",
            "classes": [
                {
                    "id": k,
                    "friendly_name": v,
                    "description": get_knowledge_base().get(k, {}).get("description", "")
                }
                for k, v in friendly.items()
            ]
        },
        {
            "crop": "Tomato",
            "scientific_name": "Solanum lycopersicum",
            "status": "Coming Soon",
            "is_active": False,
            "badge": "In Development",
            "classes": []
        },
        {
            "crop": "Corn (Maize)",
            "scientific_name": "Zea mays",
            "status": "Coming Soon",
            "is_active": False,
            "badge": "In Development",
            "classes": []
        },
        {
            "crop": "Pepper (Bell)",
            "scientific_name": "Capsicum annuum",
            "status": "Coming Soon",
            "is_active": False,
            "badge": "In Development",
            "classes": []
        }
    ]

    return {
        "active_crop": "Potato",
        "classes": friendly,
        "crops": crops_dashboard
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # -------------------------------------------------------------
    # 1. Validation Pre-Checks: File presence, extension, size limit
    # -------------------------------------------------------------
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided. Please upload a clear potato leaf image."
        )

    allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: JPG, JPEG, PNG, WEBP, BMP."
        )

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes). Please upload a valid image."
        )

    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image file size exceeds the 15MB limit. Please upload a smaller image."
        )

    # -------------------------------------------------------------
    # 2. PIL Image Integrity & Image Corruption / Quality Checks
    # -------------------------------------------------------------
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()  # Verify underlying file integrity
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The uploaded image file is corrupt or unreadable: {str(e)}"
        )

    # Re-open for image arrays and dimensions
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    original_dims = image.size  # (width, height)

    # Check minimum resolution
    if original_dims[0] < 32 or original_dims[1] < 32:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image resolution is too low. Please upload an image with at least 100x100 pixels."
        )

    # Blank / Solid Color / Zero-Variance Image Check
    img_sample = np.array(image.resize((64, 64)), dtype=np.float32)
    pixel_std = float(np.std(img_sample))
    if pixel_std < 8.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image appears to be blank, solid, or uninformative. Please upload a clear photograph of a leaf."
        )

    # Preprocess image for MobileNetV2: [-1.0, 1.0]
    img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.float32)
    img_preprocessed = (img_arr / 127.5) - 1.0
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # -------------------------------------------------------------
    # 3. STAGE 1: BOTANICAL POTATO-LEAF VALIDATION
    # -------------------------------------------------------------
    validator_model = get_validator_model()
    val_config = get_validator_config()
    thresholds = val_config.get("thresholds", {})
    high_thresh = float(thresholds.get("high_confidence_potato", 0.80))
    low_thresh = float(thresholds.get("low_confidence_rejection", 0.35))

    if validator_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Potato Doctor AI leaf validator is currently initializing on this server."
        )

    try:
        val_pred = float(validator_model.predict(img_batch, verbose=0)[0][0])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during potato-leaf validation: {str(e)}"
        )

    # Scenario A: REJECT OTHER PLANTS & UNRELATED IMAGES (p < low_thresh)
    if val_pred < low_thresh:
        return {
            "success": False,
            "status": "REJECTED_NOT_POTATO",
            "title": "Potato Leaf Not Detected",
            "message": "This image does not appear to show a potato leaf. Potato Doctor AI currently supports potato leaves only. Please upload a clear image of a potato leaf.",
            "validation": {
                "is_potato_leaf": False,
                "confidence_raw": round(val_pred, 4),
                "confidence_pct": round(val_pred * 100, 2),
                "verdict": "Potato Leaf Not Detected",
                "threshold_required": high_thresh
            },
            "crop": "Potato",
            "image_metadata": {
                "filename": file.filename,
                "original_width": original_dims[0],
                "original_height": original_dims[1]
            }
        }

    # Scenario B: UNCERTAIN / INCONCLUSIVE LEAF (low_thresh <= p < high_thresh)
    if val_pred < high_thresh:
        return {
            "success": False,
            "status": "UNCERTAIN_LEAF",
            "title": "Unable to Verify Potato Leaf",
            "message": "We could not confidently verify that this is a potato leaf. Please upload a clear, well-lit image showing the leaf.",
            "validation": {
                "is_potato_leaf": False,
                "confidence_raw": round(val_pred, 4),
                "confidence_pct": round(val_pred * 100, 2),
                "verdict": "Unable to Verify Potato Leaf",
                "threshold_required": high_thresh
            },
            "crop": "Potato",
            "image_metadata": {
                "filename": file.filename,
                "original_width": original_dims[0],
                "original_height": original_dims[1]
            }
        }

    # -------------------------------------------------------------
    # 4. STAGE 2: POTATO DISEASE PREDICTION
    #    (Executes ONLY if Stage 1 passed with p >= high_thresh)
    # -------------------------------------------------------------
    disease_model = get_model()
    mapping = get_class_mapping()
    idx_to_class = {int(k): v for k, v in mapping["indices_to_classes"].items()}
    friendly_names = mapping["friendly_names"]

    if disease_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Potato Doctor AI disease model is currently initializing on this server."
        )

    try:
        preds = disease_model.predict(img_batch, verbose=0)[0]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during disease diagnosis: {str(e)}"
        )

    pred_idx = int(np.argmax(preds))
    predicted_class = idx_to_class.get(pred_idx, "Potato___Early_blight")
    disease_confidence = float(preds[pred_idx])

    probabilities = {}
    for i, p in enumerate(preds):
        c_id = idx_to_class.get(i, f"Class_{i}")
        f_name = friendly_names.get(c_id, c_id)
        probabilities[f_name] = round(float(p) * 100, 2)

    kb = get_knowledge_base()
    advice_data = kb.get(predicted_class, {})

    return {
        "success": True,
        "status": "VERIFIED_POTATO_LEAF",
        "validation": {
            "is_potato_leaf": True,
            "confidence_raw": round(val_pred, 4),
            "confidence_pct": round(val_pred * 100, 2),
            "verdict": "Verified Potato Leaf"
        },
        "crop": "Potato",
        "predicted_class": predicted_class,
        "disease_name": friendly_names.get(predicted_class, predicted_class),
        "confidence": round(disease_confidence * 100, 2),
        "confidence_raw": disease_confidence,
        "severity": advice_data.get("severity", "Unknown"),
        "scientific_name": advice_data.get("scientific_name", "Solanum tuberosum"),
        "probabilities": probabilities,
        "image_metadata": {
            "filename": file.filename,
            "original_width": original_dims[0],
            "original_height": original_dims[1]
        },
        "guidance": advice_data
    }

@app.post("/advice")
async def get_advice(payload: Dict[str, Any]):
    class_name = payload.get("predicted_class") or payload.get("class_name")
    if not class_name:
        raise HTTPException(status_code=400, detail="Missing 'predicted_class' in advice request.")

    kb = get_knowledge_base()
    if class_name not in kb:
        matching = [k for k in kb if class_name.lower() in k.lower()]
        class_name = matching[0] if matching else "Potato___Early_blight"

    base_advice = kb.get(class_name, {})
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if openai_api_key:
        try:
            import urllib.request
            req_data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a senior agricultural plant pathologist and certified potato agronomist. Provide actionable, concise, field-proven advice."
                    },
                    {
                        "role": "user",
                        "content": f"The farmer has detected {base_advice.get('name', class_name)} on their potato crop. Question: {payload.get('question', 'How should I treat this disease?')}"
                    }
                ],
                "temperature": 0.4
            }
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(req_data).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_api_key}"
                }
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                ai_text = res_json["choices"][0]["message"]["content"]
                return {
                    "source": "OpenAI Plant Pathologist",
                    "response": ai_text,
                    "reference": base_advice
                }
        except Exception as e:
            print(f"[Backend Warning] OpenAI call failed: {e}")

    # Fallback response using curated agricultural knowledge base
    return {
        "source": "Local Verified Knowledge Base",
        "response": (
            f"Official treatment protocol for {base_advice.get('name', class_name)}: "
            f"{' '.join(base_advice.get('immediate_actions', []))} "
            f"Cultural management: {' '.join(base_advice.get('cultural_management', []))} "
            f"Fungicide timing: {base_advice.get('chemical_options', {}).get('timing', 'Follow label instructions.')}"
        ),
        "reference": base_advice
    }
