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
MODELS_DIR = WORKSPACE_ROOT / "models"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent / "knowledge_base.json"

MODEL_PATH = MODELS_DIR / "potato_model.keras"
CLASS_MAPPING_PATH = MODELS_DIR / "class_mapping.json"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

IMG_HEIGHT = 224
IMG_WIDTH = 224

app = FastAPI(
    title="Potato Doctor AI API",
    description="Precision Potato Leaf Pathology & Crop Disease Diagnostic System",
    version="1.0.0"
)

# CORS configuration
origins_env = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,https://potato-doctor-ai.vercel.app")
allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins if allow_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
_tf_model = None
_class_mapping = None
_knowledge_base = None

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

def get_model():
    global _tf_model
    if _tf_model is None and MODEL_PATH.exists():
        try:
            import tensorflow as tf
            print(f"[Backend] Loading TensorFlow model from {MODEL_PATH}...")
            _tf_model = tf.keras.models.load_model(str(MODEL_PATH))
            print("[Backend] Model loaded successfully.")
        except Exception as e:
            print(f"[Backend Warning] Failed to load model: {e}")
            _tf_model = None
    return _tf_model

@app.on_event("startup")
def startup_event():
    get_class_mapping()
    get_knowledge_base()
    # Attempt lazy model preload
    get_model()

@app.get("/health")
def health_check():
    model = get_model()
    model_exists = MODEL_PATH.exists()
    model_size_mb = round(MODEL_PATH.stat().st_size / (1024 * 1024), 2) if model_exists else 0

    return {
        "status": "healthy",
        "service": "Potato Doctor AI Inference API",
        "version": "1.0.0",
        "model_file_exists": model_exists,
        "model_loaded": model is not None,
        "model_size_mb": model_size_mb,
        "target_crop": "Potato",
        "supported_classes": list(get_class_mapping().get("friendly_names", {}).values())
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
    # 1. Validation: File presence & extension
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

    # 2. Read bytes with size limit (15MB)
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

    # 3. PIL Image integrity & corruption check
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify() # Verify file integrity
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The uploaded image file is corrupt or unreadable: {str(e)}"
        )

    # Re-open for processing
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    original_dims = image.size # (width, height)

    # Simple green-ratio heuristic to warn if image is completely non-plant
    img_np_small = np.array(image.resize((64, 64)))
    r = img_np_small[:, :, 0].astype(float)
    g = img_np_small[:, :, 1].astype(float)
    b = img_np_small[:, :, 2].astype(float)
    total_rgb = r + g + b + 1e-6
    green_ratio = np.mean(g / total_rgb)
    is_likely_foliage = green_ratio > 0.28 or np.mean(g) > 40.0

    # 4. Resize and exact MobileNetV2 preprocessing
    img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.float32)

    # MobileNetV2 preprocessing: tf.keras.applications.mobilenet_v2.preprocess_input
    # maps [0, 255] -> [-1, 1] via (x / 127.5) - 1.0
    img_preprocessed = (img_arr / 127.5) - 1.0
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # 5. Run inference
    model = get_model()
    mapping = get_class_mapping()
    idx_to_class = {int(k): v for k, v in mapping["indices_to_classes"].items()}
    friendly_names = mapping["friendly_names"]

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Potato Doctor AI model is currently initializing or not yet trained on this server."
        )

    try:
        preds = model.predict(img_batch, verbose=0)[0]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during model prediction: {str(e)}"
        )

    # 6. Parse predictions
    pred_idx = int(np.argmax(preds))
    predicted_class = idx_to_class.get(pred_idx, "Potato___Early_blight")
    confidence = float(preds[pred_idx])

    probabilities = {}
    for i, p in enumerate(preds):
        c_id = idx_to_class.get(i, f"Class_{i}")
        f_name = friendly_names.get(c_id, c_id)
        probabilities[f_name] = round(float(p) * 100, 2)

    kb = get_knowledge_base()
    advice_data = kb.get(predicted_class, {})

    # Non-foliage warning flag
    relevance_warning = None
    if not is_likely_foliage and confidence < 0.85:
        relevance_warning = "Note: Image appears to have low foliage content. Please ensure the photograph clearly frames a potato leaf."

    return {
        "success": True,
        "crop": "Potato",
        "predicted_class": predicted_class,
        "disease_name": friendly_names.get(predicted_class, predicted_class),
        "confidence": round(confidence * 100, 2),
        "confidence_raw": confidence,
        "severity": advice_data.get("severity", "Unknown"),
        "scientific_name": advice_data.get("scientific_name", "Solanum tuberosum"),
        "probabilities": probabilities,
        "image_metadata": {
            "filename": file.filename,
            "original_width": original_dims[0],
            "original_height": original_dims[1],
            "relevance_warning": relevance_warning
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
        # Fallback to closest match
        matching = [k for k in kb if class_name.lower() in k.lower()]
        class_name = matching[0] if matching else "Potato___Early_blight"

    base_advice = kb.get(class_name, {})
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    # If OpenAI API Key is provided, enrich with AI Agronomist response
    if openai_api_key:
        try:
            import urllib.request
            user_question = payload.get("question", "Provide detailed actionable treatment and cultural management advice.")
            
            system_prompt = (
                "You are an expert plant pathologist and potato agronomist assistant at Potato Doctor AI. "
                f"The image classifier has confirmed the diagnosis as: {base_advice.get('common_name', class_name)} "
                f"(Pathogen: {base_advice.get('scientific_name', 'N/A')}).\n"
                "STRICT RULES:\n"
                "1. You must NEVER alter, second-guess, or fabricate the predicted diagnosis.\n"
                "2. Provide evidence-based, scientifically verified cultural and biological management practices.\n"
                "3. Do not invent unapproved pesticide formulations, home-brew chemical concoctions, or precise unregistered dosages.\n"
                "4. Always mention safety precautions, pre-harvest intervals, and consulting local agricultural extension agents.\n"
                "5. Format your response cleanly with bullet points and clear sections."
            )

            req_body = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"The leaf has been diagnosed with {base_advice.get('common_name')}. {user_question}"}
                ],
                "temperature": 0.3,
                "max_tokens": 600
            }

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(req_body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_api_key}"
                }
            )

            with urllib.request.urlopen(req, timeout=12) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                ai_text = res_data["choices"][0]["message"]["content"]
                
                return {
                    "source": "OpenAI Plant Pathology Consultant",
                    "ai_advice": ai_text,
                    "verified_knowledge": base_advice
                }
        except Exception as e:
            print(f"[Backend Warning] OpenAI call failed ({e}); falling back to verified agricultural knowledge base.")

    # Standard verified agricultural knowledge base return
    return {
        "source": "Verified Agricultural Extension Knowledge Base",
        "ai_advice": None,
        "verified_knowledge": base_advice
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting Potato Doctor AI backend on http://{host}:{port}...")
    uvicorn.run("backend.app:app", host=host, port=port, reload=True)
