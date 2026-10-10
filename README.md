# Potato Doctor AI 🥔🩺
### Precision Potato Leaf Pathology & Agronomic Defense System

An AI-powered diagnostic and evidence-based crop management web platform designed to detect Potato foliage diseases with high confidence and provide actionable recovery protocols based on agricultural extension literature.

---

## 🌟 Key Capabilities

- **Native Dataset Training**: Trained directly on the existing workspace dataset in `Potato dataset/Potato Diseases/` without redownloading or moving files.
- **Strict Potato-Leaf Botanical Validation**: Two-stage sequential diagnostic workflow. An independent binary classifier (`models/potato_validator.keras`, MobileNetV2, ROC AUC 0.9971) screens uploads to verify *Solanum tuberosum* foliage before running disease classification.
- **Strict Crop Scope & False Inference Defense**:
  - **Potato**: Disease detection across **Early Blight**, **Late Blight**, and **Healthy** foliage executes **only** when verified as a potato leaf.
  - **Other Crops (Tomato, Corn, Pepper, etc.)**: Prediction is halted with: **"Potato Leaf Not Detected"** (*“This image does not appear to show a potato leaf. Potato Doctor AI currently supports potato leaves only. Please upload a clear image of a potato leaf.”*).
  - **Uncertain / Blurry Uploads**: Prediction is halted with: **"Unable to Verify Potato Leaf"** (*“We could not confidently verify that this is a potato leaf. Please upload a clear, well-lit image showing the leaf.”*).
  - **Invalid Uploads**: Blank images, non-plant objects, and corrupt uploads are safely rejected.
  - **Upcoming Crops**: Tomato, Corn, and Pepper remain guarded as **Coming Soon** in the interface.
- **Deep Learning Backbone**: MobileNetV2 transfer learning with inverted residual feature extraction, batch normalization, dropout regularization, and dense classification head.
- **Verified Agricultural Guidance**: Curated pathology treatment plans (Immediate 24-48h steps, Cultural management, FRAC-compliant fungicides, Biocontrol options) with optional OpenAI AI Agronomist enrichment.
- **Modern Responsive Web UI**: Dark emerald glassmorphic interface with interactive scanner animation, sample leaf presets, probability breakdown bars, and transparency telemetry.
- **Vercel Deployable**: Configured for seamless deployment on Vercel (`vercel.json` + Vite frontend) and decoupled Python inference API.

---

## 📊 Dataset Audit & Class Distribution

Audited via `training/inspect_dataset.py`:
- **Total Valid Images**: 2,152 (256x256 RGB JPEG)
- **Corrupted / Unreadable Images**: 0
- **Duplicates Detected**: 0

| Class Label | Folder Name | Image Count | Stratified Train (70%) | Validation (15%) | Held-Out Test (15%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Potato Early Blight** | `Potato___Early_blight` | 1,000 | 700 | 150 | 150 |
| **Potato Late Blight** | `Potato___Late_blight` | 1,000 | 700 | 150 | 150 |
| **Potato Healthy** | `Potato___healthy` | 152 | 106 | 22 | 24 |
| **Total** | | **2,152** | **1,506** | **322** | **324** |

*Note: Inverse class weighting (Early Blight: 0.717, Late Blight: 0.717, Healthy: 4.736) is applied during training to handle the natural representation of healthy leaves.*

---

## 🧠 Model Architecture

- **Base Network**: `MobileNetV2` (ImageNet pre-trained, top layers fine-tuned)
- **Input Resolution**: `224x224x3`
- **Data Augmentation**: Random flip, random rotation (0.15), random zoom (0.15), translation (0.1), and contrast jittering (training set only)
- **Loss Function**: `SparseCategoricalCrossentropy` with class weights
- **Optimization Strategy**:
  - Phase 1: Feature Extraction with Adam (`lr=1e-3`)
  - Phase 2: Top-block fine-tuning with Adam (`lr=1e-4`)
  - Callbacks: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`

---

## 🚀 Getting Started Locally

### 1. Prerequisites
- Python 3.10+ (with TensorFlow, FastAPI, Uvicorn)
- Node.js 18+ and npm

### 2. Backend Setup
```bash
# Install backend dependencies
py -m pip install -r backend/requirements.txt

# Run the FastAPI server
py backend/app.py
# Server runs at: http://localhost:8000
# OpenAPI Docs: http://localhost:8000/docs
```

### 3. Frontend Setup
```bash
# In another terminal:
cd frontend
npm install
npm run dev
# Frontend runs at: http://localhost:5173
```

---

## 🌐 API Endpoints

- `GET /health`: Health check and model loaded status.
- `GET /classes`: Returns crop portfolio and supported classes.
- `POST /predict`: Upload image file (`multipart/form-data`) -> returns disease classification, confidence, probabilities, and guidance.
- `POST /advice`: Request customized agronomy advice powered by OpenAI API (if configured) or the curated extension knowledge base.

---

## ☁️ Deployment Architecture

### 1. Backend Inference Server (Render / Railway / Docker)
Because TensorFlow / Keras deep learning weights exceed Vercel's 250MB serverless limit, the Python FastAPI backend is containerized:
- **Render (Recommended Free Tier)**:
  - Connect your GitHub repo to [Render](https://render.com).
  - Select **Web Service** -> Docker (or Render Blueprint via [`render.yaml`](file:///d:/SMIT%20FYP/render.yaml)).
  - Build command uses [`Dockerfile`](file:///d:/SMIT%20FYP/Dockerfile).
  - Copy your live backend URL (e.g., `https://plant-doctor-backend.onrender.com`).
- **Railway / Fly.io / VPS**:
  - Deploy using the provided [`Dockerfile`](file:///d:/SMIT%20FYP/Dockerfile) or [`Procfile`](file:///d:/SMIT%20FYP/Procfile).

### 2. Frontend (Vercel)
1. Deploy frontend to Vercel via GitHub integration.
2. In Vercel Project Settings -> **Environment Variables**:
   - `VITE_API_BASE_URL`: `https://your-backend-service.onrender.com`
3. If no environment variable is set or you need to switch backend servers on the fly, click the **Status Pill / Settings** icon in the navbar of the live app to test and connect any live API URL dynamically.

---

## ⚖️ Diagnostic Disclaimer
Diagnostic guidance is generated using machine vision and agricultural extension literature. Field conditions, environmental variables, and pest complexes can cause visual ambiguity. Always verify diagnoses with local certified crop consultants before purchasing or applying commercial pesticides.
