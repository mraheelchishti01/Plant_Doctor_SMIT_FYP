# Potato Doctor AI 🥔🩺
### Precision Potato Leaf Pathology & Agronomic Defense System

An AI-powered diagnostic and evidence-based crop management platform designed to detect Potato foliage diseases with high confidence, verify botanical species identity, and provide actionable recovery protocols based on agricultural extension literature.

---

## 🌟 Key Capabilities

- **Native Dataset Training**: Trained directly on native agricultural datasets in `Potato dataset/Potato Diseases/` without external data augmentation dependencies.
- **Strict Potato-Leaf Botanical Validation**: Two-stage sequential diagnostic workflow. An independent binary classifier (`models/potato_validator.keras`, MobileNetV2, ROC AUC 0.9971) screens uploads to verify *Solanum tuberosum* foliage before running disease classification.
- **4 Crop Portfolio Cards**:
  - 🥔 **Potato** *(Solanum tuberosum)*: Active studio with 3-class pathology (Early Blight, Late Blight, Healthy).
  - 🍅 **Tomato** *(Solanum lycopersicum)*: Guarded by botanical validator against false inferences.
  - 🌽 **Corn / Maize** *(Zea mays)*: Guarded by botanical validator against cross-species classification.
  - 🫑 **Bell Pepper** *(Capsicum annuum)*: Guarded by botanical validator against misdiagnosis.
- **Strict Crop Scope & False Inference Defense**:
  - **Potato**: Disease detection across **Early Blight**, **Late Blight**, and **Healthy** foliage executes **only** when verified as a potato leaf.
  - **Other Crops (Tomato, Corn, Pepper, etc.)**: Prediction is halted with: **"Potato Leaf Not Detected"** (*“This image does not appear to show a potato leaf. Potato Doctor AI currently supports potato leaves only. Please upload a clear image of a potato leaf.”*).
  - **Uncertain / Blurry Uploads**: Prediction is halted with: **"Unable to Verify Potato Leaf"** (*“We could not confidently verify that this is a potato leaf. Please upload a clear, well-lit image showing the leaf.”*).
  - **Invalid Uploads**: Blank images, non-plant objects, and corrupt uploads are safely rejected.
- **Deep Learning Backbone**: MobileNetV2 transfer learning with inverted residual feature extraction, batch normalization, dropout regularization, and dense classification head (**97.22% held-out test accuracy**).
- **Verified Agricultural Guidance**: Curated pathology treatment plans (Immediate 24-48h steps, Cultural management, FRAC-compliant fungicides, Biocontrol options).
- **Modern Dark Glassmorphic Streamlit Frontend**: Polished interface with custom CSS tokens, animated badges, 4 crop cards, 1-click calibrated sample presets, live camera capture, and transparency telemetry.
- **Streamlit Community Cloud Native**: Fully optimized for one-click deployment on Streamlit Cloud via `streamlit_app.py` and root `requirements.txt`.

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

---

## 🧠 Model Architecture

- **Base Network**: `MobileNetV2` (ImageNet pre-trained, top layers fine-tuned)
- **Input Resolution**: `224x224x3`
- **Botanical Validator**: Binary MobileNetV2 (`potato_validator.keras`, 23.5 MB)
- **Disease Classifier**: 3-Class MobileNetV2 (`potato_model.keras`, 21.5 MB)
- **Operational Thresholds**: High confidence $\ge 80\%$, Low rejection $< 35\%$

---

## 🚀 Running Locally with Streamlit

### 1. Prerequisites
- Python 3.10+ (with TensorFlow, Streamlit, Pillow, NumPy)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/mraheelchishti01/Plant_Doctor_SMIT_FYP.git
cd Plant_Doctor_SMIT_FYP

# Install requirements
pip install -r requirements.txt
```

### 3. Launch Streamlit Application
```bash
streamlit run streamlit_app.py
```
App will open automatically in your browser at: `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud

Deploying Potato Doctor AI to Streamlit Cloud is seamless:

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy Potato Doctor AI on Streamlit"
   git push origin main
   ```
2. **Go to [share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
3. **Click "New app"** and configure:
   - **Repository**: `mraheelchishti01/Plant_Doctor_SMIT_FYP`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. **Click "Deploy!"**
   - Streamlit Cloud will automatically install dependencies from `requirements.txt`, load the cached MobileNetV2 models from `models/`, and provide your permanent public live URL (e.g., `https://plant-doctor-smit.streamlit.app`).

---

## ⚖️ Diagnostic Disclaimer
Diagnostic guidance is generated using machine vision and agricultural extension literature. Field conditions, environmental variables, and pest complexes can cause visual ambiguity. Always verify diagnoses with local certified crop consultants before purchasing or applying commercial pesticides.

---

**Developed by:** Muhammad Raheel Chishti  
**Program:** SMIT Final Year Project (FYP)  
**System:** Potato Doctor AI • Deep Botanical Pathology
