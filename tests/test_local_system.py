import os
import sys
import json
import urllib.request
import urllib.error
import io
from pathlib import Path
from PIL import Image
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = WORKSPACE_ROOT / "frontend" / "public" / "samples"
MODELS_DIR = WORKSPACE_ROOT / "models"
API_URL = "http://127.0.0.1:8000"

def upload_file(file_path=None, file_bytes=None, filename="test.jpg", content_type="image/jpeg"):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    if file_bytes is None and file_path:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
    elif file_bytes is None:
        file_bytes = b""

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{API_URL}/predict",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}"
        }
    )
    return req

def run_tests():
    print("=" * 70)
    print("POTATO DOCTOR AI - COMPREHENSIVE INTEGRATION & VALIDATION VERIFICATION")
    print("=" * 70)

    # 1. Health check & model separation verification
    print("\n[TEST 1] GET /health (Verify Separate Models & Operational Status)")
    req = urllib.request.Request(f"{API_URL}/health")
    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read().decode())
        print(f"  Response: {json.dumps(data, indent=2)}")
        assert data["status"] == "healthy"
        assert "disease_classifier" in data
        assert "potato_validator" in data
        assert data["disease_classifier"]["loaded"] is True
        assert data["potato_validator"]["loaded"] is True
        assert data["potato_validator"]["thresholds"]["high_confidence_potato"] >= 0.75
        print("  [PASS]: Both independent models (Disease Classifier + Leaf Validator) loaded and operational.")

    # 2. Classes check
    print("\n[TEST 2] GET /classes (Active Potato Scope & Future Crop Guards)")
    req = urllib.request.Request(f"{API_URL}/classes")
    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read().decode())
        crops = {c["crop"]: c for c in data["crops"]}
        assert crops["Potato"]["status"] == "Available"
        assert crops["Potato"]["is_active"] is True
        assert crops["Tomato"]["status"] == "Coming Soon"
        assert crops["Tomato"]["is_active"] is False
        assert crops["Corn (Maize)"]["status"] == "Coming Soon"
        assert crops["Pepper (Bell)"]["status"] == "Coming Soon"
        print("  [PASS]: Only Potato is active; other crops locked with Coming Soon status.")

    # 3. Test Verified Potato Leaf: Early Blight
    print("\n[TEST 3] POST /predict - Real Potato Early Blight Specimen")
    req = upload_file(file_path=SAMPLES_DIR / "early_blight.JPG", filename="early_blight.JPG")
    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read().decode())
        print(f"  Status: {data.get('status')}")
        print(f"  Validation: {data.get('validation')}")
        print(f"  Diagnosis: {data.get('disease_name')} ({data.get('confidence')}%)")
        assert data["success"] is True
        assert data["status"] == "VERIFIED_POTATO_LEAF"
        assert data["validation"]["is_potato_leaf"] is True
        assert data["predicted_class"] == "Potato___Early_blight"
        assert data["confidence"] > 80.0
        assert "guidance" in data
        print("  [PASS]: Validated as authentic potato leaf and accurately diagnosed Early Blight.")

    # 4. Test Verified Potato Leaf: Late Blight
    print("\n[TEST 4] POST /predict - Real Potato Late Blight Specimen")
    req = upload_file(file_path=SAMPLES_DIR / "late_blight.JPG", filename="late_blight.JPG")
    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read().decode())
        print(f"  Status: {data.get('status')}")
        print(f"  Diagnosis: {data.get('disease_name')} ({data.get('confidence')}%) - Severity: {data.get('severity')}")
        assert data["success"] is True
        assert data["status"] == "VERIFIED_POTATO_LEAF"
        assert data["validation"]["is_potato_leaf"] is True
        assert data["predicted_class"] == "Potato___Late_blight"
        assert data["severity"] == "CRITICAL / SEVERE"
        print("  [PASS]: Validated as authentic potato leaf and diagnosed Late Blight severity.")

    # 5. Test Verified Potato Leaf: Healthy
    print("\n[TEST 5] POST /predict - Real Potato Healthy Leaf Specimen")
    req = upload_file(file_path=SAMPLES_DIR / "healthy.JPG", filename="healthy.JPG")
    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read().decode())
        print(f"  Status: {data.get('status')}")
        print(f"  Diagnosis: {data.get('disease_name')} ({data.get('confidence')}%)")
        assert data["success"] is True
        assert data["status"] == "VERIFIED_POTATO_LEAF"
        assert data["validation"]["is_potato_leaf"] is True
        assert data["predicted_class"] == "Potato___healthy"
        print("  [PASS]: Validated as authentic potato leaf and diagnosed Healthy.")

    # 6. Reject Other Plant Leaf: Tomato Leaf
    print("\n[TEST 6] POST /predict - Non-Potato Plant Leaf (Tomato Leaf)")
    req = upload_file(file_path=SAMPLES_DIR / "tomato_leaf.JPG", filename="tomato_leaf.JPG")
    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read().decode())
        print(f"  Status: {data.get('status')}")
        print(f"  Title: {data.get('title')}")
        print(f"  Message: {data.get('message')}")
        assert data["success"] is False
        assert data["status"] == "REJECTED_NOT_POTATO"
        assert data["title"] == "Potato Leaf Not Detected"
        assert data["message"] == "This image does not appear to show a potato leaf. Potato Doctor AI currently supports potato leaves only. Please upload a clear image of a potato leaf."
        assert "predicted_class" not in data
        assert "disease_name" not in data
        assert data["validation"]["is_potato_leaf"] is False
        print("  [PASS]: Non-potato leaf rejected with exact required 'Potato Leaf Not Detected' notification.")

    # 7. Reject Unrelated Object: Animal / Vehicle / Everyday Object
    print("\n[TEST 7] POST /predict - Unrelated Object (Non-Plant)")
    req = upload_file(file_path=SAMPLES_DIR / "unrelated_object.JPEG", filename="unrelated_object.JPEG")
    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read().decode())
        print(f"  Status: {data.get('status')}")
        print(f"  Title: {data.get('title')}")
        assert data["success"] is False
        assert data["status"] == "REJECTED_NOT_POTATO"
        assert data["title"] == "Potato Leaf Not Detected"
        assert "predicted_class" not in data
        print("  [PASS]: Unrelated object rejected before disease model execution.")

    # 8. Uncertain / Borderline Specimen
    print("\n[TEST 8] POST /predict - Unclear / Ambiguous Leaf Specimen")
    req = upload_file(file_path=SAMPLES_DIR / "unclear_leaf.JPG", filename="unclear_leaf.JPG")
    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read().decode())
        print(f"  Status: {data.get('status')}")
        print(f"  Title: {data.get('title')}")
        print(f"  Message: {data.get('message')}")
        assert data["success"] is False
        assert data["status"] == "UNCERTAIN_LEAF"
        assert data["title"] == "Unable to Verify Potato Leaf"
        assert data["message"] == "We could not confidently verify that this is a potato leaf. Please upload a clear, well-lit image showing the leaf."
        assert "predicted_class" not in data
        print("  [PASS]: Borderline specimen safely stopped with 'Unable to Verify Potato Leaf'.")

    # 9. Invalid Uploads: Unsupported File Type (.txt)
    print("\n[TEST 9] POST /predict - Unsupported File Format (.txt)")
    req = upload_file(file_bytes=b"hello potato world", filename="specimen.txt", content_type="text/plain")
    try:
        urllib.request.urlopen(req)
        assert False, "Should have returned HTTP 400"
    except urllib.error.HTTPError as e:
        assert e.code == 400
        err_msg = json.loads(e.read().decode())["detail"]
        print(f"  HTTP 400 Detail: {err_msg}")
        assert "Unsupported file format" in err_msg
        print("  [PASS]: Unsupported extension rejected with HTTP 400.")

    # 10. Invalid Uploads: Empty File (0 Bytes)
    print("\n[TEST 10] POST /predict - Empty File (0 Bytes)")
    req = upload_file(file_bytes=b"", filename="empty.jpg", content_type="image/jpeg")
    try:
        urllib.request.urlopen(req)
        assert False, "Should have returned HTTP 400"
    except urllib.error.HTTPError as e:
        assert e.code == 400
        err_msg = json.loads(e.read().decode())["detail"]
        print(f"  HTTP 400 Detail: {err_msg}")
        assert "empty (0 bytes)" in err_msg
        print("  [PASS]: 0-byte upload rejected with HTTP 400.")

    # 11. Invalid Uploads: Corrupt Image Bytes
    print("\n[TEST 11] POST /predict - Corrupt Image File")
    req = upload_file(file_bytes=b"NOT_A_VALID_JPEG_HEADER_CORRUPT_BYTES", filename="corrupt.jpg", content_type="image/jpeg")
    try:
        urllib.request.urlopen(req)
        assert False, "Should have returned HTTP 400"
    except urllib.error.HTTPError as e:
        assert e.code == 400
        err_msg = json.loads(e.read().decode())["detail"]
        print(f"  HTTP 400 Detail: {err_msg}")
        assert "corrupt or unreadable" in err_msg
        print("  [PASS]: Corrupted image rejected with HTTP 400.")

    # 12. Invalid Uploads: Blank / Solid Color Image
    print("\n[TEST 12] POST /predict - Blank Solid White Image")
    blank_im = Image.new("RGB", (224, 224), (255, 255, 255))
    buf = io.BytesIO()
    blank_im.save(buf, format="JPEG")
    req = upload_file(file_bytes=buf.getvalue(), filename="blank_white.jpg", content_type="image/jpeg")
    try:
        urllib.request.urlopen(req)
        assert False, "Should have returned HTTP 400"
    except urllib.error.HTTPError as e:
        assert e.code == 400
        err_msg = json.loads(e.read().decode())["detail"]
        print(f"  HTTP 400 Detail: {err_msg}")
        assert "blank, solid, or uninformative" in err_msg
        print("  [PASS]: Blank solid image rejected with HTTP 400.")

    # 13. Verify Separate Models on Filesystem
    print("\n[TEST 13] Filesystem Verification - Separate Models & Configs")
    assert (MODELS_DIR / "potato_model.keras").exists(), "Disease classifier must exist"
    assert (MODELS_DIR / "potato_validator.keras").exists(), "Validation model must exist"
    assert (MODELS_DIR / "validator_config.json").exists(), "Validation config must exist"
    print("  Disease Model:   ", MODELS_DIR / "potato_model.keras", f"({round((MODELS_DIR / 'potato_model.keras').stat().st_size / (1024*1024), 2)} MB)")
    print("  Validator Model: ", MODELS_DIR / "potato_validator.keras", f"({round((MODELS_DIR / 'potato_validator.keras').stat().st_size / (1024*1024), 2)} MB)")
    print("  [PASS]: Models are separate files with dedicated configurations.")

    print("\n" + "=" * 70)
    print("ALL 13 INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
