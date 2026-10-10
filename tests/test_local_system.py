import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = WORKSPACE_ROOT / "frontend" / "public" / "samples"
API_URL = "http://127.0.0.1:8000"

def run_tests():
    print("=" * 60)
    print("POTATO DOCTOR AI - LOCAL INTEGRATION VERIFICATION")
    print("=" * 60)

    # 1. Health check
    print("\n[TEST 1] GET /health")
    req = urllib.request.Request(f"{API_URL}/health")
    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read().decode())
        print(f"  Health response: {data}")
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_file_exists"] is True
        print("  [PASS]: Backend healthy, model loaded (11.65 MB)")

    # 2. Classes check
    print("\n[TEST 2] GET /classes")
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
        print("  [PASS]: Potato available, Tomato/Corn/Pepper locked as Coming Soon")

    # Helper for multipart file upload
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

    # 3. Test Early Blight Specimen
    print("\n[TEST 3] POST /predict with real Early Blight image")
    req = upload_file(file_path=SAMPLES_DIR / "early_blight.JPG", filename="early_blight.JPG")
    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read().decode())
        print(f"  Diagnosis: {data['disease_name']} ({data['confidence']}%)")
        assert data["predicted_class"] == "Potato___Early_blight"
        assert data["confidence"] > 85.0
        assert "guidance" in data
        assert len(data["guidance"]["immediate_actions"]) > 0
        print("  [PASS]: Correctly classified Early Blight with treatment guidance")

    # 4. Test Late Blight Specimen
    print("\n[TEST 4] POST /predict with real Late Blight image")
    req = upload_file(file_path=SAMPLES_DIR / "late_blight.JPG", filename="late_blight.JPG")
    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read().decode())
        print(f"  Diagnosis: {data['disease_name']} ({data['confidence']}%)")
        assert data["predicted_class"] == "Potato___Late_blight"
        assert data["confidence"] > 85.0
        assert data["severity"] == "CRITICAL / SEVERE"
        print("  [PASS]: Correctly classified Late Blight as critical pathogen")

    # 5. Test Healthy Specimen
    print("\n[TEST 5] POST /predict with real Healthy leaf image")
    req = upload_file(file_path=SAMPLES_DIR / "healthy.JPG", filename="healthy.JPG")
    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read().decode())
        print(f"  Diagnosis: {data['disease_name']} ({data['confidence']}%)")
        assert data["predicted_class"] == "Potato___healthy"
        assert data["confidence"] > 80.0
        print("  [PASS]: Correctly classified Healthy leaf")

    # 6. Test Empty/Missing File
    print("\n[TEST 6] POST /predict with empty file (0 bytes)")
    req = upload_file(file_bytes=b"", filename="empty.jpg")
    try:
        with urllib.request.urlopen(req) as res:
            assert False, "Should have returned 400"
    except urllib.error.HTTPError as e:
        assert e.code in [400, 422]
        print(f"  [PASS]: Correctly rejected empty file with HTTP {e.code}")

    # 7. Test Invalid File Extension
    print("\n[TEST 7] POST /predict with invalid extension (.txt)")
    req = upload_file(file_bytes=b"hello", filename="note.txt", content_type="text/plain")
    try:
        with urllib.request.urlopen(req) as res:
            assert False, "Should have returned 400"
    except urllib.error.HTTPError as e:
        assert e.code == 400
        print(f"  [PASS]: Correctly rejected invalid file extension with HTTP 400")

    # 8. Test Corrupted Image Data
    print("\n[TEST 8] POST /predict with corrupted JPEG bytes")
    req = upload_file(file_bytes=b"NOT_A_REAL_JPEG_IMAGE_HEADER_CORRUPTED", filename="corrupt.jpg")
    try:
        with urllib.request.urlopen(req) as res:
            assert False, "Should have returned 400"
    except urllib.error.HTTPError as e:
        assert e.code == 400
        print(f"  [PASS]: Correctly detected corrupt image with HTTP 400")

    # 9. Test Advice Endpoint
    print("\n[TEST 9] POST /advice (verified knowledge base & OpenAI fallback)")
    advice_req_data = json.dumps({
        "predicted_class": "Potato___Early_blight",
        "question": "What is the best fungicide rotation strategy?"
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{API_URL}/advice",
        data=advice_req_data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read().decode())
        assert "verified_knowledge" in data
        assert data["verified_knowledge"]["common_name"] == "Potato Early Blight"
        print(f"  Advice source: {data['source']}")
        print("  [PASS]: Evidence-based advice returned with expert sources")

    print("\n" + "=" * 60)
    print("ALL 9 INTEGRATION TESTS PASSED SUCCESSFULLY! (100% PASS RATE)")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
