import os
import sys
import json
import hashlib
from pathlib import Path
from PIL import Image
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = WORKSPACE_ROOT / "Potato dataset" / "Potato Diseases"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
MODELS_DIR = WORKSPACE_ROOT / "models"

EXPECTED_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

FRIENDLY_NAMES = {
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___Late_blight": "Potato Late Blight",
    "Potato___healthy": "Potato Healthy"
}

def calculate_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def inspect_dataset():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("POTATO DOCTOR AI - DATASET AUDIT & PREPARATION")
    print("=" * 60)
    print(f"Dataset root: {DATASET_DIR}")

    if not DATASET_DIR.exists():
        print(f"ERROR: Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    subdirs = [d.name for d in DATASET_DIR.iterdir() if d.is_dir()]
    print(f"Found subdirectories: {subdirs}")

    missing_classes = [c for c in EXPECTED_CLASSES if c not in subdirs]
    if missing_classes:
        print(f"ERROR: Missing expected classes: {missing_classes}")
        sys.exit(1)

    dataset_summary = {
        "dataset_path": str(DATASET_DIR),
        "total_images": 0,
        "valid_images": 0,
        "corrupt_images": 0,
        "classes": {},
        "duplicates_found": 0,
        "resolutions": {},
        "channels": {},
        "formats": {}
    }

    all_file_records = []
    seen_hashes = {}
    duplicates = []

    for class_name in EXPECTED_CLASSES:
        class_folder = DATASET_DIR / class_name
        class_files = [f for f in class_folder.iterdir() if f.is_file()]
        
        valid_in_class = []
        corrupt_in_class = []

        print(f"\nAuditing class: {class_name} ({FRIENDLY_NAMES[class_name]})...")

        for img_path in class_files:
            dataset_summary["total_images"] += 1
            # Check format & readability
            try:
                with Image.open(img_path) as img:
                    img.verify() # Check integrity
                
                # Re-open for mode and size inspection
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode
                    fmt = img.format or "UNKNOWN"

                    res_key = f"{width}x{height}"
                    dataset_summary["resolutions"][res_key] = dataset_summary["resolutions"].get(res_key, 0) + 1
                    dataset_summary["channels"][mode] = dataset_summary["channels"].get(mode, 0) + 1
                    dataset_summary["formats"][fmt] = dataset_summary["formats"].get(fmt, 0) + 1

                # Check duplicates via hash
                file_hash = calculate_sha256(img_path)
                if file_hash in seen_hashes:
                    duplicates.append({
                        "file": str(img_path.relative_to(WORKSPACE_ROOT)),
                        "duplicate_of": seen_hashes[file_hash]
                    })
                else:
                    seen_hashes[file_hash] = str(img_path.relative_to(WORKSPACE_ROOT))

                rel_path = str(img_path.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
                record = {
                    "path": rel_path,
                    "class": class_name,
                    "friendly_name": FRIENDLY_NAMES[class_name],
                    "hash": file_hash,
                    "width": width,
                    "height": height,
                    "mode": mode
                }
                valid_in_class.append(record)
                all_file_records.append(record)

            except Exception as e:
                print(f"  [CORRUPT/UNREADABLE] {img_path.name}: {e}")
                corrupt_in_class.append(str(img_path.name))
                dataset_summary["corrupt_images"] += 1

        dataset_summary["classes"][class_name] = {
            "friendly_name": FRIENDLY_NAMES[class_name],
            "total_files": len(class_files),
            "valid_images": len(valid_in_class),
            "corrupt_images": len(corrupt_in_class)
        }
        dataset_summary["valid_images"] += len(valid_in_class)
        print(f"  Valid: {len(valid_in_class)}, Corrupt: {len(corrupt_in_class)}")

    dataset_summary["duplicates_found"] = len(duplicates)
    print(f"\nTotal scanned images: {dataset_summary['total_images']}")
    print(f"Total valid images: {dataset_summary['valid_images']}")
    print(f"Total corrupt images: {dataset_summary['corrupt_images']}")
    print(f"Exact duplicates detected: {len(duplicates)}")

    # Stratified Train (70%), Val (15%), Test (15%) split
    print("\nCreating stratified splits (70% train, 15% validation, 15% test)...")
    np.random.seed(42)

    splits = {
        "train": [],
        "val": [],
        "test": []
    }

    split_counts = {
        "train": {c: 0 for c in EXPECTED_CLASSES},
        "val": {c: 0 for c in EXPECTED_CLASSES},
        "test": {c: 0 for c in EXPECTED_CLASSES}
    }

    for class_name in EXPECTED_CLASSES:
        class_records = [r for r in all_file_records if r["class"] == class_name]
        indices = np.random.permutation(len(class_records))
        
        n_total = len(class_records)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        n_test = n_total - n_train - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        for i in train_idx:
            splits["train"].append(class_records[i])
            split_counts["train"][class_name] += 1

        for i in val_idx:
            splits["val"].append(class_records[i])
            split_counts["val"][class_name] += 1

        for i in test_idx:
            splits["test"].append(class_records[i])
            split_counts["test"][class_name] += 1

    dataset_summary["split_counts"] = split_counts
    dataset_summary["split_totals"] = {
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"])
    }

    print("Splits summary:")
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name.upper()}: {len(splits[split_name])} images {split_counts[split_name]}")

    # Write summary and splits
    with open(REPORTS_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2)

    with open(REPORTS_DIR / "dataset_splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    # Class mapping
    class_mapping = {
        "indices_to_classes": {str(i): c for i, c in enumerate(EXPECTED_CLASSES)},
        "classes_to_indices": {c: i for i, c in enumerate(EXPECTED_CLASSES)},
        "friendly_names": FRIENDLY_NAMES
    }
    with open(MODELS_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)

    print(f"\nSaved reports to {REPORTS_DIR}")
    print(f"Saved class mapping to {MODELS_DIR / 'class_mapping.json'}")
    print("=" * 60)

if __name__ == "__main__":
    inspect_dataset()
