import os
import sys
import json
import random
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
POTATO_DATASET_DIR = WORKSPACE_ROOT / "Potato dataset" / "Potato Diseases"
VALIDATION_DIR = WORKSPACE_ROOT / "validation_dataset"
REPORTS_DIR = WORKSPACE_ROOT / "reports"

random.seed(42)
np.random.seed(42)

HEADERS = {'User-Agent': 'PotatoDoctorAI/1.0 (dataset-curator)'}

def fetch_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode('utf-8'))

def download_file(url, target_path):
    if target_path.exists() and target_path.stat().st_size > 0:
        return True
    target_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp, open(target_path, 'wb') as out:
            out.write(resp.read())
        # Validate Pillow can open it
        with Image.open(target_path) as img:
            img.verify()
        return True
    except Exception as e:
        if target_path.exists():
            target_path.unlink()
        print(f"Warning: Failed to download {url}: {e}")
        return False

def curate_potato_samples(dest_dir, samples_per_class=140):
    dest_dir.mkdir(parents=True, exist_ok=True)
    classes = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
    collected = []

    for c in classes:
        c_dir = POTATO_DATASET_DIR / c
        images = list(c_dir.glob("*.JPG")) + list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png"))
        random.shuffle(images)
        count = min(samples_per_class if c != "Potato___healthy" else 130, len(images))
        selected = images[:count]

        for img_path in selected:
            target_name = f"{c}_{img_path.name}"
            target_file = dest_dir / target_name
            if not target_file.exists():
                with Image.open(img_path) as im:
                    im.convert("RGB").save(target_file, "JPEG", quality=95)
            collected.append(target_file)

    print(f"[Dataset] Collected {len(collected)} positive potato leaf samples.")
    return collected

def curate_other_plant_leaves(dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    collected = []

    # Target folders in PlantVillage
    target_categories = [
        ("Tomato___Early_blight", 25),
        ("Tomato___Late_blight", 25),
        ("Tomato___Leaf_Mold", 20),
        ("Tomato___healthy", 25),
        ("Pepper,_bell___Bacterial_spot", 25),
        ("Pepper,_bell___healthy", 25),
        ("Corn_(maize)___Common_rust_", 25),
        ("Corn_(maize)___healthy", 25),
        ("Apple___Apple_scab", 20),
        ("Grape___Black_rot", 20),
        ("Peach___Bacterial_spot", 20),
    ]

    for cat_name, count in target_categories:
        api_url = f"https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/{urllib.parse.quote(cat_name)}"
        try:
            items = fetch_json(api_url)
            file_items = [item for item in items if item.get("type") == "file" and item.get("name", "").lower().endswith((".jpg", ".jpeg", ".png"))]
            random.shuffle(file_items)
            chosen = file_items[:count]

            print(f"[Dataset] Downloading {len(chosen)} samples for {cat_name}...")
            for item in chosen:
                raw_url = item.get("download_url")
                if not raw_url:
                    continue
                safe_name = f"{cat_name.replace(',', '_')}_{item['name']}"
                target_path = dest_dir / safe_name
                if download_file(raw_url, target_path):
                    collected.append(target_path)
        except Exception as e:
            print(f"[Dataset Warning] Failed category {cat_name}: {e}")

    print(f"[Dataset] Collected {len(collected)} other plant leaf negative samples.")
    return collected

def curate_unrelated_objects(dest_dir, target_count=130):
    dest_dir.mkdir(parents=True, exist_ok=True)
    collected = []

    api_url = "https://api.github.com/repos/EliSchwartz/imagenet-sample-images/contents"
    try:
        items = fetch_json(api_url)
        img_items = [item for item in items if item.get("name", "").lower().endswith((".jpeg", ".jpg", ".png"))]
        random.shuffle(img_items)
        chosen = img_items[:target_count]

        print(f"[Dataset] Downloading {len(chosen)} unrelated object samples (animals, cars, objects)...")
        for item in chosen:
            raw_url = item.get("download_url")
            if not raw_url:
                continue
            target_path = dest_dir / f"unrelated_{item['name']}"
            if download_file(raw_url, target_path):
                collected.append(target_path)
    except Exception as e:
        print(f"[Dataset Warning] Failed fetching ImageNet samples: {e}")

    print(f"[Dataset] Collected {len(collected)} unrelated object negative samples.")
    return collected

def create_synthetic_and_edge_negatives(dest_dir, count=25):
    dest_dir.mkdir(parents=True, exist_ok=True)
    created = []

    # 1. Blank solid colors
    solids = [(255, 255, 255), (0, 0, 0), (128, 128, 128), (200, 220, 240), (40, 40, 40)]
    for i, rgb in enumerate(solids):
        p = dest_dir / f"edge_blank_solid_{i}.jpg"
        im = Image.new("RGB", (224, 224), rgb)
        im.save(p, "JPEG")
        created.append(p)

    # 2. Random noise patterns
    for i in range(5):
        p = dest_dir / f"edge_noise_{i}.jpg"
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(p, "JPEG")
        created.append(p)

    # 3. Document / text / screenshot simulations
    for i in range(5):
        p = dest_dir / f"edge_document_screenshot_{i}.jpg"
        im = Image.new("RGB", (224, 224), (250, 250, 250))
        draw = ImageDraw.Draw(im)
        for line_y in range(20, 200, 16):
            w = random.randint(80, 180)
            draw.line([(20, line_y), (20 + w, line_y)], fill=(50, 50, 50), width=3)
        draw.rectangle([(20, 20), (204, 204)], outline=(180, 180, 180), width=2)
        im.save(p, "JPEG")
        created.append(p)

    # 4. Colorful geometrical shapes & UI diagrams
    for i in range(5):
        p = dest_dir / f"edge_ui_diagram_{i}.jpg"
        im = Image.new("RGB", (224, 224), (240, 243, 246))
        draw = ImageDraw.Draw(im)
        draw.rectangle([(30, 30), (110, 80)], fill=(59, 130, 246))
        draw.rectangle([(120, 90), (200, 150)], fill=(239, 68, 68))
        draw.ellipse([(40, 120), (100, 180)], fill=(16, 185, 129))
        draw.line([(110, 55), (160, 90)], fill=(100, 100, 100), width=2)
        im.save(p, "JPEG")
        created.append(p)

    # 5. Gradient textures
    for i in range(5):
        p = dest_dir / f"edge_gradient_{i}.jpg"
        x = np.linspace(0, 1, 224)
        y = np.linspace(0, 1, 224)
        xx, yy = np.meshgrid(x, y)
        r = np.uint8(xx * 255)
        g = np.uint8(yy * 200)
        b = np.uint8((1.0 - xx) * 220)
        arr = np.stack([r, g, b], axis=-1)
        Image.fromarray(arr).save(p, "JPEG")
        created.append(p)

    print(f"[Dataset] Generated {len(created)} synthetic/edge negative images.")
    return created

def build_dataset_and_splits():
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    potato_dir = VALIDATION_DIR / "potato_leaf"
    non_potato_dir = VALIDATION_DIR / "not_potato_leaf"
    leaves_neg_dir = non_potato_dir / "other_plants"
    unrelated_neg_dir = non_potato_dir / "unrelated_objects"
    edge_neg_dir = non_potato_dir / "edge_cases"

    # Step 1: Collect positive potato samples
    potato_files = curate_potato_samples(potato_dir, samples_per_class=140)

    # Step 2: Collect negative non-potato leaves
    other_leaves = curate_other_plant_leaves(leaves_neg_dir)

    # Step 3: Collect unrelated object samples
    unrelated = curate_unrelated_objects(unrelated_neg_dir, target_count=130)

    # Step 4: Synthetic and edge negatives
    edge = create_synthetic_and_edge_negatives(edge_neg_dir, count=25)

    negative_files = other_leaves + unrelated + edge

    print(f"\nSummary:")
    print(f"Total Potato Leaf samples: {len(potato_files)}")
    print(f"Total Not Potato Leaf samples: {len(negative_files)}")
    print(f"  - Other plant leaves: {len(other_leaves)}")
    print(f"  - Unrelated objects: {len(unrelated)}")
    print(f"  - Edge/synthetic/blanks: {len(edge)}")

    # Prepare stratified records
    records_pos = [{"path": str(p.relative_to(WORKSPACE_ROOT)), "label": 1, "class": "potato_leaf", "subtype": "potato"} for p in potato_files]
    
    records_neg = []
    for p in other_leaves:
        records_neg.append({"path": str(p.relative_to(WORKSPACE_ROOT)), "label": 0, "class": "not_potato_leaf", "subtype": "other_plant_leaf"})
    for p in unrelated:
        records_neg.append({"path": str(p.relative_to(WORKSPACE_ROOT)), "label": 0, "class": "not_potato_leaf", "subtype": "unrelated_object"})
    for p in edge:
        records_neg.append({"path": str(p.relative_to(WORKSPACE_ROOT)), "label": 0, "class": "not_potato_leaf", "subtype": "edge_synthetic"})

    random.shuffle(records_pos)
    random.shuffle(records_neg)

    def split_list(items, train_ratio=0.70, val_ratio=0.15):
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]

    train_pos, val_pos, test_pos = split_list(records_pos)
    train_neg, val_neg, test_neg = split_list(records_neg)

    train_all = train_pos + train_neg
    val_all = val_pos + val_neg
    test_all = test_pos + test_neg

    random.shuffle(train_all)
    random.shuffle(val_all)
    random.shuffle(test_all)

    splits_data = {
        "train": train_all,
        "val": val_all,
        "test": test_all,
        "counts": {
            "total_potato_leaf": len(records_pos),
            "total_not_potato_leaf": len(records_neg),
            "train_total": len(train_all),
            "val_total": len(val_all),
            "test_total": len(test_all),
            "test_breakdown": {
                "potato_leaf": len(test_pos),
                "other_plant_leaf": sum(1 for r in test_neg if r["subtype"] == "other_plant_leaf"),
                "unrelated_object": sum(1 for r in test_neg if r["subtype"] == "unrelated_object"),
                "edge_synthetic": sum(1 for r in test_neg if r["subtype"] == "edge_synthetic")
            }
        }
    }

    out_splits = REPORTS_DIR / "validator_dataset_splits.json"
    with open(out_splits, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)

    print(f"\nSplits saved to {out_splits}")
    print(f"Train set: {len(train_all)} ({len(train_pos)} pos, {len(train_neg)} neg)")
    print(f"Val set:   {len(val_all)} ({len(val_pos)} pos, {len(val_neg)} neg)")
    print(f"Test set:  {len(test_all)} ({len(test_pos)} pos, {len(test_neg)} neg)")
    print(f"Test breakdown: {splits_data['counts']['test_breakdown']}")

if __name__ == "__main__":
    build_dataset_and_splits()
