import shutil
import json
from pathlib import Path
from PIL import Image, ImageFilter

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
PUB_SAMPLES = WORKSPACE_ROOT / "samples"
VAL_DIR = WORKSPACE_ROOT / "validation_dataset"

PUB_SAMPLES.mkdir(parents=True, exist_ok=True)

# 1. Copy Tomato Leaf
tomatoes = list((VAL_DIR / "not_potato_leaf" / "other_plants").glob("*Tomato*"))
if tomatoes:
    shutil.copy(tomatoes[0], PUB_SAMPLES / "tomato_leaf.JPG")
    print(f"Copied tomato leaf: {tomatoes[0].name}")

# 2. Copy Unrelated Object
unrelated = list((VAL_DIR / "not_potato_leaf" / "unrelated_objects").glob("*"))
if unrelated:
    shutil.copy(unrelated[0], PUB_SAMPLES / "unrelated_object.JPEG")
    print(f"Copied unrelated object: {unrelated[0].name}")

# 3. Create unclear/blurry specimen
with Image.open(PUB_SAMPLES / "early_blight.JPG") as im:
    blurry = im.filter(ImageFilter.GaussianBlur(radius=7))
    blurry.save(PUB_SAMPLES / "unclear_leaf.JPG", "JPEG", quality=90)
    print("Generated unclear_leaf.JPG")

samples = [
    {
        "name": "early_blight.JPG",
        "label": "Potato Early Blight",
        "badge": "Potato Leaf",
        "src": "/samples/early_blight.JPG",
        "category": "potato"
    },
    {
        "name": "late_blight.JPG",
        "label": "Potato Late Blight",
        "badge": "Potato Leaf",
        "src": "/samples/late_blight.JPG",
        "category": "potato"
    },
    {
        "name": "healthy.JPG",
        "label": "Potato Healthy Leaf",
        "badge": "Potato Leaf",
        "src": "/samples/healthy.JPG",
        "category": "potato"
    },
    {
        "name": "tomato_leaf.JPG",
        "label": "Tomato Leaf (Non-Potato)",
        "badge": "Species Rejected",
        "src": "/samples/tomato_leaf.JPG",
        "category": "negative"
    },
    {
        "name": "unrelated_object.JPEG",
        "label": "Unrelated Non-Plant",
        "badge": "Invalid Target",
        "src": "/samples/unrelated_object.JPEG",
        "category": "negative"
    },
    {
        "name": "unclear_leaf.JPG",
        "label": "Unclear / Blurry Leaf",
        "badge": "Inconclusive Quality",
        "src": "/samples/unclear_leaf.JPG",
        "category": "uncertain"
    }
]

with open(PUB_SAMPLES / "samples.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2)

print(f"Wrote {len(samples)} sample presets to {PUB_SAMPLES / 'samples.json'}")
