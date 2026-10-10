import os
import sys
import json
from pathlib import Path
import numpy as np

# Ensure Windows finds 64-bit MSVC runtime DLLs
_msvc_dir = r"C:\Program Files\Microsoft Office\root\Client"
if os.path.exists(_msvc_dir) and _msvc_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _msvc_dir + ";" + os.environ.get("PATH", "")

# Suppress excessive TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SPLITS_FILE = WORKSPACE_ROOT / "reports" / "dataset_splits.json"
MODELS_DIR = WORKSPACE_ROOT / "models"
REPORTS_DIR = WORKSPACE_ROOT / "reports"

IMG_HEIGHT = 224
IMG_WIDTH = 224

def evaluate():
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("POTATO DOCTOR AI - INDEPENDENT TEST EVALUATION")
    print("=" * 60)

    model_path = MODELS_DIR / "potato_model.keras"
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}. Please run train_potato.py first.")
        sys.exit(1)

    with open(SPLITS_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    with open(MODELS_DIR / "class_mapping.json", "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    class_to_idx = class_mapping["classes_to_indices"]
    idx_to_class = {int(k): v for k, v in class_mapping["indices_to_classes"].items()}
    friendly_names = class_mapping["friendly_names"]
    target_names = [friendly_names[idx_to_class[i]] for i in range(len(idx_to_class))]

    test_records = splits["test"]
    print(f"Evaluating model on {len(test_records)} unseen test images...")

    print(f"Loading trained model: {model_path}...")
    model = tf.keras.models.load_model(str(model_path))

    # Load and preprocess all test images
    y_true = []
    y_pred_probs = []
    y_pred = []
    misclassified = []

    for idx, record in enumerate(test_records):
        full_path = WORKSPACE_ROOT / record["path"]
        true_label = class_to_idx[record["class"]]
        y_true.append(true_label)

        # Read and preprocess
        img = tf.keras.preprocessing.image.load_img(
            str(full_path), target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array)

        # Predict
        preds = model.predict(img_preprocessed, verbose=0)[0]
        pred_label = int(np.argmax(preds))
        confidence = float(np.max(preds))

        y_pred_probs.append(preds.tolist())
        y_pred.append(pred_label)

        if pred_label != true_label:
            misclassified.append({
                "image_path": record["path"],
                "true_class": record["class"],
                "true_friendly": friendly_names[record["class"]],
                "pred_class": idx_to_class[pred_label],
                "pred_friendly": friendly_names[idx_to_class[pred_label]],
                "confidence": confidence,
                "probabilities": {idx_to_class[i]: float(preds[i]) for i in range(len(preds))}
            })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    test_acc = float(np.mean(y_true == y_pred))
    print(f"\n>>> TEST ACCURACY: {test_acc * 100:.2f}% <<<")

    # Scikit-learn classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        digits=4
    )

    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot Confusion Matrix with rich styling
    plt.figure(figsize=(8, 6), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Potato Doctor AI - Test Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=25, ha='right', fontsize=10)
    plt.yticks(tick_marks, target_names, fontsize=10)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12, fontweight='bold'
            )

    plt.ylabel('Actual Leaf Condition', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Condition', fontsize=11, fontweight='bold')
    plt.tight_layout()

    cm_save_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(str(cm_save_path), bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix chart to: {cm_save_path}")

    # Save complete evaluation report
    eval_results = {
        "test_samples": len(test_records),
        "test_accuracy": test_acc,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "class_labels": target_names,
        "num_misclassified": len(misclassified),
        "misclassified_samples": misclassified[:15] # Top 15 samples
    }

    with open(REPORTS_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print(f"Saved classification report to: {REPORTS_DIR / 'classification_report.json'}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate()
