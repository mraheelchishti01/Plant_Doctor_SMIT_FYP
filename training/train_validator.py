import os
import sys
import json
import time
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
SPLITS_FILE = WORKSPACE_ROOT / "reports" / "validator_dataset_splits.json"
MODELS_DIR = WORKSPACE_ROOT / "models"
REPORTS_DIR = WORKSPACE_ROOT / "reports"

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 7
FINE_TUNE_EPOCHS = 6

def load_data_from_splits():
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(f"Splits file not found at {SPLITS_FILE}. Run prepare_validator_dataset.py first.")

    with open(SPLITS_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    def extract(records):
        paths = [str(WORKSPACE_ROOT / r["path"]) for r in records]
        labels = [r["label"] for r in records]
        return paths, labels

    train_paths, train_labels = extract(splits["train"])
    val_paths, val_labels = extract(splits["val"])
    test_paths, test_labels = extract(splits["test"])

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), splits

def create_tf_dataset(paths, labels, is_training=True):
    import tensorflow as tf

    paths_tensor = tf.constant(paths)
    labels_tensor = tf.constant(labels, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))

    def parse_and_preprocess(filepath, label):
        image_bytes = tf.io.read_file(filepath)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        # MobileNetV2 preprocess: [-1, 1]
        image = (image / 127.5) - 1.0
        return image, label

    dataset = dataset.map(parse_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        ])
        dataset = dataset.shuffle(buffer_size=len(paths), seed=42)
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def build_validator_model():
    import tensorflow as tf

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="PotatoLeafValidator")
    return model, base_model

def train_and_evaluate():
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("POTATO DOCTOR AI - BINARY POTATO-LEAF VALIDATOR TRAINING")
    print("=" * 60)

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), splits = load_data_from_splits()

    print(f"Train samples: {len(train_paths)} (Pos: {sum(train_labels)}, Neg: {len(train_labels)-sum(train_labels)})")
    print(f"Val samples:   {len(val_paths)} (Pos: {sum(val_labels)}, Neg: {len(val_labels)-sum(val_labels)})")
    print(f"Test samples:  {len(test_paths)} (Pos: {sum(test_labels)}, Neg: {len(test_labels)-sum(test_labels)})")

    train_ds = create_tf_dataset(train_paths, train_labels, is_training=True)
    val_ds = create_tf_dataset(val_paths, val_labels, is_training=False)

    model, base_model = build_validator_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    model_save_path = MODELS_DIR / "potato_validator.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    print("\n[Phase 1] Training classification head with frozen MobileNetV2 base...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks
    )

    print("\n[Phase 2] Fine-tuning top 35 layers of MobileNetV2...")
    base_model.trainable = True
    for layer in base_model.layers[:-35]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks
    )

    # Save final model
    model.save(str(model_save_path))
    print(f"[Model Saved] Saved validated binary classifier to: {model_save_path}")

    # ==========================================
    # Rigorous Evaluation & Threshold Calibration
    # ==========================================
    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET EVALUATION & THRESHOLD CALIBRATION")
    print("=" * 60)

    test_records = splits["test"]
    raw_predictions = []
    y_true = []
    categories = []

    print(f"Evaluating {len(test_records)} held-out test samples...")
    for rec in test_records:
        full_p = WORKSPACE_ROOT / rec["path"]
        from PIL import Image
        with Image.open(full_p) as im:
            im_resized = im.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
            arr = (np.array(im_resized, dtype=np.float32) / 127.5) - 1.0
            inp = np.expand_dims(arr, axis=0)
            score = float(model.predict(inp, verbose=0)[0][0])
            raw_predictions.append(score)
            y_true.append(rec["label"])
            categories.append(rec["subtype"])

    raw_predictions = np.array(raw_predictions)
    y_true = np.array(y_true)

    # Calculate overall AUC
    auc_score = float(roc_auc_score(y_true, raw_predictions))

    # Evaluate multiple candidate thresholds
    threshold_candidates = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85]
    threshold_evaluations = {}

    for t in threshold_candidates:
        y_pred_t = (raw_predictions >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred_t, average='binary', zero_division=0)
        acc = float(np.mean(y_pred_t == y_true))
        cm = confusion_matrix(y_true, y_pred_t).tolist()
        
        # Subtype analysis
        fp_plant = sum(1 for p, y, c in zip(raw_predictions, y_true, categories) if y == 0 and c == 'other_plant_leaf' and p >= t)
        total_plant = sum(1 for y, c in zip(y_true, categories) if y == 0 and c == 'other_plant_leaf')
        
        fp_unrelated = sum(1 for p, y, c in zip(raw_predictions, y_true, categories) if y == 0 and c == 'unrelated_object' and p >= t)
        total_unrelated = sum(1 for y, c in zip(y_true, categories) if y == 0 and c == 'unrelated_object')
        
        threshold_evaluations[str(t)] = {
            "accuracy": round(acc, 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm,
            "false_positives_other_plants": fp_plant,
            "total_other_plants": total_plant,
            "other_plants_fp_rate": round(fp_plant / max(1, total_plant), 4),
            "false_positives_unrelated": fp_unrelated,
            "total_unrelated": total_unrelated,
            "unrelated_fp_rate": round(fp_unrelated / max(1, total_unrelated), 4)
        }

    # Selected operational policy:
    # HIGH threshold: P >= 0.80 -> Verified potato leaf (proceed to disease model)
    # LOW threshold:  P < 0.35  -> Potato Leaf Not Detected
    # BETWEEN:       0.35 <= P < 0.80 -> Unable to Verify Potato Leaf
    HIGH_THRESH = 0.80
    LOW_THRESH = 0.35

    # Scores breakdown by subtype
    scores_by_subtype = {}
    for subtype in set(categories):
        subset_scores = [float(p) for p, c in zip(raw_predictions, categories) if c == subtype]
        scores_by_subtype[subtype] = {
            "count": len(subset_scores),
            "mean": round(float(np.mean(subset_scores)), 4),
            "median": round(float(np.median(subset_scores)), 4),
            "min": round(float(np.min(subset_scores)), 4),
            "max": round(float(np.max(subset_scores)), 4),
            "pct_above_high_thresh": round(sum(1 for s in subset_scores if s >= HIGH_THRESH) / len(subset_scores) * 100, 2),
            "pct_below_low_thresh": round(sum(1 for s in subset_scores if s < LOW_THRESH) / len(subset_scores) * 100, 2),
            "pct_uncertain_zone": round(sum(1 for s in subset_scores if LOW_THRESH <= s < HIGH_THRESH) / len(subset_scores) * 100, 2)
        }

    # Validator Configuration
    validator_config = {
        "model_file": "potato_validator.keras",
        "architecture": "MobileNetV2-BinaryClassifier",
        "input_dimensions": [224, 224, 3],
        "classes": ["not_potato_leaf", "potato_leaf"],
        "thresholds": {
            "high_confidence_potato": HIGH_THRESH,
            "low_confidence_rejection": LOW_THRESH
        },
        "response_templates": {
            "rejected_not_potato": {
                "title": "Potato Leaf Not Detected",
                "message": "This image does not appear to show a potato leaf. Potato Doctor AI currently supports potato leaves only. Please upload a clear image of a potato leaf."
            },
            "uncertain_leaf": {
                "title": "Unable to Verify Potato Leaf",
                "message": "We could not confidently verify that this is a potato leaf. Please upload a clear, well-lit image showing the leaf."
            }
        },
        "performance_metrics": {
            "roc_auc": round(auc_score, 4),
            "high_thresh_eval": threshold_evaluations.get(str(HIGH_THRESH)),
            "scores_by_subtype": scores_by_subtype
        },
        "training_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    config_path = MODELS_DIR / "validator_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(validator_config, f, indent=2)

    eval_report = {
        "test_samples_total": len(test_records),
        "roc_auc": round(auc_score, 4),
        "candidate_thresholds": threshold_evaluations,
        "selected_policy": {
            "high_threshold": HIGH_THRESH,
            "low_threshold": LOW_THRESH
        },
        "scores_by_subtype": scores_by_subtype
    }

    report_path = REPORTS_DIR / "validator_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    print("\nValidation Evaluation Summary:")
    print(f"ROC AUC: {round(auc_score, 4)}")
    print(f"High Threshold ({HIGH_THRESH}) Accuracy: {threshold_evaluations[str(HIGH_THRESH)]['accuracy']}")
    print(f"High Threshold ({HIGH_THRESH}) Precision: {threshold_evaluations[str(HIGH_THRESH)]['precision']}")
    print(f"Scores by subtype:")
    for st, data in scores_by_subtype.items():
        print(f"  {st:20s}: Mean {data['mean']:.3f} | % >= {HIGH_THRESH}: {data['pct_above_high_thresh']}% | % < {LOW_THRESH}: {data['pct_below_low_thresh']}%")

    print(f"\nConfiguration saved to: {config_path}")
    print(f"Evaluation report saved to: {report_path}")

if __name__ == "__main__":
    train_and_evaluate()
