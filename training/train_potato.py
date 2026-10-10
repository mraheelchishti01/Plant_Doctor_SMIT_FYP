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
BATCH_SIZE = 32
NUM_CLASSES = 3
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

def load_data_from_splits():
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(f"Splits file not found at {SPLITS_FILE}. Run inspect_dataset.py first.")

    with open(SPLITS_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    with open(MODELS_DIR / "class_mapping.json", "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    class_to_idx = class_mapping["classes_to_indices"]

    def extract_paths_and_labels(records):
        paths = [str(WORKSPACE_ROOT / r["path"]) for r in records]
        labels = [class_to_idx[r["class"]] for r in records]
        return paths, labels

    train_paths, train_labels = extract_paths_and_labels(splits["train"])
    val_paths, val_labels = extract_paths_and_labels(splits["val"])
    test_paths, test_labels = extract_paths_and_labels(splits["test"])

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_mapping

def create_tf_dataset(paths, labels, is_training=True, augment_fn=None, preprocess_fn=None):
    import tensorflow as tf

    paths_tensor = tf.constant(paths)
    labels_tensor = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))

    def parse_and_preprocess(filepath, label):
        image_bytes = tf.io.read_file(filepath)
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        if is_training and augment_fn is not None:
            image = augment_fn(image, training=True)
        if preprocess_fn is not None:
            image = preprocess_fn(image)
        return image, label

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=42)

    dataset = dataset.map(parse_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def train():
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

    print("=" * 60)
    print("POTATO DOCTOR AI - MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_mapping = load_data_from_splits()

    print(f"Train samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples (held out): {len(test_paths)}")

    # Compute class weights for imbalanced dataset
    labels_arr = np.array(train_labels)
    unique_classes, counts = np.unique(labels_arr, return_counts=True)
    total_train = len(labels_arr)
    class_weights = {}
    for cls, count in zip(unique_classes, counts):
        class_weights[int(cls)] = float(total_train / (NUM_CLASSES * count))

    print("\nCalculated Class Weights:")
    for cls, weight in class_weights.items():
        cls_name = class_mapping["indices_to_classes"][str(cls)]
        print(f"  Class {cls} ({cls_name}): weight = {weight:.3f}")

    # Data augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    train_ds = create_tf_dataset(train_paths, train_labels, is_training=True, augment_fn=data_augmentation, preprocess_fn=preprocess_input)
    val_ds = create_tf_dataset(val_paths, val_labels, is_training=False, preprocess_fn=preprocess_input)

    # Base Model: MobileNetV2 pre-trained on ImageNet
    print("\nBuilding MobileNetV2 Transfer Learning Architecture...")
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(128, activation='relu', name="dense_128")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="classifier_output")(x)

    model = models.Model(inputs, outputs, name="potato_doctor_mobilenetv2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    model.summary()

    model_save_path = MODELS_DIR / "potato_model.keras"

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(model_save_path),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    print("\n--- PHASE 1: Training Classification Head (Feature Extraction) ---")
    history_phase1 = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint, reduce_lr]
    )

    print("\n--- PHASE 2: Fine-Tuning Top Layers ---")
    base_model.trainable = True
    # Freeze bottom layers, fine-tune top 25 layers
    for layer in base_model.layers[:-25]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    history_phase2 = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=history_phase1.epoch[-1] + 1 if history_phase1.epoch else 0,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint, reduce_lr]
    )

    # Save model config metadata
    model_config = {
        "model_name": "potato_doctor_mobilenetv2",
        "architecture": "MobileNetV2 Transfer Learning + Custom Dense Classification Head",
        "input_shape": [IMG_HEIGHT, IMG_WIDTH, 3],
        "preprocessing": {
            "method": "tf.keras.applications.mobilenet_v2.preprocess_input",
            "range": "[-1.0, 1.0]",
            "resize_dims": [IMG_HEIGHT, IMG_WIDTH]
        },
        "classes": class_mapping["indices_to_classes"],
        "class_weights": class_weights,
        "parameters": {
            "initial_learning_rate": 1e-3,
            "fine_tune_learning_rate": 1e-4,
            "batch_size": BATCH_SIZE
        }
    }

    with open(MODELS_DIR / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    # Combine training history
    def serialize_history(hist):
        return {k: [float(v) for v in vals] for k, vals in hist.history.items()}

    combined_history = {
        "phase1": serialize_history(history_phase1),
        "phase2": serialize_history(history_phase2)
    }

    with open(REPORTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)

    print(f"\nModel training completed successfully!")
    print(f"Best model saved to: {model_save_path}")
    print(f"Model configuration saved to: {MODELS_DIR / 'model_config.json'}")
    print("=" * 60)

if __name__ == "__main__":
    train()
