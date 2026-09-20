"""
Rock-Paper-Scissors Live Gesture Recognition - Training Script
Loads dataset, trains MobileNetV2 transfer learning model, evaluates, and exports model.
"""

import os
import sys
import json
import zipfile
import urllib.request
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "rock_paper_scissors.keras")
METADATA_PATH = os.path.join(MODEL_DIR, "classes.json")
CACHE_DIR = os.path.join("scratch", "dataset_cache")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Step 1: Load and Inspect Dataset
# -----------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Loading Dataset")
print("=" * 60)

class_names = ["rock", "paper", "scissors"]
images = []
labels = []

# Try Hugging Face datasets first
hf_loaded = False
try:
    from datasets import load_dataset
    print("Attempting to load 'randall-lab/rock-paper-scissors' from Hugging Face...")
    ds = load_dataset("randall-lab/rock-paper-scissors", split="train")
    if "label" in ds.features and hasattr(ds.features["label"], "names"):
        class_names = list(ds.features["label"].names)
    print(f"Loaded {len(ds)} samples from Hugging Face.")
    for item in ds:
        img = item["image"].convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        images.append(np.array(img, dtype=np.float32))
        labels.append(int(item["label"]))
    hf_loaded = True
except Exception as e:
    print(f"Notice: HF load_dataset bypassed ({e}).")
    print("Falling back to canonical Laurence Moroney Rock-Paper-Scissors dataset...")

# Fallback: Download official canonical dataset
if not hf_loaded:
    data_url = "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip"
    zip_path = os.path.join(CACHE_DIR, "rps.zip")
    extract_dir = os.path.join(CACHE_DIR, "rps")

    if not os.path.exists(zip_path):
        print(f"Downloading dataset from {data_url}...")
        urllib.request.urlretrieve(data_url, zip_path)
        print("Download complete.")

    if not os.path.exists(extract_dir):
        print("Extracting zip archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(CACHE_DIR)
        print("Extraction complete.")

    # Locate classes
    print(f"Scanning images in {extract_dir}...")
    for idx, class_name in enumerate(class_names):
        class_folder = os.path.join(extract_dir, class_name)
        if not os.path.exists(class_folder):
            continue
        file_list = [f for f in os.listdir(class_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"  Class '{class_name}': {len(file_list)} images")
        for fname in file_list:
            fpath = os.path.join(class_folder, fname)
            try:
                with Image.open(fpath) as img:
                    img_rgb = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                    images.append(np.array(img_rgb, dtype=np.float32))
                    labels.append(idx)
            except Exception as img_err:
                print(f"Warning: Skipping corrupted image {fpath}: {img_err}")

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

print("\n--- Dataset Summary ---")
print(f"Total samples: {len(images)}")
print(f"Classes: {class_names}")
print(f"Image tensor shape: {images.shape}")
print(f"Class distribution:")
for idx, name in enumerate(class_names):
    count = int(np.sum(labels == idx))
    print(f"  [{idx}] {name}: {count} samples ({count / len(labels) * 100:.1f}%)")

# -----------------------------------------------------------------------------
# Step 2: Stratified Train / Validation Split (80 / 20)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Splitting Dataset (80% Train, 20% Validation)")
print("=" * 60)

X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.20, random_state=SEED, stratify=labels
)
print(f"Training set:   {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")

# -----------------------------------------------------------------------------
# Step 3: TensorFlow Dataset Pipeline & Data Augmentation
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Building Pipeline with Augmentation & MobileNetV2 Normalization")
print("=" * 60)

# Data augmentation layer for training
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.1, seed=SEED),
    layers.RandomZoom(0.1, seed=SEED),
    layers.RandomTranslation(0.05, 0.05, seed=SEED),
], name="data_augmentation")

# MobileNetV2 preprocessing: scales pixels from [0, 255] to [-1, 1]
# Formula: (x / 127.5) - 1.0
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(buffer_size=1024, seed=SEED)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -----------------------------------------------------------------------------
# Step 4: Model Architecture (MobileNetV2 Transfer Learning)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: Constructing MobileNetV2 Transfer Learning Model")
print("=" * 60)

# Input layer accepts raw [0, 255] RGB images
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_layer")
x = data_augmentation(inputs)
x = layers.Lambda(preprocess_input, name="mobilenetv2_preprocess")(x)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze backbone

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
x = layers.Dropout(0.2, name="dropout")(x)
outputs = layers.Dense(len(class_names), activation="softmax", name="predictions")(x)

model = models.Model(inputs, outputs, name="rock_paper_scissors_mobilenetv2")
model.summary()

# -----------------------------------------------------------------------------
# Step 5: Training
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 5: Training Model")
print("=" * 60)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

cb_list = [
    callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cb_list
)

# -----------------------------------------------------------------------------
# Step 6: Evaluation
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 6: Model Evaluation")
print("=" * 60)

val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"Validation Loss:     {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc * 100:.2f}%\n")

# Predict on validation set
raw_preds = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
y_pred = np.argmax(raw_preds, axis=1)

print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=class_names, digits=4))

print("Confusion Matrix:")
cm = confusion_matrix(y_val, y_pred)
print(f"{'':>12} " + " ".join([f"{name:>10}" for name in class_names]))
for i, row in enumerate(cm):
    print(f"{class_names[i]:>12} " + " ".join([f"{val:>10}" for val in row]))

print("\n--- Sample Predictions ---")
sample_indices = np.random.choice(len(X_val), size=min(5, len(X_val)), replace=False)
for idx in sample_indices:
    actual = class_names[y_val[idx]]
    pred_idx = y_pred[idx]
    predicted = class_names[pred_idx]
    conf = raw_preds[idx][pred_idx] * 100
    print(f"Actual: {actual:<10} | Predicted: {predicted:<10} | Confidence: {conf:6.2f}%")

# -----------------------------------------------------------------------------
# Step 7: Export Model and Metadata
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 7: Exporting Model and Metadata")
print("=" * 60)

metadata = {
    "input_size": IMG_SIZE,
    "classes": class_names,
    "normalization": "mobilenetv2: (x / 127.5) - 1.0",
    "val_accuracy": float(val_acc)
}

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Model saved to:    {MODEL_PATH}")
print(f"Metadata saved to: {METADATA_PATH}")
print("\nTraining completed successfully.")
