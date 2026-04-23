"""
train_model.py
==============
CNN model training script — MobileNetV2 transfer learning.

Expects Data/processed/train/ and Data/processed/val/ to exist.
Run preprocess_data.py first if they don't.

Output:
  Model/mask_detector_mobilenetv2.h5
  Model/class_labels.json
  Model/training_history.png
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "Data" / "processed"
MODEL_DIR = BASE_DIR / "Model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "mask_detector_mobilenetv2.h5"
CLASS_LABELS_PATH = MODEL_DIR / "class_labels.json"
HISTORY_PLOT_PATH = MODEL_DIR / "training_history.png"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

CLASSES = ["proper_mask", "no_mask", "incorrect_mask"]


# ──────────────────────────────────────────────
# DATA GENERATORS
# ──────────────────────────────────────────────

def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        zoom_range=0.1,
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        PROCESSED_DIR / "train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        PROCESSED_DIR / "val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )

    return train_gen, val_gen


# ──────────────────────────────────────────────
# MODEL ARCHITECTURE
# ──────────────────────────────────────────────

def build_model(num_classes: int = 3) -> tf.keras.Model:
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze base

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def train(model, train_gen, val_gen):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(
            str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────

def evaluate(model, val_gen):
    print("\n[Evaluation] Generating predictions on validation set…")
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)


# ──────────────────────────────────────────────
# PLOT HISTORY
# ──────────────────────────────────────────────

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0a1a")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")

    # Accuracy
    axes[0].plot(history.history["accuracy"], color="#00e5ff", label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], color="#76ff03", label="Val Acc")
    axes[0].set_title("Accuracy", color="white")
    axes[0].set_xlabel("Epoch", color="white")
    axes[0].set_ylabel("Accuracy", color="white")
    axes[0].legend(facecolor="#1a1a2e", labelcolor="white")

    # Loss
    axes[1].plot(history.history["loss"], color="#ff6b6b", label="Train Loss")
    axes[1].plot(history.history["val_loss"], color="#ffd93d", label="Val Loss")
    axes[1].set_title("Loss", color="white")
    axes[1].set_xlabel("Epoch", color="white")
    axes[1].set_ylabel("Loss", color="white")
    axes[1].legend(facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    plt.savefig(str(HISTORY_PLOT_PATH), dpi=150, facecolor=fig.get_facecolor())
    print(f"[✓] Training history saved to {HISTORY_PLOT_PATH}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Face Mask Compliance — Model Training (MobileNetV2)")
    print("=" * 60)

    if not (PROCESSED_DIR / "train").exists():
        print("[ERROR] Processed dataset not found. Run preprocess_data.py first.")
        return

    print("\n[1/4] Building data generators…")
    train_gen, val_gen = build_generators()
    print(f"  Train samples: {train_gen.samples}")
    print(f"  Val samples  : {val_gen.samples}")

    # Save class label mapping
    class_labels = {v: k for k, v in train_gen.class_indices.items()}
    with open(CLASS_LABELS_PATH, "w") as f:
        json.dump(class_labels, f, indent=2)
    print(f"[✓] Class labels saved: {class_labels}")

    print("\n[2/4] Building MobileNetV2 model…")
    model = build_model(num_classes=len(CLASSES))
    model.summary()

    print("\n[3/4] Training…")
    history = train(model, train_gen, val_gen)

    print("\n[4/4] Evaluating…")
    evaluate(model, val_gen)
    plot_history(history)

    print("\n" + "=" * 60)
    print(f"  Training complete! Model saved to {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
