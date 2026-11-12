"""
Model Training & Fine-Tuning Script for Plant Species Detection System
Uses TensorFlow / Keras with Transfer Learning (MobileNetV2), Data Augmentation,
and built-in MobileNetV2 pixel normalization [-1.0, 1.0].
"""

import os
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / 'ml_model' / 'dataset'
TRAIN_DIR = DATASET_DIR / 'train'
TEST_DIR = DATASET_DIR / 'test'
MODEL_DIR = BASE_DIR / 'ml_model' / 'trained_model'
MODEL_SAVE_PATH = MODEL_DIR / 'plant_species_model.keras'
TFLITE_SAVE_PATH = MODEL_DIR / 'plant_species_model.tflite'
CLASS_INDICES_PATH = MODEL_DIR / 'class_indices.json'

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

def build_transfer_model(num_classes):
    """
    Constructs a high-accuracy botanical species classifier
    combining MobileNetV2 feature representations with specialized classification layers.
    Includes built-in normalization so the model accepts standard [0, 255] RGB float tensors.
    """
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="input_image")

    # Built-in MobileNetV2 preprocessing: [0, 255] -> [-1, 1]
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenet_norm")(inputs)

    # Data Augmentation pipeline (active only during model training)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.2)(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    # Freeze base model layers initially
    base_model.trainable = False

    # MobileNetV2 feature extractor
    x = base_model(x, training=False)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', name="dense_features")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="species_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PlantSpeciesClassifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )

    return model, base_model


def train_model(train_path=TRAIN_DIR, val_path=TEST_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading Training Dataset ===", flush=True)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True,
        seed=42
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Detected {num_classes} botanical classes: {class_names}", flush=True)

    val_ds = None
    if os.path.exists(val_path) and len(os.listdir(val_path)) > 0:
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_path,
            image_size=IMAGE_SIZE,
            batch_size=batch_size,
            label_mode='categorical',
            shuffle=False
        )

    # Save exact class-label mapping
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    with open(CLASS_INDICES_PATH, 'w', encoding='utf-8') as f:
        json.dump(class_indices, f, indent=4)
    print(f"Saved class index mapping to: {CLASS_INDICES_PATH}", flush=True)

    # Build model
    model, base_model = build_transfer_model(num_classes)

    # Pre-train classification head
    print(f"\n--- Training Classification Head ({epochs} epochs) ---", flush=True)
    model.fit(
        train_ds,
        validation_data=val_ds if val_ds else None,
        epochs=epochs,
        verbose=1
    )

    # Fine-tuning: unfreeze the top layers of MobileNetV2
    print("\n--- Fine-Tuning Top Layers of MobileNetV2 ---", flush=True)
    base_model.trainable = True
    for layer in base_model.layers[:-25]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )

    model.fit(
        train_ds,
        validation_data=val_ds if val_ds else None,
        epochs=10,
        verbose=1
    )

    # Save final trained Keras model
    model.save(str(MODEL_SAVE_PATH))
    print(f"\n[SUCCESS] Trained Keras model saved successfully to: {MODEL_SAVE_PATH}", flush=True)

    # Convert to TFLite for ultra-fast low-latency inference
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_bytes = converter.convert()
        with open(TFLITE_SAVE_PATH, 'wb') as f:
            f.write(tflite_bytes)
        print(f"[SUCCESS] Exported optimized TFLite model to: {TFLITE_SAVE_PATH}", flush=True)
    except Exception as e:
        print(f"Warning: Could not export TFLite model: {e}", flush=True)

    return model


if __name__ == '__main__':
    train_model()
