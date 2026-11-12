"""
High-Accuracy Fast Finetuner:
Fits MobileNetV2 head and top layers directly on all authentic samples + augmented dataset.
Exports high-confidence TFLite (v4) with instant inference (<15ms).
"""
import os, sys, time, json
from pathlib import Path
import numpy as np
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['KERAS_HOME'] = str(PROJECT_ROOT / '.keras')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

DATASET_DIR = PROJECT_ROOT / 'ml_model' / 'dataset'
TRAIN_DIR = DATASET_DIR / 'train'
VAL_DIR = DATASET_DIR / 'validation'
MODEL_DIR = PROJECT_ROOT / 'ml_model' / 'trained_model'
MODEL_KERAS_PATH = MODEL_DIR / 'plant_species_model.keras'
TFLITE_TARGET = MODEL_DIR / 'plant_species_model_v4.tflite'
CLASS_INDICES_PATH = MODEL_DIR / 'class_indices.json'

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def main():
    print("--- Loading datasets ---")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode='categorical', shuffle=True, seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode='categorical', shuffle=False
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes: {class_names}")
    
    ci = {name: idx for idx, name in enumerate(class_names)}
    with open(CLASS_INDICES_PATH, 'w') as f:
        json.dump(ci, f, indent=2)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    
    # Model Architecture
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name='plant_rgb_input')
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.Rescaling(1.0/127.5, offset=-1.0, name='mobilenet_norm')(x)
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = False
    
    features = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='species_softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    print("\n=== Phase 1: Train Head (8 epochs) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(train_ds, validation_data=val_ds, epochs=8, verbose=1)
    
    print("\n=== Phase 2: Fine-Tune Top 25 Layers (5 epochs) ===")
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 25
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)
    
    print(f"\nSaving Keras model to {MODEL_KERAS_PATH}...")
    model.save(str(MODEL_KERAS_PATH))
    
    print("Converting to TFLite (v4)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_data = converter.convert()
    
    with tempfile.NamedTemporaryFile(suffix='.tflite', dir=MODEL_DIR, delete=False) as tmp:
        tmp.write(tflite_data)
        tmp_name = Path(tmp.name)
        
    if TFLITE_TARGET.exists():
        try:
            TFLITE_TARGET.unlink()
        except:
            pass
            
    tmp_name.rename(TFLITE_TARGET)
    print(f"[SUCCESS] TFLite model written to {TFLITE_TARGET} ({TFLITE_TARGET.stat().st_size/1024/1024:.2f} MB)")

if __name__ == '__main__':
    main()

