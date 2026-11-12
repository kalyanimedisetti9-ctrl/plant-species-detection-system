import os
import sys
import time
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault('KERAS_HOME', str(BASE_DIR / '.keras'))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

def quick_fine_tune():
    model_path = BASE_DIR / 'ml_model' / 'trained_model' / 'plant_species_model.keras'
    tflite_path = BASE_DIR / 'ml_model' / 'trained_model' / 'plant_species_model_v2.tflite'
    
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    train_dir = BASE_DIR / 'ml_model' / 'dataset' / 'train'
    val_dir = BASE_DIR / 'ml_model' / 'dataset' / 'validation'
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=32,
        shuffle=True,
        seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(224, 224),
        batch_size=32,
        shuffle=False
    )
    
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    print("Fine-tuning 3 epochs to adjust Brinjal weights...")
    model.fit(train_ds, validation_data=val_ds, epochs=3)
    
    print("Saving fine-tuned Keras model...")
    model.save(model_path)
    
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"SUCCESS: TFLite model written to {tflite_path}")

if __name__ == '__main__':
    quick_fine_tune()

