import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault('KERAS_HOME', str(BASE_DIR / '.keras'))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

def convert():
    keras_path = BASE_DIR / 'ml_model' / 'trained_model' / 'plant_species_model.keras'
    tflite_path = BASE_DIR / 'ml_model' / 'trained_model' / 'plant_species_model_v2.tflite'
    
    print(f"Loading Keras model from {keras_path}...")
    model = tf.keras.models.load_model(keras_path)
    print("Model loaded. Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"SUCCESS: TFLite model written to {tflite_path} ({len(tflite_model)} bytes)")

if __name__ == '__main__':
    convert()

