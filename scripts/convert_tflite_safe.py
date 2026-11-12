"""
Convert the retrained Keras model to TFLite.
Writes to a local temp path first to avoid OneDrive file-lock, then copies.
"""
import os, sys, shutil
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['KERAS_HOME'] = str(PROJECT_ROOT / '.keras')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

MODEL_KERAS = PROJECT_ROOT / 'ml_model' / 'trained_model' / 'plant_species_model.keras'
TFLITE_TARGET = PROJECT_ROOT / 'ml_model' / 'trained_model' / 'plant_species_model_v2.tflite'

print(f"Loading Keras model from {MODEL_KERAS}...")
model = tf.keras.models.load_model(MODEL_KERAS)
print(f"Model loaded: inputs={model.input_shape}, outputs={model.output_shape}")

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_data = converter.convert()
print(f"Conversion done: {len(tflite_data)/1024/1024:.1f} MB")

# Write to a temp file in system TEMP first (not OneDrive), then copy
with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmp:
    tmp.write(tflite_data)
    tmp_path = tmp.name
print(f"Written to temp: {tmp_path}")

# Remove the old locked file and copy new one
if TFLITE_TARGET.exists():
    try:
        TFLITE_TARGET.unlink()
        print("Removed old TFLite file")
    except Exception as e:
        print(f"Could not remove old file ({e}), will try overwrite via copy")

shutil.copy2(tmp_path, TFLITE_TARGET)
os.unlink(tmp_path)
print(f"SUCCESS: TFLite saved to {TFLITE_TARGET} ({TFLITE_TARGET.stat().st_size/1024/1024:.1f} MB)")

