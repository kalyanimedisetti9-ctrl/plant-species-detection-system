"""
Save TFLite to a new filename (v3) to avoid locked v2 file.
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
# Use v3 name to avoid the locked v2 file
TFLITE_TARGET = PROJECT_ROOT / 'ml_model' / 'trained_model' / 'plant_species_model_v3.tflite'

print(f"Loading Keras model...")
model = tf.keras.models.load_model(MODEL_KERAS)
print(f"Loaded: inputs={model.input_shape}, outputs={model.output_shape}")

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_data = converter.convert()
print(f"Done: {len(tflite_data)/1024/1024:.1f} MB")

# Write to temp first, then rename (atomic)
with tempfile.NamedTemporaryFile(
    suffix='.tflite', 
    dir=PROJECT_ROOT / 'ml_model' / 'trained_model',
    delete=False
) as tmp:
    tmp.write(tflite_data)
    tmp_path = Path(tmp.name)

print(f"Temp file: {tmp_path}")

# Remove target if exists and not locked
if TFLITE_TARGET.exists():
    try:
        TFLITE_TARGET.unlink()
    except Exception as e:
        print(f"Note: could not remove old v3: {e}")

tmp_path.rename(TFLITE_TARGET)
print(f"SUCCESS: {TFLITE_TARGET} ({TFLITE_TARGET.stat().st_size/1024/1024:.1f} MB)")

