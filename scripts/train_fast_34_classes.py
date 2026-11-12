"""
High-Speed Batched Training, Calibration, Export, and Evaluation Pipeline for 34 Botanical Classes
Extracts features in high-performance batches (32 per batch), computes multi-exemplar prototypes,
exports both Keras (.keras) and optimized TFLite (.tflite),
and performs comprehensive evaluation on separate unseen test images.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

TRAIN_DIR = PROJECT_ROOT / 'ml_model' / 'dataset' / 'train'
TEST_DIR = PROJECT_ROOT / 'ml_model' / 'dataset' / 'test'
SPECIES_DIR = PROJECT_ROOT / 'media' / 'species_images'
MODEL_DIR = PROJECT_ROOT / 'ml_model' / 'trained_model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODEL_DIR / 'plant_species_model.keras'
TFLITE_SAVE_PATH = MODEL_DIR / 'plant_species_model.tflite'
CLASS_INDICES_PATH = MODEL_DIR / 'class_indices.json'

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# 1. Classes
classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
num_classes = len(classes)
print(f"=== Fast Training Pipeline for {num_classes} Botanical Species ===", flush=True)

class_indices = {name: idx for idx, name in enumerate(classes)}
with open(CLASS_INDICES_PATH, 'w', encoding='utf-8') as f:
    json.dump(class_indices, f, indent=4)
print(f"Saved class index mapping ({num_classes} classes) to {CLASS_INDICES_PATH}", flush=True)

# 2. Base Extractor
base_extractor = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    pooling='avg',
    weights='imagenet'
)
base_extractor.trainable = False

def batch_extract(img_paths):
    """Loads a list of image paths and extracts normalized embeddings in batches."""
    all_features = []
    for i in range(0, len(img_paths), BATCH_SIZE):
        batch_paths = img_paths[i:i + BATCH_SIZE]
        batch_arrays = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB').resize(IMAGE_SIZE)
            arr = np.array(img, dtype=np.float32)
            batch_arrays.append(arr)
        
        batch_tensor = np.array(batch_arrays, dtype=np.float32)
        preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(batch_tensor)
        feats = base_extractor(preprocessed, training=False).numpy()
        norms = np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-7
        feats_norm = feats / norms
        all_features.append(feats_norm)
    
    return np.vstack(all_features)

# 3. Collect Exemplars and Extract Features
print("\n--- Extracting multi-exemplar features per class ---", flush=True)
t0 = time.time()
class_prototypes = []

for idx, cls_name in enumerate(classes):
    cls_folder = TRAIN_DIR / cls_name
    slug = cls_name.lower().replace(' ', '_')
    img_paths = list(cls_folder.glob('*.*'))

    card_path = SPECIES_DIR / f"{slug}.jpg"
    if card_path.exists() and card_path not in img_paths:
        img_paths.append(card_path)

    feats = batch_extract(img_paths)
    centroid = np.mean(feats, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-7)
    class_prototypes.append(centroid)
    print(f"  [{idx+1:2d}/{num_classes}] {cls_name:18s}: {len(img_paths):2d} images", flush=True)

print(f"Feature extraction completed in {time.time()-t0:.2f} seconds!", flush=True)

prototypes = np.array(class_prototypes, dtype=np.float32) # (num_classes, 1280)

# 4. Construct Unified Keras Model
# Raw RGB input in [0, 255]
inputs = tf.keras.Input(shape=(224, 224, 3), name='plant_image_input')
rescaled = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name='mobilenet_norm')(inputs)
features = base_extractor(rescaled)
norm_features = tf.keras.layers.UnitNormalization(axis=-1, name='feature_l2_norm')(features)

# Calibrated cosine-similarity softmax layer
temperature = 0.15
scaled_weights = prototypes.T / temperature

dense_layer = tf.keras.layers.Dense(
    num_classes,
    use_bias=False,
    activation='softmax',
    name='species_softmax'
)
outputs = dense_layer(norm_features)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='PlantSpeciesNet_34')
dense_layer.set_weights([scaled_weights])

# Save Keras Model
model.save(str(MODEL_SAVE_PATH))
print(f"\n[SUCCESS] Keras model saved to {MODEL_SAVE_PATH}", flush=True)

# Export TFLite Model
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    with open(TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_bytes)
    print(f"[SUCCESS] TFLite model exported to {TFLITE_SAVE_PATH}", flush=True)
except Exception as e:
    print(f"Warning: TFLite export error: {e}", flush=True)

# 5. Evaluate on Unseen Test Set
print("\n" + "="*80, flush=True)
print("EVALUATION ON SEPARATE UNSEEN TEST IMAGES ACROSS ALL 34 BOTANICAL CLASSES", flush=True)
print("="*80, flush=True)

# Initialize TFLite interpreter for ultra-fast evaluation
interp = tf.lite.Interpreter(model_path=str(TFLITE_SAVE_PATH))
interp.allocate_tensors()
inp_idx = interp.get_input_details()[0]['index']
out_idx = interp.get_output_details()[0]['index']

total_tests = 0
correct_tests = 0
dandelion_tests = 0
dandelion_passes = 0
money_plant_tests = 0
money_plant_passes = 0

for cls_name in classes:
    test_folder = TEST_DIR / cls_name
    test_files = list(test_folder.glob('*.*')) if test_folder.exists() else []

    for test_f in test_files:
        total_tests += 1
        img = Image.open(test_f).convert('RGB').resize(IMAGE_SIZE)
        tensor = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

        interp.set_tensor(inp_idx, tensor)
        interp.invoke()
        preds = interp.get_tensor(out_idx)[0]

        best_idx = int(np.argmax(preds))
        pred_species = classes[best_idx]
        conf = float(preds[best_idx]) * 100.0

        is_match = (pred_species == cls_name)
        if is_match:
            correct_tests += 1

        if cls_name == "Dandelion":
            dandelion_tests += 1
            if is_match: dandelion_passes += 1
        elif cls_name == "Money Plant":
            money_plant_tests += 1
            if is_match: money_plant_passes += 1

        status = "PASS" if is_match else "FAIL"
        print(f"Test File: {test_f.name:25s} | Expected: {cls_name:16s} | Predicted: {pred_species:16s} | Conf: {conf:5.1f}% | {status}", flush=True)

acc = (correct_tests / total_tests) * 100.0 if total_tests > 0 else 0
print("\n" + "="*80, flush=True)
print(f"OVERALL UNSEEN TEST ACCURACY: {correct_tests}/{total_tests} ({acc:.2f}%)", flush=True)
if dandelion_tests:
    print(f"Dandelion Leaves/Flower Accuracy: {dandelion_passes}/{dandelion_tests} ({dandelion_passes/dandelion_tests*100:.1f}%)", flush=True)
if money_plant_tests:
    print(f"Money Plant Accuracy: {money_plant_passes}/{money_plant_tests} ({money_plant_passes/money_plant_tests*100:.1f}%)", flush=True)
print("="*80, flush=True)
