"""
Complete Training, Optimization, and Evaluation Pipeline for 34 Botanical Classes
Constructs high-accuracy MobileNetV2 feature representation, multi-exemplar prototypes,
and calibrated cosine-similarity classifier with built-in preprocessing.
Evaluates on unseen test images for every single supported species.
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

# 1. Identify all 34 classes alphabetically
classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
num_classes = len(classes)
print(f"=== Training Pipeline for {num_classes} Botanical Species ===")
print("Classes:", classes)

# Save class indices mapping
class_indices = {name: idx for idx, name in enumerate(classes)}
with open(CLASS_INDICES_PATH, 'w', encoding='utf-8') as f:
    json.dump(class_indices, f, indent=4)
print(f"Saved class indices mapping to {CLASS_INDICES_PATH}")

# 2. Build MobileNetV2 Feature Extractor
base_extractor = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    pooling='avg',
    weights='imagenet'
)
base_extractor.trainable = False

def extract_features(img_path_or_pil):
    if isinstance(img_path_or_pil, (str, Path)):
        img = Image.open(img_path_or_pil).convert('RGB').resize(IMAGE_SIZE)
    else:
        img = img_path_or_pil.convert('RGB').resize(IMAGE_SIZE)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img, dtype=np.float32))
    feats = base_extractor(np.expand_dims(arr, axis=0), training=False).numpy()[0]
    norm = np.linalg.norm(feats)
    return feats / (norm + 1e-7)

# 3. Extract and compute centroid prototypes for all classes
print("\n--- Extracting multi-exemplar botanical prototypes ---")
class_prototypes = []

for idx, cls_name in enumerate(classes):
    cls_folder = TRAIN_DIR / cls_name
    slug = cls_name.lower().replace(' ', '_')
    img_paths = list(cls_folder.glob('*.*'))

    # Include cover image from media/species_images if available
    card_path = SPECIES_DIR / f"{slug}.jpg"
    if card_path.exists() and card_path not in img_paths:
        img_paths.append(card_path)

    vectors = [extract_features(p) for p in img_paths]
    
    # Compute normalized centroid vector
    centroid = np.mean(vectors, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-7)
    class_prototypes.append(centroid)
    print(f"  [{idx:2d}] {cls_name:18s}: {len(img_paths):2d} exemplar images incorporated")

prototypes = np.array(class_prototypes, dtype=np.float32) # shape: (num_classes, 1280)

# 4. Build End-to-End Unified Keras Model
# Input accepts [0, 255] float32 RGB images (batch_size, 224, 224, 3)
inputs = tf.keras.Input(shape=(224, 224, 3), name='plant_image_input')

# Internal MobileNetV2 preprocessing maps [0, 255] -> [-1, 1]
preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
features = base_extractor(preprocessed)
norm_features = tf.keras.layers.Lambda(
    lambda t: tf.math.l2_normalize(t, axis=-1),
    name='feature_l2_norm'
)(features)

# Dense layer computing cosine similarities with calibrated temperature tau = 0.15
# tau = 0.15 gives crisp discrimination without artificial 100.00% saturation
temperature = 0.15
scaled_weights = prototypes.T / temperature # shape: (1280, num_classes)

dense_layer = tf.keras.layers.Dense(
    num_classes,
    use_bias=False,
    activation='softmax',
    name='species_softmax'
)
outputs = dense_layer(norm_features)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='PlantSpeciesNet_34')
dense_layer.set_weights([scaled_weights])

# Save trained Keras model
model.save(str(MODEL_SAVE_PATH))
print(f"\n[SUCCESS] Model saved successfully to {MODEL_SAVE_PATH}")

# Export optimized TFLite model
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    with open(TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_bytes)
    print(f"[SUCCESS] Exported optimized TFLite model to {TFLITE_SAVE_PATH}")
except Exception as e:
    print(f"Warning: Could not export TFLite: {e}")

# 5. Comprehensive Evaluation on Separate, Unseen Test Images
print("\n" + "="*75)
print("COMPREHENSIVE EVALUATION ON UNSEEN TEST IMAGES ACROSS ALL 34 CLASSES")
print("="*75)

total_tests = 0
correct_tests = 0
dandelion_tests = 0
dandelion_passes = 0
money_plant_tests = 0
money_plant_passes = 0

for cls_name in classes:
    test_folder = TEST_DIR / cls_name
    test_files = list(test_folder.glob('*.*')) if test_folder.exists() else []
    
    if not test_files:
        print(f"No test files found for {cls_name}")
        continue

    for test_f in test_files:
        total_tests += 1
        img = Image.open(test_f).convert('RGB').resize(IMAGE_SIZE)
        tensor = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

        # Run model inference
        preds = model(tensor, training=False).numpy()[0]
        best_idx = int(np.argmax(preds))
        pred_species = classes[best_idx]
        conf = float(preds[best_idx]) * 100.0

        is_correct = (pred_species == cls_name)
        if is_correct:
            correct_tests += 1

        if cls_name == "Dandelion":
            dandelion_tests += 1
            if is_correct: dandelion_passes += 1
        elif cls_name == "Money Plant":
            money_plant_tests += 1
            if is_correct: money_plant_passes += 1

        status = "PASS" if is_correct else "FAIL"
        print(f"Test Image: {test_f.name:25s} | Expected: {cls_name:16s} | Predicted: {pred_species:16s} | Conf: {conf:5.1f}% | {status}")

accuracy = (correct_tests / total_tests) * 100.0 if total_tests > 0 else 0
print("\n" + "="*75)
print(f"Overall Test Accuracy: {correct_tests}/{total_tests} ({accuracy:.2f}%)")
if dandelion_tests:
    print(f"Dandelion Specific Accuracy: {dandelion_passes}/{dandelion_tests} ({dandelion_passes/dandelion_tests*100:.1f}%)")
if money_plant_tests:
    print(f"Money Plant Specific Accuracy: {money_plant_passes}/{money_plant_tests} ({money_plant_passes/money_plant_tests*100:.1f}%)")
print("="*75)
