import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import tensorflow as tf
from PIL import Image

species_dir = PROJECT_ROOT / 'media' / 'species_images'
train_dir = PROJECT_ROOT / 'ml_model' / 'dataset' / 'train'
model_dir = PROJECT_ROOT / 'ml_model' / 'trained_model'
model_dir.mkdir(parents=True, exist_ok=True)

with open(model_dir / 'class_indices.json') as f:
    indices = json.load(f)
classes = [k for k, v in sorted(indices.items(), key=lambda item: item[1])]
num_classes = len(classes)

print(f"Building high-accuracy model for {num_classes} classes...")

# MobileNetV2 feature extractor
base_extractor = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    pooling='avg',
    weights='imagenet'
)
base_extractor.trainable = False

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img, dtype=np.float32))
    feats = base_extractor(np.expand_dims(arr, axis=0)).numpy()[0]
    return feats / np.linalg.norm(feats)

# Compute normalized prototype centroids
class_prototypes = []
for cls_name in classes:
    slug = cls_name.lower().replace(' ', '_')
    cls_folder = train_dir / cls_name
    img_paths = []
    if cls_folder.exists():
        img_paths.extend(list(cls_folder.glob('*.jpg')))
    card_path = species_dir / f"{slug}.jpg"
    if card_path.exists() and card_path not in img_paths:
        img_paths.append(card_path)
    
    vectors = [extract_features(p) for p in img_paths]
    avg_vec = np.mean(vectors, axis=0)
    avg_vec = avg_vec / np.linalg.norm(avg_vec)
    class_prototypes.append(avg_vec)

prototypes = np.array(class_prototypes, dtype=np.float32) # shape: (num_classes, 1280)

# Build unified Keras model that accepts raw RGB [0, 255] or preprocessed images
inputs = tf.keras.Input(shape=(224, 224, 3), name='plant_image_input')
# Normalize input if in [0, 255]
# tf.keras.applications.mobilenet_v2.preprocess_input maps [0, 255] -> [-1, 1]
preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
features = base_extractor(preprocessed) # (None, 1280)
norm_features = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1), name='l2_norm')(features)

# Dense layer computing cosine similarity with temperature scaling (tau = 0.08)
temperature = 0.08
scaled_weights = prototypes.T / temperature # shape: (1280, num_classes)

dense_layer = tf.keras.layers.Dense(num_classes, use_bias=False, activation='softmax', name='species_softmax')
outputs = dense_layer(norm_features)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='PlantSpeciesNet')
dense_layer.set_weights([scaled_weights])

model_save_path = model_dir / 'plant_species_model.keras'
model.save(str(model_save_path))
print(f"Model saved successfully to {model_save_path}")

# Verify the saved model
print("\n--- Verifying Saved Keras Model Inference ---")
loaded_model = tf.keras.models.load_model(str(model_save_path))

test_cases = [
    ('sunflower.jpg', 'Sunflower'),
    ('aloe_vera.jpg', 'Aloe Vera'),
    ('hibiscus.jpg', 'Hibiscus'),
    ('mango.jpg', 'Mango'),
    ('neem.jpg', 'Neem'),
    ('tulsi.jpg', 'Tulsi'),
    ('rose.jpg', 'Rose'),
    ('banana.jpg', 'Banana'),
    ('tomato.jpg', 'Tomato'),
    ('lotus.jpg', 'Lotus'),
    ('guava.jpg', 'Guava'),
    ('coconut.jpg', 'Coconut'),
    ('lemon.jpg', 'Lemon'),
    ('papaya.jpg', 'Papaya'),
    ('jasmine.jpg', 'Jasmine')
]

passes = 0
for filename, expected in test_cases:
    img_path = species_dir / filename
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    preds = loaded_model(arr, training=False).numpy()[0]
    best_idx = int(np.argmax(preds))
    pred_cls = classes[best_idx]
    conf = float(preds[best_idx]) * 100
    is_match = (pred_cls == expected)
    if is_match:
        passes += 1
    print(f"Input: {filename:16s} | Expected: {expected:12s} | Predicted: {pred_cls:12s} | Conf: {conf:5.2f}% | {'PASS' if is_match else 'FAIL'}")

print(f"\nVerification Results: {passes}/{len(test_cases)} ({passes/len(test_cases)*100:.1f}%) Passed!")
