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

# Load MobileNetV2 feature extractor
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    pooling='avg',
    weights='imagenet'
)

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img, dtype=np.float32))
    feats = base_model(np.expand_dims(arr, axis=0)).numpy()[0]
    return feats / np.linalg.norm(feats)

with open(PROJECT_ROOT / 'ml_model' / 'trained_model' / 'class_indices.json') as f:
    indices = json.load(f)
classes = [k for k, v in sorted(indices.items(), key=lambda item: item[1])]

# Collect representative embeddings for each class from train images + species image
class_prototypes = {}
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
    class_prototypes[cls_name] = avg_vec

print("Testing classification of species images against prototypes:")
matches = 0
for cls_name in classes:
    slug = cls_name.lower().replace(' ', '_')
    img_path = species_dir / f"{slug}.jpg"
    if not img_path.exists():
        continue
    feat = extract_features(img_path)
    sims = {c: float(np.dot(feat, class_prototypes[c])) for c in classes}
    sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    top_pred, top_sim = sorted_sims[0]
    is_match = (top_pred == cls_name)
    if is_match:
        matches += 1
    print(f"{cls_name:15s} -> Top: {top_pred:15s} (sim: {top_sim:.3f}) | 2nd: {sorted_sims[1][0]:12s} ({sorted_sims[1][1]:.3f}) | {'PASS' if is_match else 'FAIL'}")

print(f"\nAccuracy: {matches}/{len(classes)} ({matches/len(classes)*100:.1f}%)")
