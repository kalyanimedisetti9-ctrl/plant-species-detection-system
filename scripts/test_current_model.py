import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import tensorflow as tf
from detection.preprocessing import preprocess_image_from_path

with open(PROJECT_ROOT / 'ml_model' / 'trained_model' / 'class_indices.json') as f:
    indices = json.load(f)
classes = [k for k, v in sorted(indices.items(), key=lambda item: item[1])]

model = tf.keras.models.load_model(PROJECT_ROOT / 'ml_model' / 'trained_model' / 'plant_species_model.keras')

print(f"Loaded model with {len(classes)} classes.")

test_images = [
    ('sunflower.jpg', 'Sunflower'),
    ('aloe_vera.jpg', 'Aloe Vera'),
    ('hibiscus.jpg', 'Hibiscus'),
    ('mango.jpg', 'Mango'),
    ('neem.jpg', 'Neem'),
    ('rose.jpg', 'Rose'),
    ('tulsi.jpg', 'Tulsi'),
    ('banana.jpg', 'Banana'),
    ('tomato.jpg', 'Tomato'),
    ('lotus.jpg', 'Lotus')
]

for filename, expected in test_images:
    img_path = PROJECT_ROOT / 'media' / 'species_images' / filename
    tensor = preprocess_image_from_path(str(img_path))
    pred = model.predict(tensor, verbose=0)[0]
    best_idx = np.argmax(pred)
    pred_class = classes[best_idx]
    conf = pred[best_idx] * 100
    print(f"File: {filename:16s} | Expected: {expected:12s} | Predicted: {pred_class:12s} | Conf: {conf:5.2f}% | {'MATCH' if expected == pred_class else 'FAIL'}")
