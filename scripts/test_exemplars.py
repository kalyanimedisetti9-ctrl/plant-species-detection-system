import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / 'ml_model' / 'dataset' / 'train'
TEST_DIR = PROJECT_ROOT / 'ml_model' / 'dataset' / 'test'
SPECIES_DIR = PROJECT_ROOT / 'media' / 'species_images'

classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    pooling='avg',
    weights='imagenet'
)
base.trainable = False

def get_feats(paths):
    batch = [np.array(Image.open(p).convert('RGB').resize((224, 224)), dtype=np.float32) for p in paths]
    prep = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(batch))
    f = base(prep, training=False).numpy()
    return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-7)

print("Extracting train exemplars...")
class_exemplars = {}
for c in classes:
    paths = list((TRAIN_DIR / c).glob('*.*'))
    slug = c.lower().replace(' ', '_')
    sp = SPECIES_DIR / f'{slug}.jpg'
    if sp.exists() and sp not in paths:
        paths.append(sp)
    class_exemplars[c] = get_feats(paths)

correct = 0
total = 0
d_corr, d_tot = 0, 0
m_corr, m_tot = 0, 0

for c in classes:
    test_paths = list((TEST_DIR / c).glob('*.*'))
    if not test_paths:
        continue
    feats = get_feats(test_paths)
    for p, f in zip(test_paths, feats):
        total += 1
        # Max cosine similarity across all exemplars of each class
        scores = {cls: float(np.max(np.dot(class_exemplars[cls], f))) for cls in classes}
        pred_cls, pred_score = max(scores.items(), key=lambda x: x[1])
        is_match = (pred_cls == c)
        if is_match:
            correct += 1
        if c == 'Dandelion':
            d_tot += 1
            if is_match: d_corr += 1
        elif c == 'Money Plant':
            m_tot += 1
            if is_match: m_corr += 1
        res_str = "PASS" if is_match else "FAIL"
        print(f"{p.name:25s} | Expected: {c:15s} | Pred: {pred_cls:15s} | Sim: {pred_score:.3f} | {res_str}")

print(f"\nOverall: {correct}/{total} ({correct/total*100:.1f}%) | Dandelion: {d_corr}/{d_tot} | Money Plant: {m_corr}/{m_tot}")
