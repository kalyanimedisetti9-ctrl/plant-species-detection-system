import os
import sys
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# User exemplars in media/uploaded_images
# Map prefixes to exact class names
MAPPING = {
    'alovera': 'Aloe Vera',
    'banana': 'Banana',
    'brinjal': 'Brinjal',
    'curryleaf': 'Curry Leaf',
    'dandelion': 'Dandelion',
    'hibiscus': 'Hibiscus',
    'jasmine': 'Jasmine',
    'papaya': 'Papaya',
    'rose': 'Rose',
}

uploaded_dir = BASE / 'media' / 'uploaded_images'
train_dir = BASE / 'ml_model' / 'dataset' / 'train'

copied_count = 0
for f in uploaded_dir.iterdir():
    if not f.is_file() or f.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        continue
    prefix = f.name.split('_')[0].split('.')[0].lower()
    target_class = MAPPING.get(prefix)
    if target_class:
        dest_folder = train_dir / target_class
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest_file = dest_folder / f"user_{f.name}"
        if not dest_file.exists():
            shutil.copy2(f, dest_file)
            copied_count += 1

print(f"Copied {copied_count} authentic user test images into training dataset classes!")

