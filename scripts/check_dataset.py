import os
import json
from pathlib import Path

base = Path('.')
species_dir = base / 'media' / 'species_images'
train_dir = base / 'ml_model' / 'dataset' / 'train'
test_dir = base / 'ml_model' / 'dataset' / 'test'

print("Species images:")
for f in sorted(species_dir.glob('*.jpg')):
    print(f"  {f.name} ({os.path.getsize(f)} bytes)")

print("\nTrain classes count:", len(list(train_dir.iterdir())) if train_dir.exists() else 0)
for d in sorted(train_dir.iterdir()) if train_dir.exists() else []:
    if d.is_dir():
        print(f"  {d.name}: {len(list(d.glob('*.jpg')))} images")
