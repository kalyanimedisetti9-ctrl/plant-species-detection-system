import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_model.prediction import predict_plant

for split in ['test', 'val']:
    brinjal_dir = Path('ml_model/dataset') / split / 'Brinjal'
    images = list(brinjal_dir.glob('*.*'))
    print(f"\n--- Checking {split}/Brinjal ({len(images)} images) ---")
    correct = 0
    for img in images[:10]:
        res = predict_plant(str(img))
        pred = res['predicted_species']
        conf = res['confidence']
        if pred == 'Brinjal':
            correct += 1
            print(f"  [CORRECT] {img.name} -> {pred} ({conf}%)")
        else:
            print(f"  [MISMATCH] {img.name} -> {pred} ({conf}%)")
    print(f"Accuracy on sample: {correct}/{min(len(images), 10)}")

