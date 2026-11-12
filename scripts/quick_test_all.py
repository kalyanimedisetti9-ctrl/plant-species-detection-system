import os
from pathlib import Path
from ml_model.prediction import predict_plant

BASE = Path('media/uploaded_images')
test_files = [
    ('Brinjal', 'brinjal_nZ64tNt.jpeg'),
    ('Aloe Vera', 'alovera_pC3w3ZJ.jpeg'),
    ('Hibiscus', 'hibiscus_c1hiAku.jpeg'),
    ('Banana', 'banana.jpeg'),
    ('Jasmine', 'jasmine.jpeg'),
    ('Papaya', 'papaya.jpeg'),
    ('Rose', 'rose.jpeg'),
    ('Curry Leaf', 'curryleaf.jpeg'),
]

print(f"{'Target Plant':<15} | {'Predicted Plant':<15} | {'Confidence':<12} | {'Status'}")
print("-" * 55)

for expected, filename in test_files:
    file_path = BASE / filename
    if file_path.exists():
        res = predict_plant(str(file_path))
        pred = res['predicted_species']
        conf = res['confidence']
        status = "PASS" if pred.lower() == expected.lower() else "FAIL"
        print(f"{expected:<15} | {pred:<15} | {conf:>6.2f}%     | {status}")

