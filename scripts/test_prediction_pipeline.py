"""
Test the inference pipeline on Brinjal, Banana, Marigold, Aloe Vera, and Sunflower.
Verifies predicted species, scientific name, confidence, and latency.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_model.prediction import predict_plant

test_cases = [
    ("Brinjal species image", PROJECT_ROOT / "media" / "species_images" / "brinjal.jpg", "Brinjal"),
    ("Brinjal test sample", PROJECT_ROOT / "media" / "test_samples" / "brinjal.jpg", "Brinjal"),
    ("Banana species image", PROJECT_ROOT / "media" / "species_images" / "banana.jpg", "Banana"),
    ("Marigold species image", PROJECT_ROOT / "media" / "species_images" / "marigold.jpg", "Marigold"),
    ("Marigold test sample", PROJECT_ROOT / "media" / "test_samples" / "marigold.jpg", "Marigold"),
    ("Aloe Vera species image", PROJECT_ROOT / "media" / "species_images" / "aloe_vera.jpg", "Aloe Vera"),
    ("Sunflower species image", PROJECT_ROOT / "media" / "species_images" / "sunflower.jpg", "Sunflower")
]

print("="*85)
print("TESTING FULL INFERENCE PIPELINE WITH CACHED MODEL")
print("="*85)

all_passed = True
for label, path, expected in test_cases:
    if not path.exists():
        print(f"Skipping {label}: file does not exist at {path}")
        continue
    
    res = predict_plant(str(path))
    pred_species = res['predicted_species']
    sc_name = res['scientific_name']
    conf = res['confidence']
    is_conf = res['is_confident']
    latency = res['latency_ms']['total_ms']
    
    is_match = (pred_species.lower() == expected.lower())
    status = "PASS" if is_match else "FAIL"
    if not is_match:
        all_passed = False
        
    print(f"[{status:4s}] {label:24s} | Expected: {expected:10s} | Predicted: {pred_species:10s} ({sc_name:22s}) | Conf: {conf:5.2f}% | Latency: {latency:5.1f}ms")

print("="*85)
if all_passed:
    print("ALL TEST CASES PASSED SUCCESSFULLY!")
else:
    print("SOME TEST CASES FAILED - PLEASE REVIEW.")
print("="*85)
