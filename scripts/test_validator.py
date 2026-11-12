import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.environ['KERAS_HOME'] = str(BASE / '.keras')

from ml_model.plant_validator import validate_is_plant

test_cases = [
    # True plants
    ("Plant: Brinjal", "media/uploaded_images/brinjal_nZ64tNt.jpeg", True),
    ("Plant: Aloe Vera", "media/uploaded_images/alovera_pC3w3ZJ.jpeg", True),
    ("Plant: Hibiscus", "media/uploaded_images/hibiscus_c1hiAku.jpeg", True),
    ("Plant: Banana", "media/uploaded_images/banana.jpeg", True),
    ("Plant: Rose", "media/uploaded_images/rose.jpeg", True),
    
    # Non-plants
    ("Non-Plant: Hand / Skin", "media/test_validation_samples/hand_sample.jpg", False),
    ("Non-Plant: Vehicle / Car", "media/test_validation_samples/vehicle_sample.jpg", False),
    ("Non-Plant: Mobile Phone", "media/test_validation_samples/phone_sample.jpg", False),
]

print(f"{'Sample Name':<25} | {'Expected':<10} | {'Is Plant':<10} | {'Result Message'}")
print("-" * 80)

for name, rel_path, expected_plant in test_cases:
    full_path = BASE / rel_path
    if not full_path.exists():
        print(f"Skipping {rel_path} (not found)")
        continue
    res = validate_is_plant(str(full_path))
    is_p = res['is_plant']
    match = (is_p == expected_plant)
    mark = "PASS" if match else "FAIL"
    msg = res['message'] if not is_p else f"Accepted as Plant ({res['top_detected_category']})"
    print(f"[{mark}] {name:<22} | {str(expected_plant):<10} | {str(is_p):<10} | {msg}")

