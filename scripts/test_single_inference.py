import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault('KERAS_HOME', str(BASE_DIR / '.keras'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'plant_detection_project.settings')

from ml_model.prediction import predict_plant, load_or_create_model, load_tflite_interpreter

def test_inference():
    print("Testing model inference...")
    
    # Warmup interpreter
    t0 = time.perf_counter()
    interp = load_tflite_interpreter()
    print(f"TFLite interpreter loaded in {(time.perf_counter()-t0)*1000:.1f}ms")
    
    test_images = [
        ("Brinjal", BASE_DIR / "media" / "uploaded_images" / "brinjal_nZ64tNt.jpeg"),
        ("Aloe Vera", BASE_DIR / "media" / "uploaded_images" / "alovera_pC3w3ZJ.jpeg"),
        ("Hibiscus", BASE_DIR / "media" / "uploaded_images" / "hibiscus_c1hiAku.jpeg"),
        ("Sunflower", BASE_DIR / "media" / "uploaded_images" / "sunflower_4R0J5c3.jpeg"),
        ("Rose", BASE_DIR / "media" / "uploaded_images" / "rose_bF1q1yL.jpeg"),
    ]
    
    print("\n--- INFERENCE TESTS ON USER SAMPLES ---")
    for expected, img_path in test_images:
        if not img_path.exists():
            print(f"Skipping {img_path} (does not exist)")
            continue
        
        t_start = time.perf_counter()
        result = predict_plant(str(img_path))
        elapsed = (time.perf_counter() - t_start) * 1000
        
        predicted = result.get('predicted_species') or result.get('name') or "Unknown"
        conf = result.get('confidence', 0.0)
        is_correct = (predicted.lower() == expected.lower())
        mark = "PASS" if is_correct else "FAIL"
        
        print(f"[{mark}] Expected: {expected:<12} | Predicted: {predicted:<12} | Conf: {conf:6.2f}% | Latency: {elapsed:5.1f}ms")
        top_preds = result.get('top_predictions', [])
        top_str = ", ".join([f"{p['species']} ({p['confidence']:.1f}%)" for p in top_preds])
        print(f"       Top: {top_str}")

if __name__ == '__main__':
    test_inference()
