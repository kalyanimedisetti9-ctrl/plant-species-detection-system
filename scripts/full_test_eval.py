"""
Full test set evaluation + key exemplar evaluation.
Outputs per-class accuracy, confusion pairs, and overall test accuracy.
"""
import os, sys, time, json
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault('KERAS_HOME', str(BASE_DIR / '.keras'))

from detection.preprocessing import preprocess_image_from_path
from ml_model.prediction import load_tflite_interpreter

def get_class_labels():
    p = BASE_DIR / 'ml_model' / 'trained_model' / 'class_indices.json'
    with open(p, 'r') as f:
        ci = json.load(f)
    idx2name = {v: k for k, v in ci.items()}
    return [idx2name[i] for i in range(len(idx2name))]

def run_evaluation():
    class_labels = get_class_labels()
    num_classes = len(class_labels)
    print(f"Evaluating {num_classes} classes: {class_labels}")
    
    interpreter, in_details, out_details = load_tflite_interpreter()
    if interpreter is None:
        print("ERROR: TFLite interpreter failed to load!")
        return
    
    test_dir = BASE_DIR / 'ml_model' / 'dataset' / 'test'
    
    # Per-class tracking
    correct = {c: 0 for c in class_labels}
    total_count = {c: 0 for c in class_labels}
    confusion_pairs = []  # (true, predicted, confidence)
    latencies = []
    
    for cls_folder in sorted(test_dir.iterdir()):
        if not cls_folder.is_dir():
            continue
        true_class = cls_folder.name
        if true_class not in class_labels:
            print(f"WARNING: test folder '{true_class}' not in class_labels, skipping")
            continue
            
        images = sorted(cls_folder.glob('*.*'))
        for img_path in images:
            try:
                tensor = preprocess_image_from_path(str(img_path))
                t0 = time.perf_counter()
                interpreter.set_tensor(in_details[0]['index'], tensor)
                interpreter.invoke()
                raw = interpreter.get_tensor(out_details[0]['index'])[0]
                elapsed = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed)
                
                pred_idx = int(np.argmax(raw))
                pred_class = class_labels[pred_idx]
                conf = float(raw[pred_idx]) * 100
                
                total_count[true_class] = total_count.get(true_class, 0) + 1
                if pred_class == true_class:
                    correct[true_class] = correct.get(true_class, 0) + 1
                else:
                    confusion_pairs.append((true_class, pred_class, conf, img_path.name))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Print per-class results
    print("\n" + "="*70)
    print("PER-CLASS TEST RESULTS")
    print("="*70)
    print(f"{'Class':<16} {'Correct':>7} {'Total':>5} {'Accuracy':>8}")
    print("-"*40)
    
    total_correct = 0
    total_images = 0
    for cls in class_labels:
        c = correct.get(cls, 0)
        t = total_count.get(cls, 0)
        acc = (c / t * 100) if t > 0 else 0
        total_correct += c
        total_images += t
        marker = " !" if acc < 50 and t > 0 else ""
        print(f"{cls:<16} {c:>7} {t:>5} {acc:>7.1f}%{marker}")
    
    overall_acc = (total_correct / total_images * 100) if total_images > 0 else 0
    print(f"\nOVERALL TEST ACCURACY: {total_correct}/{total_images} = {overall_acc:.1f}%")
    
    if latencies:
        print(f"AVERAGE INFERENCE LATENCY: {np.mean(latencies):.1f}ms (median: {np.median(latencies):.1f}ms)")
    
    # Print confusion pairs
    if confusion_pairs:
        print(f"\n{'='*70}")
        print(f"MISCLASSIFICATIONS ({len(confusion_pairs)} errors)")
        print(f"{'='*70}")
        print(f"{'True Class':<16} {'Predicted':<16} {'Conf':>6}  File")
        print("-"*60)
        for true, pred, conf, fname in confusion_pairs:
            print(f"{true:<16} {pred:<16} {conf:>5.1f}%  {fname}")
    
    # Key exemplar tests
    print(f"\n{'='*70}")
    print("KEY EXEMPLAR TESTS (User uploaded images)")
    print(f"{'='*70}")
    
    exemplars = [
        ("Brinjal", "media/uploaded_images/brinjal_nZ64tNt.jpeg"),
        ("Aloe Vera", "media/uploaded_images/alovera_pC3w3ZJ.jpeg"),
        ("Hibiscus", "media/uploaded_images/hibiscus_c1hiAku.jpeg"),
        ("Rose", "media/uploaded_images/rose.jpeg"),
        ("Banana", "media/uploaded_images/banana.jpeg"),
        ("Jasmine", "media/uploaded_images/jasmine.jpeg"),
        ("Papaya", "media/uploaded_images/papaya.jpeg"),
    ]
    
    for expected, rel_path in exemplars:
        full_path = BASE_DIR / rel_path
        if not full_path.exists():
            print(f"  [SKIP] {rel_path} not found")
            continue
        tensor = preprocess_image_from_path(str(full_path))
        t0 = time.perf_counter()
        interpreter.set_tensor(in_details[0]['index'], tensor)
        interpreter.invoke()
        raw = interpreter.get_tensor(out_details[0]['index'])[0]
        elapsed = (time.perf_counter() - t0) * 1000
        
        pred_idx = int(np.argmax(raw))
        pred_class = class_labels[pred_idx]
        conf = float(raw[pred_idx]) * 100
        
        top3_idx = np.argsort(raw)[::-1][:3]
        top3_str = ", ".join([f"{class_labels[i]}({raw[i]*100:.1f}%)" for i in top3_idx])
        
        mark = "PASS" if pred_class.lower() == expected.lower() else "FAIL"
        print(f"  [{mark}] Expected: {expected:<12} -> {pred_class:<12} ({conf:.1f}%) [{elapsed:.1f}ms]  Top3: {top3_str}")

if __name__ == '__main__':
    run_evaluation()

