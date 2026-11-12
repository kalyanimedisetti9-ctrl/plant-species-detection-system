"""
Retrain the plant classifier from scratch with corrected approach:
1. Proper data augmentation
2. Reduced unfreezing scope in fine-tuning  
3. Correct loss function matching
4. Label smoothing to improve calibration
5. Stronger regularization for small dataset
"""
import os, sys, json, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['KERAS_HOME'] = str(PROJECT_ROOT / '.keras')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

DATASET_DIR = PROJECT_ROOT / 'ml_model' / 'dataset'
TRAIN_DIR = DATASET_DIR / 'train'
VAL_DIR = DATASET_DIR / 'validation'
TEST_DIR = DATASET_DIR / 'test'
MODEL_DIR = PROJECT_ROOT / 'ml_model' / 'trained_model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_KERAS_PATH = MODEL_DIR / 'plant_species_model.keras'
TFLITE_PATH = MODEL_DIR / 'plant_species_model_v2.tflite'
CLASS_INDICES_PATH = MODEL_DIR / 'class_indices.json'

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16  # Smaller batch for better generalization with small dataset

def build_pipelines():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode='categorical', shuffle=True, seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode='categorical', shuffle=False
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode='categorical', shuffle=False
    )
    
    class_names = train_ds.class_names
    print(f"Classes ({len(class_names)}): {class_names}", flush=True)
    
    # Save class indices
    ci = {name: idx for idx, name in enumerate(class_names)}
    with open(CLASS_INDICES_PATH, 'w') as f:
        json.dump(ci, f, indent=2)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes):
    """Build model with stronger augmentation and regularization for small dataset."""
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name='plant_rgb_input')
    
    # Aggressive augmentation for small dataset
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomFlip("vertical")(x)  # Plants can appear in various orientations
    x = layers.RandomRotation(0.3)(x)
    x = layers.RandomZoom((-0.2, 0.2))(x)
    x = layers.RandomContrast(0.2)(x)
    x = layers.RandomBrightness(0.15)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    
    # MobileNetV2 normalization
    x = layers.Rescaling(1.0/127.5, offset=-1.0, name='mobilenet_norm')(x)
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = False
    
    features = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name='gap')(features)
    x = layers.BatchNormalization(name='bn')(x)
    x = layers.Dropout(0.5, name='drop1')(x)  # Higher dropout for small dataset
    x = layers.Dense(128, activation='relu', 
                     kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                     name='dense')(x)
    x = layers.Dropout(0.3, name='drop2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='PlantClassifier')
    return model, base_model


def class_weights(train_dir, class_names):
    counts = np.array([len(list((train_dir / c).glob('*.*'))) for c in class_names], dtype=np.float32)
    total = counts.sum()
    w = total / (len(class_names) * counts + 1e-5)
    w = w / w.mean()
    return {i: float(w[i]) for i in range(len(class_names))}


def confusion_matrix_np(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def train():
    t0 = time.time()
    train_ds, val_ds, test_ds, class_names = build_pipelines()
    num_classes = len(class_names)
    cw = class_weights(TRAIN_DIR, class_names)
    
    model, base_model = build_model(num_classes)
    
    # Phase 1: Train head with label smoothing
    print("\n=== Phase 1: Training Classification Head (20 epochs) ===", flush=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3')]
    )
    
    phase1_cb = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]
    
    model.fit(train_ds, validation_data=val_ds, epochs=20, class_weight=cw,
              callbacks=phase1_cb, verbose=1)
    
    # Phase 2: Fine-tune top 15 backbone layers (conservative)
    print("\n=== Phase 2: Fine-Tuning Top 15 Backbone Layers (10 epochs) ===", flush=True)
    base_model.trainable = True
    freeze_at = len(base_model.layers) - 15  # Only unfreeze top 15
    for layer in base_model.layers[:freeze_at]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3')]
    )
    
    phase2_cb = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]
    
    model.fit(train_ds, validation_data=val_ds, epochs=10, class_weight=cw,
              callbacks=phase2_cb, verbose=1)
    
    # Save model
    model.save(str(MODEL_KERAS_PATH))
    print(f"\n[OK] Keras model saved to {MODEL_KERAS_PATH}", flush=True)
    
    # Export TFLite
    print("\n=== Converting to TFLite ===", flush=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_data = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_data)
    print(f"[OK] TFLite saved to {TFLITE_PATH} ({len(tflite_data)/1024/1024:.1f} MB)", flush=True)
    
    # Evaluate on test set
    print("\n=== Test Set Evaluation ===", flush=True)
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    
    y_true_all = []
    y_pred_all = []
    latencies = []
    
    for images_batch, labels_batch in test_ds:
        for i in range(images_batch.shape[0]):
            img = np.expand_dims(images_batch[i].numpy(), axis=0)
            true_idx = int(np.argmax(labels_batch[i].numpy()))
            
            t1 = time.perf_counter()
            interpreter.set_tensor(in_det[0]['index'], img)
            interpreter.invoke()
            raw = interpreter.get_tensor(out_det[0]['index'])[0]
            latencies.append((time.perf_counter() - t1) * 1000)
            
            pred_idx = int(np.argmax(raw))
            y_true_all.append(true_idx)
            y_pred_all.append(pred_idx)
    
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    overall_acc = np.mean(y_true_all == y_pred_all) * 100
    print(f"\nOVERALL TEST ACCURACY: {overall_acc:.1f}%")
    print(f"MEAN INFERENCE LATENCY: {np.mean(latencies):.1f}ms")
    
    # Per-class metrics
    cm = confusion_matrix_np(y_true_all, y_pred_all, num_classes)
    print(f"\n{'Class':<16} {'Prec':>5} {'Rec':>5} {'F1':>5} {'Cnt':>4}")
    print("-" * 40)
    for i, cls in enumerate(class_names):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        cnt = cm[i, :].sum()
        marker = " !" if rec < 0.4 and cnt > 0 else ""
        print(f"{cls:<16} {prec*100:>4.0f}% {rec*100:>4.0f}% {f1*100:>4.0f}% {cnt:>4}{marker}")
    
    # Exemplar tests
    print("\n=== KEY EXEMPLAR TESTS ===", flush=True)
    from detection.preprocessing import preprocess_image_from_path
    
    exemplars = [
        ("Brinjal", "media/uploaded_images/brinjal_nZ64tNt.jpeg"),
        ("Aloe Vera", "media/uploaded_images/alovera_pC3w3ZJ.jpeg"),
        ("Hibiscus", "media/uploaded_images/hibiscus_c1hiAku.jpeg"),
        ("Rose", "media/uploaded_images/rose.jpeg"),
        ("Banana", "media/uploaded_images/banana.jpeg"),
        ("Jasmine", "media/uploaded_images/jasmine.jpeg"),
        ("Papaya", "media/uploaded_images/papaya.jpeg"),
        ("Curry Leaf", "media/uploaded_images/curryleaf.jpeg"),
    ]
    
    for expected, rel in exemplars:
        full = PROJECT_ROOT / rel
        if not full.exists():
            print(f"  [SKIP] {rel}")
            continue
        tensor = preprocess_image_from_path(str(full))
        interpreter.set_tensor(in_det[0]['index'], tensor)
        interpreter.invoke()
        raw = interpreter.get_tensor(out_det[0]['index'])[0]
        pred_idx = int(np.argmax(raw))
        pred_cls = class_names[pred_idx]
        conf = float(raw[pred_idx]) * 100
        top3 = np.argsort(raw)[::-1][:3]
        t3_str = ", ".join([f"{class_names[j]}({raw[j]*100:.1f}%)" for j in top3])
        mark = "PASS" if pred_cls.lower() == expected.lower() else "FAIL"
        print(f"  [{mark}] {expected:<14} -> {pred_cls:<14} ({conf:.1f}%)  {t3_str}")
    
    total_time = time.time() - t0
    print(f"\nTotal training+eval time: {total_time/60:.1f} minutes", flush=True)

if __name__ == '__main__':
    train()

