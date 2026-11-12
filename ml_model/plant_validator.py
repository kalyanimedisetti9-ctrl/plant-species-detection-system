"""
Plant vs Non-Plant Validator Module.

Uses pre-trained ImageNet MobileNetV2 (1000 categories) + OpenCV color & texture heuristics
to reliably distinguish genuine plant and leaf images from non-plant images:
- Human hands, faces, persons, skin
- Vehicles, cars, bikes, trucks
- Mobile phones, computers, electronics
- Buildings, architecture, monuments
- Animals, birds, pets
- Food items, furniture, and everyday non-plant objects.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault('KERAS_HOME', str(BASE_DIR / '.keras'))
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

_VALIDATOR_MODEL = None
_CLASS_INDEX = None

def load_imagenet_index():
    global _CLASS_INDEX
    if _CLASS_INDEX is None:
        index_path = BASE_DIR / 'ml_model' / 'imagenet_class_index.json'
        if not index_path.exists():
            index_path = BASE_DIR / '.keras' / 'datasets' / 'imagenet_class_index.json'
        with open(index_path, 'r', encoding='utf-8') as f:
            _CLASS_INDEX = json.load(f)
    return _CLASS_INDEX


def decode_predictions_offline(preds, top=5):
    class_index = load_imagenet_index()
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(class_index[str(i)]) + (float(pred[i]),) for i in top_indices]
        results.append(result)
    return results


# Curated Semantic Categories in ImageNet 1000 classes

# 1. Botanical classes (Fruits, vegetables, flowers, fungi, potted plants, trees)
PLANT_INDICES = set()
PLANT_INDICES.add(738)  # pot (flowerpot / potted plant)
PLANT_INDICES.add(108)  # sea_anemone (frequently triggered by succulent rosettes / Aloe Vera)
PLANT_INDICES.add(580)  # greenhouse

# Produce & vegetables (cabbage, broccoli, cauliflower, zucchini, squash, cucumber, artichoke,
# bell_pepper, cardoon, mushroom, Granny_Smith, strawberry, orange, lemon, fig, pineapple,
# banana, jackfruit, custard_apple, pomegranate, hay)
for i in range(936, 959):
    PLANT_INDICES.add(i)

# Wildflowers, grains, acorns, rose hips, buckeye, mushrooms & fungi
for i in range(984, 999):
    PLANT_INDICES.add(i)

# 2. Animals (0 to 397 in ImageNet)
ANIMAL_INDICES = set(range(0, 398))
INSECT_INDICES = set(range(300, 327))  # Insects & arachnids (often present on flowers/leaves)
VERTEBRATE_ANIMAL_INDICES = ANIMAL_INDICES - INSECT_INDICES

# 3. Human and Personal Apparel Keywords
HUMAN_KEYWORDS = {
    'band_aid', 'hand', 'finger', 'face', 'person', 'man', 'woman', 'boy', 'girl',
    'suit', 'jersey', 'groom', 'bride', 'wig', 'mask', 'costume', 'jean',
    'trench_coat', 'sweatshirt', 'brassiere', 'swimming_trunks', 'bikini',
    'sunglasses', 'sunscreen', 'stethoscope', 'scuba_diver', 'lipstick',
    'ballplayer', 'bath_towel', 'bathing_cap', 'cardigan', 'cloak', 'cowboy_hat',
    'fur_coat', 'hard_hat', 'lab_coat', 'miniskirt', 'overskirt'
}

# 4. Electronics & Tech Keywords
ELECTRONICS_KEYWORDS = {
    'cellular_telephone', 'telephone', 'phone', 'ipod', 'laptop', 'desktop_computer',
    'computer_keyboard', 'monitor', 'screen', 'television', 'radio', 'remote_control',
    'modem', 'printer', 'pencil_box', 'mouse', 'typewriter', 'cassette'
}

# 5. Vehicles & Transport Keywords
VEHICLE_KEYWORDS = {
    'car', 'sports_car', 'cab', 'minivan', 'truck', 'jeep', 'limousine', 'tractor',
    'bicycle', 'bike', 'motorcycle', 'moped', 'scooter', 'airliner', 'airship',
    'speedboat', 'canoe', 'streetcar', 'trailer', 'bus', 'train', 'locomotive',
    'moving_van', 'police_van', 'fire_engine', 'garbage_truck', 'tow_truck',
    'racer', 'beach_wagon', 'convertible', 'golfcart', 'toaster', 'piggy_bank',
    'mailbox'
}

# 6. Buildings & Architecture Keywords
BUILDING_KEYWORDS = {
    'mosque', 'church', 'palace', 'castle', 'cathedral', 'dome', 'monastery',
    'suspension_bridge', 'steel_arch_bridge', 'bridge', 'water_tower', 'lighthouse',
    'beacon', 'barn', 'boathouse', 'patio', 'prison', 'vault'
}


def get_validator_model():
    """Caches and returns the full 1000-class ImageNet MobileNetV2 classifier."""
    global _VALIDATOR_MODEL
    if _VALIDATOR_MODEL is None:
        _VALIDATOR_MODEL = MobileNetV2(weights='imagenet', include_top=True)
    return _VALIDATOR_MODEL


def inspect_plant_color_and_texture(img_bgr):
    """
    Computes botanical color metrics and skin color heuristics using HSV color space.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]
    
    total_pixels = float(img_bgr.shape[0] * img_bgr.shape[1])
    
    # Green hue mask in OpenCV: H: 25-90, S >= 30, V >= 25
    green_mask = (h >= 25) & (h <= 90) & (s >= 30) & (v >= 25)
    green_ratio = float(np.sum(green_mask)) / total_pixels
    
    # Human skin detection in HSV: H: 0-25, S: 35-180, V: 50-255
    skin_mask = (h >= 0) & (h <= 25) & (s >= 35) & (s <= 180) & (v >= 50)
    skin_ratio = float(np.sum(skin_mask)) / total_pixels
    
    # Flower colors (Vibrant reds, pinks, yellows, purples)
    vibrant_mask = (s >= 60) & (v >= 50)
    vibrant_ratio = float(np.sum(vibrant_mask)) / total_pixels
    
    return {
        'green_ratio': green_ratio,
        'skin_ratio': skin_ratio,
        'vibrant_ratio': vibrant_ratio
    }


def validate_is_plant(image_path):
    """
    Validates if an image contains a genuine plant or is a non-plant/hand/object.

    Returns:
        dict: {
            'is_plant': bool,
            'is_human_or_hand': bool,
            'message': str,
            'reason': str,
            'top_detected_category': str,
            'confidence': float
        }
    """
    if not os.path.exists(image_path):
        return {
            'is_plant': False,
            'is_human_or_hand': False,
            'message': 'Image file not found.',
            'reason': 'file_missing',
            'top_detected_category': 'None',
            'confidence': 0.0
        }
    
    try:
        with open(image_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception:
        img_bgr = cv2.imread(str(image_path))
        
    if img_bgr is None:
        return {
            'is_plant': False,
            'is_human_or_hand': False,
            'message': 'Plant Not Detected! Please upload a valid plant or leaf image.',
            'reason': 'corrupt_image',
            'top_detected_category': 'None',
            'confidence': 0.0
        }
        
    color_stats = inspect_plant_color_and_texture(img_bgr)
    green_ratio = color_stats['green_ratio']
    skin_ratio = color_stats['skin_ratio']
    
    # Run pre-trained ImageNet classifier
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    img_batch = np.expand_dims(img_resized.astype(np.float32), axis=0)
    img_preprocessed = preprocess_input(img_batch.copy())
    
    model = get_validator_model()
    preds = model.predict(img_preprocessed, verbose=0)[0]
    top_indices = preds.argsort()[-5:][::-1]
    class_index = load_imagenet_index()
    
    top1_idx = top_indices[0]
    top1_name = class_index[str(top1_idx)][1]
    top1_prob = float(preds[top1_idx])
    
    plant_score = sum(float(preds[i]) for i in top_indices if i in PLANT_INDICES)
    animal_score = sum(float(preds[i]) for i in top_indices if i in VERTEBRATE_ANIMAL_INDICES)
    
    human_score = 0.0
    electronics_score = 0.0
    vehicle_score = 0.0
    building_score = 0.0
    artifact_score = 0.0
    
    for i in top_indices:
        prob = float(preds[i])
        name = class_index[str(i)][1].lower()
        if any(k in name for k in HUMAN_KEYWORDS):
            human_score += prob
        if any(k in name for k in ELECTRONICS_KEYWORDS):
            electronics_score += prob
        if any(k in name for k in VEHICLE_KEYWORDS):
            vehicle_score += prob
        if any(k in name for k in BUILDING_KEYWORDS):
            building_score += prob
        if (398 <= i <= 935) and (i not in [738, 580]):
            artifact_score += prob
            
    # STAGE 1: HUMAN HAND OR PERSON REJECTION
    if (skin_ratio >= 0.35 and plant_score < 0.15) or human_score >= 0.15 or (skin_ratio >= 0.25 and green_ratio < 0.10):
        return {
            'is_plant': False,
            'is_human_or_hand': True,
            'message': 'Not a Plant \u2013 Please upload a valid plant image.',
            'reason': 'human_or_hand_detected',
            'top_detected_category': top1_name,
            'confidence': top1_prob
        }
        
    # STAGE 2: VERTEBRATE ANIMALS REJECTION (Dogs, cats, mammals, birds)
    if animal_score >= 0.20 or (top1_idx in VERTEBRATE_ANIMAL_INDICES and top1_prob >= 0.12):
        return {
            'is_plant': False,
            'is_human_or_hand': False,
            'message': 'Plant Not Detected! Please upload a valid plant or leaf image.',
            'reason': 'animal_detected',
            'top_detected_category': top1_name,
            'confidence': top1_prob
        }
        
    # STAGE 3: ELECTRONICS, VEHICLES, OR BUILDINGS REJECTION
    if electronics_score >= 0.15 or vehicle_score >= 0.15 or building_score >= 0.20:
        return {
            'is_plant': False,
            'is_human_or_hand': False,
            'message': 'Plant Not Detected! Please upload a valid plant or leaf image.',
            'reason': 'man_made_object_or_structure',
            'top_detected_category': top1_name,
            'confidence': top1_prob
        }
        
    # STAGE 4: EXPLICIT BOTANICAL CLASS MATCH
    if plant_score >= 0.08 or top1_idx in PLANT_INDICES:
        return {
            'is_plant': True,
            'is_human_or_hand': False,
            'message': '',
            'reason': 'botanical_class_match',
            'top_detected_category': top1_name,
            'confidence': top1_prob
        }
        
    # STAGE 5: DOMINANT MAN-MADE ARTIFACT / OBJECT REJECTION
    if artifact_score >= 0.30 and plant_score < 0.05:
        return {
            'is_plant': False,
            'is_human_or_hand': False,
            'message': 'Plant Not Detected! Please upload a valid plant or leaf image.',
            'reason': 'dominant_artifact',
            'top_detected_category': top1_name,
            'confidence': top1_prob
        }
        
    # STAGE 6: HIGH GREEN FOLIAGE / CHLOROPHYLL VERIFICATION
    if green_ratio >= 0.25 and (animal_score + electronics_score + vehicle_score + building_score < 0.20):
        return {
            'is_plant': True,
            'is_human_or_hand': False,
            'message': '',
            'reason': 'foliage_chlorophyll_verified',
            'top_detected_category': top1_name,
            'confidence': top1_prob
        }
        
    # STAGE 7: DEFAULT REJECTION FOR UNRECOGNIZABLE / NON-PLANT
    return {
        'is_plant': False,
        'is_human_or_hand': False,
        'message': 'Plant Not Detected! Please upload a valid plant or leaf image.',
        'reason': 'non_plant_subject',
        'top_detected_category': top1_name,
        'confidence': top1_prob
    }
