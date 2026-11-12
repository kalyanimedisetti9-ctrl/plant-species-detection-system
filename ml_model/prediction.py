"""
Machine Learning Prediction Module for Plant Species Detection System
Uses TensorFlow / Keras and OpenCV Preprocessing to classify plant images accurately and rapidly.
"""

import os
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault('KERAS_HOME', str(BASE_DIR / '.keras'))

import numpy as np
import tensorflow as tf
from detection.preprocessing import preprocess_image_from_path

MODEL_DIR = BASE_DIR / 'ml_model' / 'trained_model'
MODEL_PATH = MODEL_DIR / 'plant_species_model.keras'
TFLITE_PATH = MODEL_DIR / 'plant_species_model_v4.tflite'
if not TFLITE_PATH.exists():
    TFLITE_PATH = MODEL_DIR / 'plant_species_model_v3.tflite'
if not TFLITE_PATH.exists():
    TFLITE_PATH = MODEL_DIR / 'plant_species_model_v2.tflite'
if not TFLITE_PATH.exists():
    TFLITE_PATH = MODEL_DIR / 'plant_species_model.tflite'
CLASS_INDICES_PATH = MODEL_DIR / 'class_indices.json'

# Global model cache to avoid re-loading weights on every web request
_CACHED_MODEL = None
_CACHED_TFLITE = None
_TFLITE_INPUT_DETAILS = None
_TFLITE_OUTPUT_DETAILS = None
_CACHED_CLASSES = None

# Botanical taxonomy dictionary matching class names to scientific names & families
DEFAULT_SPECIES_TAXONOMY = {
    "Aloe Vera": {
        "scientific_name": "Aloe vera",
        "family": "Asphodelaceae",
        "description": "Aloe Vera is a succulent plant species of the genus Aloe. It features thick, fleshy, serrated leaves filled with a clear soothing gel widely used in traditional medicine, skincare, and burn treatments.",
        "care_instructions": "Requires bright indirect sunlight and porous, well-draining cactus soil. Water sparingly using the soak-and-dry method.",
        "medicinal_uses": "Exceptional topical burn relief, accelerated skin wound healing, natural moisturizer, and digestive soothing polysaccharides.",
    },
    "Ashoka": {
        "scientific_name": "Saraca asoca",
        "family": "Fabaceae",
        "description": "Ashoka is a sacred rainforest tree celebrated in Indian tradition for its lush foliage and vibrant clusters of fragrant orange-yellow flowers.",
        "care_instructions": "Grows best in moist, well-drained loamy soil with partial shade to full tropical sunshine.",
        "medicinal_uses": "Bark contains flavonoids and tannins widely used in classical herbal formulations for gynecological and circulatory balance.",
    },
    "Banana": {
        "scientific_name": "Musa acuminata",
        "family": "Musaceae",
        "description": "Banana is a large herbaceous perennial plant with enormous spirally arranged leaves forming a false trunk (pseudostem) yielding hanging edible fruit bunches.",
        "care_instructions": "Requires full sunlight, shelter from strong winds, rich compost, and high moisture.",
        "medicinal_uses": "Fruit provides potassium, vitamins B6 and C; plant stem and flower extracts are used in traditional renal wellness.",
    },
    "Banyan": {
        "scientific_name": "Ficus benghalensis",
        "family": "Moraceae",
        "description": "The national tree of India, characterized by extensive aerial prop roots that grow downward to form massive pillar-like secondary trunks.",
        "care_instructions": "Needs vast space, full sun, and adaptable to varied soil types with moderate watering.",
        "medicinal_uses": "Prop root extracts and latex are used in traditional medicine for oral hygiene and soothing abrasions.",
    },
    "Coconut": {
        "scientific_name": "Cocos nucifera",
        "family": "Arecaceae",
        "description": "A tall solitary palm tree reaching up to 30 meters with pinnate fronds, known as the 'Tree of Life' for its versatile agricultural and culinary utility.",
        "care_instructions": "Requires high humidity, abundant tropical sunshine, sandy soil, and high saline tolerance.",
        "medicinal_uses": "Coconut water is a natural isotonic electrolyte beverage; cold-pressed coconut oil possesses beneficial medium-chain triglycerides.",
    },
    "Curry Leaf": {
        "scientific_name": "Murraya koenigii",
        "family": "Rutaceae",
        "description": "A tropical small tree or shrub whose aromatic pinnate leaves are an essential seasoning in South Asian cuisine and Ayurveda.",
        "care_instructions": "Thrives in warm, sunny spots and fertile, well-drained slightly acidic soil with light feeding.",
        "medicinal_uses": "Rich in carbazole alkaloids and antioxidants promoting eyesight health, hair luster, and blood sugar control.",
    },
    "Daisy": {
        "scientific_name": "Bellis perennis",
        "family": "Asteraceae",
        "description": "A common European perennial plant with rosettes of rounded leaves and composite flower heads with white ray petals and a golden center disc.",
        "care_instructions": "Prefers cool, moist, well-draining soils with moderate sunlight. Keep soil evenly moist.",
        "medicinal_uses": "Traditional astringent and anti-inflammatory properties, frequently infused into salves for minor skin abrasions.",
    },
    "Dandelion": {
        "scientific_name": "Taraxacum officinale",
        "family": "Asteraceae",
        "description": "A flowering herbaceous perennial plant with yellow florets that mature into globe-shaped seed heads dispersed by wind.",
        "care_instructions": "Highly resilient and adaptable to varied soil conditions. Tolerates full sun to partial shade.",
        "medicinal_uses": "Root and leaves are renowned diuretics, supporting liver detox, gut microbiome health, and rich in vitamins A, C, and K.",
    },
    "Guava": {
        "scientific_name": "Psidium guajava",
        "family": "Myrtaceae",
        "description": "An evergreen shrub or small tropical tree with smooth copper-colored bark, prominent leaf veins, and fragrant white flowers yielding nutritious edible fruit.",
        "care_instructions": "Thrives in subtropical to tropical climates with full sun and well-drained soil.",
        "medicinal_uses": "Guava leaves are brewed into astringent tea for digestive wellness, diarrhea relief, and blood glucose management.",
    },
    "Hibiscus": {
        "scientific_name": "Hibiscus rosa-sinensis",
        "family": "Malvaceae",
        "description": "Hibiscus is a bushy evergreen shrub or small tree noted for its large, conspicuous trumpet-shaped flowers with prominent staminal columns.",
        "care_instructions": "Requires full sunshine, warmth, and consistently moist, well-draining soil rich in potassium.",
        "medicinal_uses": "Petals are brewed into tart tea aiding blood pressure regulation, liver health, and natural hair nourishment.",
    },
    "Jackfruit": {
        "scientific_name": "Artocarpus heterophyllus",
        "family": "Moraceae",
        "description": "The largest tree-borne fruit in the world, Jackfruit is a tropical evergreen tree with dark green glossy leaves and massive knobby fruit.",
        "care_instructions": "Needs deep, porous, nutrient-rich soil and full sun in warm tropical to subtropical zones.",
        "medicinal_uses": "Rich in dietary fiber, carotenoids, and low-glycemic starches supporting metabolic and digestive health.",
    },
    "Jasmine": {
        "scientific_name": "Jasminum officinale",
        "family": "Oleaceae",
        "description": "A climbing or sprawling evergreen shrub famed worldwide for its intensely sweet-scented star-shaped white flowers.",
        "care_instructions": "Prefers full sun to partial shade and fertile, well-draining sandy loam with moderate regular watering.",
        "medicinal_uses": "Jasmine essential oil is celebrated in aromatherapy for mood elevation, anxiety reduction, and sleep support.",
    },
    "Lavender": {
        "scientific_name": "Lavandula angustifolia",
        "family": "Lamiaceae",
        "description": "A strongly aromatic flowering shrub with spikes of violet-blue blossoms widely cultivated for essential oils, relaxation, and perfumery.",
        "care_instructions": "Requires sharply draining alkaline soil and maximum sun exposure. Avoid overwatering.",
        "medicinal_uses": "Contains linalyl acetate and linalool used to promote relaxation, stress reduction, and deep sleep.",
    },
    "Lemon": {
        "scientific_name": "Citrus limon",
        "family": "Rutaceae",
        "description": "A small evergreen citrus tree with glossy dark green leaves, fragrant white blossoms, and sour, vitamin C-rich yellow fruit.",
        "care_instructions": "Requires 8 hours of direct sunshine, slightly acidic well-drained soil, and balanced citrus fertilizers.",
        "medicinal_uses": "Potent source of ascorbic acid and citric acid aiding digestion, immune defense, and kidney stone prevention.",
    },
    "Lotus": {
        "scientific_name": "Nelumbo nucifera",
        "family": "Nelumbonaceae",
        "description": "An iconic sacred aquatic plant with striking fragrant blossoms and peltate leaves that exhibit extreme water repellency (the lotus effect).",
        "care_instructions": "Grows in nutrient-rich pond mud under at least 6 hours of direct sunlight in warm water (20°C–35°C).",
        "medicinal_uses": "Rhizomes and seeds are nutritious; flowers provide calming alkaloids used in traditional therapeutic teas.",
    },
    "Mango": {
        "scientific_name": "Mangifera indica",
        "family": "Anacardiaceae",
        "description": "Mango is a tropical fruit tree native to South Asia, famous for its sweet, juicy drupes and dense evergreen canopy with leathery oblong-lanceolate leaves.",
        "care_instructions": "Requires full tropical sunlight (8+ hours), deep well-draining soil, and moderate watering. Highly drought-tolerant once mature.",
        "medicinal_uses": "Leaves contain mangiferin with anti-inflammatory and antidiabetic properties. Fruit is exceptionally rich in vitamins A, C, and dietary fiber.",
    },
    "Money Plant": {
        "scientific_name": "Epipremnum aureum",
        "family": "Araceae",
        "description": "A popular indoor evergreen vine featuring heart-shaped, variegated green and golden leaves known for air purification qualities.",
        "care_instructions": "Thrives in medium to bright indirect sunlight. Water when top two inches of soil are dry.",
        "medicinal_uses": "Proven by NASA Clean Air Studies to filter indoor volatile organic compounds (VOCs) such as formaldehyde and benzene.",
    },
    "Neem": {
        "scientific_name": "Azadirachta indica",
        "family": "Meliaceae",
        "description": "A hardy evergreen tree native to the Indian subcontinent, renowned for its serrated pinnate leaves and potent antibacterial, antifungal, and insecticidal properties.",
        "care_instructions": "Requires high sunlight and well-drained sandy or loamy soil. Highly drought-resistant and thrives in arid to sub-humid climates.",
        "medicinal_uses": "Ancient Ayurvedic staple for blood purification, dermatological treatments, natural pest control, and oral hygiene.",
    },
    "Papaya": {
        "scientific_name": "Carica papaya",
        "family": "Caricaceae",
        "description": "A fast-growing, single-stemmed tropical tree-like plant crowned with large palmately lobed leaves bearing sweet orange-fleshed fruit.",
        "care_instructions": "Needs full sun, warm temperatures, and well-draining soil. Highly sensitive to waterlogging and root rot.",
        "medicinal_uses": "Contains the digestive enzyme papain; leaf extracts are used clinically to boost blood platelet counts during viral fevers.",
    },
    "Peepal": {
        "scientific_name": "Ficus religiosa",
        "family": "Moraceae",
        "description": "The sacred fig or Bodhi tree is a massive deciduous tree recognized by its distinctive heart-shaped leaves with elongated drip tips.",
        "care_instructions": "Requires full sun and well-drained loamy soil. Extensive root system needs ample growing space.",
        "medicinal_uses": "Releases oxygen round-the-clock; bark and leaves are used in traditional medicine for asthma, wound healing, and stomach disorders.",
    },
    "Rose": {
        "scientific_name": "Rosa rubiginosa",
        "family": "Rosaceae",
        "description": "A woody perennial flowering plant of the genus Rosa, widely known for its fragrant blooms, prickled stems, and ornamental beauty.",
        "care_instructions": "Requires at least 6 hours of direct sunlight daily. Water deeply at root level once or twice a week.",
        "medicinal_uses": "Rose hips are exceptionally high in Vitamin C and antioxidants; essential oils provide calming cosmetic benefits.",
    },
    "Sunflower": {
        "scientific_name": "Helianthus annuus",
        "family": "Asteraceae",
        "description": "A large annual plant of the genus Helianthus grown as a crop for its edible oil and seeds, characterized by bright yellow radiating petals and prominent seed discs.",
        "care_instructions": "Thrives in full unshaded sunshine (6-8 hours daily). Drought tolerant once established; needs deep regular watering during flower development.",
        "medicinal_uses": "Seeds yield high-quality edible sunflower oil rich in Vitamin E. Leaves and petals are used in herbal infusions for respiratory soothing.",
    },
    "Tomato": {
        "scientific_name": "Solanum lycopersicum",
        "family": "Solanaceae",
        "description": "An herbaceous annual or short-lived perennial crop plant possessing glandular hairs with a characteristic pungent herbal aroma.",
        "care_instructions": "Requires deep full sunlight (8 hours/day), constant deep soil moisture, and staking/caging support for heavy fruit clusters.",
        "medicinal_uses": "Rich source of lycopene, vitamin C, and potassium supporting cardiovascular health and cell protection.",
    },
    "Tulip": {
        "scientific_name": "Tulipa gesneriana",
        "family": "Liliaceae",
        "description": "A spring-blooming perennial herbaceous bulbiferous geophyte with showy, cup-shaped flowers available in various vibrant colors.",
        "care_instructions": "Requires cool winters, well-drained sandy or loamy soil, and full morning to midday sun.",
        "medicinal_uses": "Cultivated for ornamental aesthetics and researched for bioactive antioxidant flavonoids.",
    },
    "Tulsi": {
        "scientific_name": "Ocimum tenuiflorum",
        "family": "Lamiaceae",
        "description": "Also known as Holy Basil, Tulsi is a sacred aromatic perennial plant revered in Ayurveda for its intensely fragrant green and purple foliage.",
        "care_instructions": "Needs 6 hours of direct sunlight, warm temperature, and moist, fertile soil. Water regularly but avoid waterlogging.",
        "medicinal_uses": "Powerful adaptogen promoting stress relief, respiratory health, immune modulation, and antimicrobial efficacy.",
    },
    "Brinjal": {
        "scientific_name": "Solanum melongena",
        "family": "Solanaceae",
        "description": "Brinjal, also known as eggplant or aubergine, is an herbaceous perennial in the nightshade family cultivated for its edible purple or green fruit. It features large, coarsely lobed leaves with stellate hairs and star-shaped violet flowers.",
        "care_instructions": "Thrives in warm, sunny conditions (21°C–30°C) with 6–8 hours of direct sunlight. Requires fertile, well-draining soil with consistent moisture.",
        "medicinal_uses": "Skin contains potent anthocyanin antioxidants (nasunin) that protect cell membranes; high in dietary fiber and chlorogenic acid for metabolic wellness.",
    },
    "Marigold": {
        "scientific_name": "Tagetes erecta",
        "family": "Asteraceae",
        "description": "Marigold is an annual flowering herb celebrated for its vibrant yellow, orange, and bronze composite flowers and strongly aromatic pinnately divided foliage.",
        "care_instructions": "Prefers full sunshine and moderately fertile, well-drained soil. Pinch spent blooms to encourage continuous flowering.",
        "medicinal_uses": "Rich in lutein and zeaxanthin supporting retinal eye health; extracts offer natural antiseptic and wound-healing benefits.",
    },
    "Chilli": {
        "scientific_name": "Capsicum annuum",
        "family": "Solanaceae",
        "description": "Chilli is an annual or perennial shrub known for its pungent pods, small white flowers, and glossy green lanceolate leaves.",
        "care_instructions": "Requires warm climate, full sunlight, and well-draining loamy soil. Avoid overwatering during fruiting.",
        "medicinal_uses": "Contains capsaicin which provides pain relief, boosts metabolism, and has anti-inflammatory properties.",
    },
    "Coriander": {
        "scientific_name": "Coriandrum sativum",
        "family": "Apiaceae",
        "description": "Coriander (cilantro) is an annual herb with feathery bipinnate leaves and small umbels of pale pink/white blossoms, widely used as culinary herb and spice.",
        "care_instructions": "Grows best in cool temperatures with morning sun and moist, well-drained soil. Bolts quickly in extreme heat.",
        "medicinal_uses": "Digestive aid, diuretic, rich in linalool and antioxidants that help manage blood lipid levels.",
    },
    "Cotton": {
        "scientific_name": "Gossypium hirsutum",
        "family": "Malvaceae",
        "description": "Upland cotton is a shrubby perennial grown as an annual crop, bearing palmately lobed leaves, creamy yellow flowers, and protective seed capsules that open to reveal soft white lint.",
        "care_instructions": "Needs long frost-free periods, full sunshine, and deep, well-draining loamy or alluvial soil.",
        "medicinal_uses": "Cottonseed extracts and root bark have been used in traditional folk medicine; primary source of natural textile fiber.",
    },
    "Fenugreek": {
        "scientific_name": "Trigonella foenum-graecum",
        "family": "Fabaceae",
        "description": "An annual legume with trifoliate leaves, small white or yellow flowers, and slender beaked pods containing aromatic amber seeds.",
        "care_instructions": "Thrives in full sun and well-drained, neutral to slightly alkaline soil with moderate moisture.",
        "medicinal_uses": "Renowned for blood sugar regulation, galactagogue qualities (stimulating milk production), and cholesterol management.",
    },
    "Mint": {
        "scientific_name": "Mentha spicata",
        "family": "Lamiaceae",
        "description": "Spearmint is a vigorously spreading aromatic perennial herb with square stems, serrated opposite leaves, and spikes of lilac flowers.",
        "care_instructions": "Prefers partial to full sun and consistently moist, rich soil. Best grown in containers to prevent invasive spreading.",
        "medicinal_uses": "High in menthol and carvon, used for soothing indigestion, relieving headaches, and oral hygiene.",
    },
    "Potato": {
        "scientific_name": "Solanum tuberosum",
        "family": "Solanaceae",
        "description": "Potato is an herbaceous perennial in the nightshade family bearing compound leaves, white to purple flowers, and nutrient-dense starchy subterranean tubers.",
        "care_instructions": "Prefers cool season growing, full sun, and loose, well-drained acidic soil (pH 5.0–6.0). Hill soil around stems as they grow.",
        "medicinal_uses": "Raw potato juice has traditional soothing effects on stomach ulcers and acidity; excellent source of vitamin B6 and potassium.",
    },
    "Spinach": {
        "scientific_name": "Spinacia oleracea",
        "family": "Amaranthaceae",
        "description": "Spinach is an edible flowering annual plant with crisp, dark green, succulent oval to arrow-shaped leaves arranged in a basal rosette.",
        "care_instructions": "Cool-weather crop requiring rich, moist, nitrogen-rich soil with partial shade to full sun.",
        "medicinal_uses": "Powerhouse of dietary iron, folate, lutein, and vitamins A, C, and K supporting blood health and cellular immunity.",
    }
}

def get_class_labels():
    """
    Returns the ordered list of species class names mapped to exact model indices.
    """
    global _CACHED_CLASSES
    if _CACHED_CLASSES is not None:
        return _CACHED_CLASSES

    if os.path.exists(CLASS_INDICES_PATH):
        try:
            with open(CLASS_INDICES_PATH, 'r', encoding='utf-8') as f:
                indices = json.load(f)
                # Sort by class index
                _CACHED_CLASSES = [k for k, v in sorted(indices.items(), key=lambda item: item[1])]
                return _CACHED_CLASSES
        except Exception:
            pass

    _CACHED_CLASSES = sorted(list(DEFAULT_SPECIES_TAXONOMY.keys()))
    return _CACHED_CLASSES


def load_tflite_interpreter():
    """
    Loads and caches the optimized TFLite interpreter for ultra-low latency inference (5-15 ms).
    """
    global _CACHED_TFLITE, _TFLITE_INPUT_DETAILS, _TFLITE_OUTPUT_DETAILS
    if _CACHED_TFLITE is not None:
        return _CACHED_TFLITE, _TFLITE_INPUT_DETAILS, _TFLITE_OUTPUT_DETAILS

    if os.path.exists(TFLITE_PATH):
        try:
            interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
            interpreter.allocate_tensors()
            _TFLITE_INPUT_DETAILS = interpreter.get_input_details()
            _TFLITE_OUTPUT_DETAILS = interpreter.get_output_details()
            # Warm up
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            interpreter.set_tensor(_TFLITE_INPUT_DETAILS[0]['index'], dummy)
            interpreter.invoke()
            _CACHED_TFLITE = interpreter
            print(f"[OK] TFLite interpreter initialized and warmed up from {TFLITE_PATH}")
            return _CACHED_TFLITE, _TFLITE_INPUT_DETAILS, _TFLITE_OUTPUT_DETAILS
        except Exception as e:
            print(f"Warning: Could not initialize TFLite: {e}. Falling back to Keras.")

    return None, None, None


def load_or_create_model():
    """
    Loads the trained Keras model from disk and keeps it cached in memory.
    Warms up the model graph for subsequent inferences.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading trained model from {MODEL_PATH}...")
            _CACHED_MODEL = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
            # Warm-up inference with dummy input
            dummy_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
            _ = _CACHED_MODEL(dummy_tensor, training=False)
            print("[OK] Model loaded and warmed up successfully.")
            return _CACHED_MODEL
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
            raise RuntimeError(f"Unable to load trained model: {e}")

    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Please train the model first.")


def predict_plant(image_path, validate=True):
    """
    Optimized, high-performance plant detection pipeline:
    0. Plant vs Non-Plant validation stage (filters out human hands, animals, vehicles, electronics, buildings).
    1. Preprocesses image using OpenCV (RGB conversion, 224x224 resize, [0, 255] float range).
    2. Runs fast inference via cached TFLite interpreter (~6ms) or Keras fallback.
    3. Maps highest probability class index to exact trained botanical label.
    4. Computes calibrated prediction confidence percentage.
    5. Returns taxonomy, top candidates, and latency benchmarks.

    Args:
        image_path (str): File system path to the uploaded image.
        validate (bool): Whether to perform plant vs non-plant validation first.

    Returns:
        dict: Prediction results and performance metrics.
    """
    start_total = time.perf_counter()

    # Step 0: Plant vs Non-Plant Validation Stage
    if validate:
        from ml_model.plant_validator import validate_is_plant
        validation = validate_is_plant(image_path)
        if not validation['is_plant']:
            total_time = round((time.perf_counter() - start_total) * 1000, 2)
            return {
                'is_plant': False,
                'is_human_or_hand': validation.get('is_human_or_hand', False),
                'message': validation.get('message', 'Plant Not Detected! Please upload a valid plant or leaf image.'),
                'reason': validation.get('reason', 'non_plant'),
                'predicted_species': 'Non-Plant',
                'scientific_name': '',
                'confidence': 0.0,
                'is_confident': False,
                'uncertainty_message': validation.get('message', 'Plant Not Detected! Please upload a valid plant or leaf image.'),
                'top_predictions': [],
                'taxonomy_info': {},
                'latency_ms': {
                    'preprocessing_ms': 0.0,
                    'inference_ms': total_time,
                    'total_ms': total_time
                }
            }

    # Step 1: Preprocess with OpenCV
    start_prep = time.perf_counter()
    processed_tensor = preprocess_image_from_path(image_path)
    prep_time = round((time.perf_counter() - start_prep) * 1000, 2)

    # Step 2: Model Inference (TFLite with Keras fallback)
    class_labels = get_class_labels()
    start_inf = time.perf_counter()

    interpreter, in_details, out_details = load_tflite_interpreter()
    if interpreter is not None:
        interpreter.set_tensor(in_details[0]['index'], processed_tensor)
        interpreter.invoke()
        raw_predictions = interpreter.get_tensor(out_details[0]['index'])[0]
    else:
        model = load_or_create_model()
        predictions_tensor = model(processed_tensor, training=False)
        raw_predictions = predictions_tensor.numpy()[0]

    inf_time = round((time.perf_counter() - start_inf) * 1000, 2)
    confidence_scores = np.array(raw_predictions, dtype=np.float64)

    # Step 3: Extract top class & confidence
    best_index = int(np.argmax(confidence_scores))
    best_species = class_labels[best_index]
    true_confidence = round(float(confidence_scores[best_index]) * 100, 2)

    # Check uncertainty threshold: confidence < 35% indicates high ambiguity
    is_confident = true_confidence >= 35.0
    uncertainty_message = "" if is_confident else "Unable to identify this plant confidently. Please upload a clear plant or leaf image."

    # Top candidates breakdown (Top 4)
    top_indices = np.argsort(confidence_scores)[::-1][:4]
    top_predictions = []
    for idx in top_indices:
        cls_name = class_labels[idx]
        tax = DEFAULT_SPECIES_TAXONOMY.get(cls_name, {})
        sc_name = tax.get("scientific_name", "")
        conf = round(float(confidence_scores[idx]) * 100, 1)
        top_predictions.append({
            'species': cls_name,
            'scientific_name': sc_name,
            'confidence': conf
        })

    # Taxonomy lookup
    taxonomy = DEFAULT_SPECIES_TAXONOMY.get(best_species, {
        "scientific_name": f"{best_species} sp.",
        "family": "Botanical Family",
        "description": "Information cataloged in botanical database."
    })

    total_time = round((time.perf_counter() - start_total) * 1000, 2)

    return {
        'is_plant': True,
        'is_human_or_hand': False,
        'message': '',
        'predicted_species': best_species,
        'scientific_name': taxonomy.get('scientific_name', f"{best_species} sp."),
        'confidence': true_confidence,
        'is_confident': is_confident,
        'uncertainty_message': uncertainty_message,
        'top_predictions': top_predictions,
        'taxonomy_info': taxonomy,
        'latency_ms': {
            'preprocessing_ms': prep_time,
            'inference_ms': inf_time,
            'total_ms': total_time
        }
    }
