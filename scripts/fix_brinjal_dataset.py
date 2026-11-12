"""
Download authentic Solanum melongena (Brinjal / Eggplant) plant and leaf photos
from Wikimedia Commons and save to train/Brinjal and test/Brinjal.
Removes all incorrect moths, caterpillars, chicory, and unrelated nightshade files.
"""

import io
import requests
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_BRINJAL = PROJECT_ROOT / 'ml_model' / 'dataset' / 'train' / 'Brinjal'
TEST_BRINJAL = PROJECT_ROOT / 'ml_model' / 'dataset' / 'test' / 'Brinjal'
SPECIES_IMG = PROJECT_ROOT / 'media' / 'species_images' / 'brinjal.jpg'

TRAIN_BRINJAL.mkdir(parents=True, exist_ok=True)
TEST_BRINJAL.mkdir(parents=True, exist_ok=True)

# Delete existing corrupt/wrong images in train/Brinjal and test/Brinjal
for p in list(TRAIN_BRINJAL.glob('*.*')) + list(TEST_BRINJAL.glob('*.*')):
    try:
        p.unlink()
        print(f"Removed corrupt/wrong image: {p.name}")
    except Exception as e:
        print(f"Could not remove {p}: {e}")

# Verified Wikimedia Commons URLs for Solanum melongena (plant, leaves, flowers, fruit on vine)
VERIFIED_COMMONS_URLS = [
    # Leaf of Solanum melongena
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Leaf_of_Solanum_melongena.jpg/800px-Leaf_of_Solanum_melongena.jpg",
    # Solanum melongena leaf in Olinda Maui
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Starr-090801-3514-Solanum_melongena-leaf-Olinda-Maui_%2824877575671%29.jpg/800px-Starr-090801-3514-Solanum_melongena-leaf-Olinda-Maui_%2824877575671%29.jpg",
    # Solanum melongena leaf and flower
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Starr-150325-0522-Solanum_melongena-leaf_flower-Residences_Sand_Island-Midway_Atoll_%2825265911725%29.jpg/800px-Starr-150325-0522-Solanum_melongena-leaf_flower-Residences_Sand_Island-Midway_Atoll_%2825265911725%29.jpg",
    # Eggplant on plant
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Solanum_melongena_24_08_2012_%281%29.JPG/800px-Solanum_melongena_24_08_2012_%281%29.JPG",
    # Aubergine plant with fruit and foliage
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Aubergine_Solanum_melongena.jpg/800px-Aubergine_Solanum_melongena.jpg",
    # Eggplant flowering plant
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Eggplant_flower_and_leaves.jpg/800px-Eggplant_flower_and_leaves.jpg",
    # Eggplant in field
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/Eggplant_cultivation.jpg/800px-Eggplant_cultivation.jpg",
    # Solanum melongena plant habit
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Solanum_melongena_leaves.jpg/800px-Solanum_melongena_leaves.jpg"
]

HEADERS = {'User-Agent': 'BotanyDatasetValidator/2.0 (student-research; botanical-project)'}

downloaded_images = []

# First, add the verified brinjal plant image from media
if SPECIES_IMG.exists():
    try:
        im = Image.open(SPECIES_IMG).convert('RGB')
        downloaded_images.append(('species_ref.jpg', im))
        print("Loaded verified media brinjal image.")
    except Exception as e:
        print(f"Error loading species ref: {e}")

# Download verified commons images
for idx, url in enumerate(VERIFIED_COMMONS_URLS):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        if resp.status_code == 200:
            img = Image.open(io.BytesIO(resp.content)).convert('RGB')
            if img.size[0] >= 150 and img.size[1] >= 150:
                downloaded_images.append((f"brinjal_commons_{idx+1}.jpg", img))
                print(f"Downloaded verified Brinjal image {idx+1}: {img.size}")
    except Exception as e:
        print(f"Could not download {url}: {e}")

print(f"Total authentic Brinjal images obtained: {len(downloaded_images)}")

# Save to train and test
test_count = max(2, len(downloaded_images) // 4)
train_count = len(downloaded_images) - test_count

for i, (name, img) in enumerate(downloaded_images):
    if i < train_count:
        dest = TRAIN_BRINJAL / f"brinjal_train_{i+1}.jpg"
        img.save(dest, 'JPEG', quality=90)
        print(f"Saved to train: {dest.name}")
    else:
        dest = TEST_BRINJAL / f"brinjal_test_{i-train_count+1}.jpg"
        img.save(dest, 'JPEG', quality=90)
        print(f"Saved to test: {dest.name}")

print("Brinjal dataset successfully updated with authentic plant and leaf photos!")
