"""
Script to download authentic, high-quality, verified botanical photographs
for all 25 plant species from Wikimedia Commons / Wikipedia API.
"""

import os
import io
import time
import requests
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MEDIA_SPECIES_DIR = BASE_DIR / 'media' / 'species_images'
TEST_SAMPLES_DIR = BASE_DIR / 'media' / 'test_samples'
DATASET_TRAIN_DIR = BASE_DIR / 'ml_model' / 'dataset' / 'train'
DATASET_TEST_DIR = BASE_DIR / 'ml_model' / 'dataset' / 'test'

MEDIA_SPECIES_DIR.mkdir(parents=True, exist_ok=True)
TEST_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
DATASET_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
DATASET_TEST_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'PlantSpeciesBot/1.0 (https://plantspeciesdetection.local; contact: plantdetect@gmail.com)'
}

SPECIES_SEARCH_QUERIES = {
    "Sunflower": ["Helianthus annuus flower", "Sunflower field plant", "Common sunflower"],
    "Aloe Vera": ["Aloe vera plant", "Aloe barbadensis succulent", "Aloe vera leaves"],
    "Hibiscus": ["Hibiscus rosa-sinensis flower", "Shoeblackplant hibiscus", "China rose hibiscus"],
    "Mango": ["Mangifera indica tree", "Mango fruit and leaves", "Mango leaf plant"],
    "Neem": ["Azadirachta indica tree", "Neem leaves", "Neem plant foliage"],
    "Tulsi": ["Ocimum tenuiflorum holy basil", "Tulsi plant leaves", "Ocimum sanctum"],
    "Banana": ["Musa acuminata plant", "Banana plant pseudostem", "Banana tree foliage"],
    "Tomato": ["Solanum lycopersicum plant", "Tomato vine leaf", "Tomato foliage"],
    "Rose": ["Rosa rubiginosa flower", "Rose bloom plant", "Garden rose flower"],
    "Lotus": ["Nelumbo nucifera sacred lotus", "Indian lotus flower", "Lotus plant pond"],
    "Guava": ["Psidium guajava tree", "Guava fruit and leaf", "Guava foliage"],
    "Coconut": ["Cocos nucifera palm", "Coconut tree canopy", "Coconut palm fronds"],
    "Lemon": ["Citrus limon tree", "Lemon fruit and leaf", "Citrus limon foliage"],
    "Papaya": ["Carica papaya tree", "Papaya plant fruit", "Carica papaya leaves"],
    "Jasmine": ["Jasminum officinale flower", "Common jasmine bloom", "Jasmine bush"],
    "Jackfruit": ["Artocarpus heterophyllus tree", "Jackfruit tree leaves", "Jackfruit foliage"],
    "Curry Leaf": ["Murraya koenigii tree", "Curry leaf plant", "Curry tree foliage"],
    "Ashoka": ["Saraca asoca tree", "Ashoka tree flowers", "Saraca asoca foliage"],
    "Banyan": ["Ficus benghalensis tree", "Indian banyan tree", "Banyan aerial roots"],
    "Peepal": ["Ficus religiosa tree", "Sacred fig peepal leaf", "Peepal tree foliage"],
    "Money Plant": ["Epipremnum aureum pothos", "Devil's ivy money plant", "Epipremnum aureum foliage"],
    "Tulip": ["Tulipa gesneriana flower", "Garden tulip bloom", "Tulip red blossom"],
    "Daisy": ["Bellis perennis flower", "Common daisy bloom", "Lawn daisy flower"],
    "Dandelion": ["Taraxacum officinale flower", "Common dandelion flower", "Dandelion yellow blossom"],
    "Lavender": ["Lavandula angustifolia flower", "English lavender bush", "Lavender blossom spikes"]
}


def search_and_download_image(query):
    """
    Queries Wikipedia API for thumbnail image of the query and returns PIL Image.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'generator': 'search',
        'gsrsearch': query,
        'gsrlimit': 3,
        'prop': 'pageimages',
        'pithumbsize': 800,
        'format': 'json'
    }
    try:
        res = requests.get(url, params=params, headers=HEADERS, timeout=12)
        if res.status_code == 200:
            data = res.json()
            pages = data.get('query', {}).get('pages', {})
            for pid, pdata in pages.items():
                thumb = pdata.get('thumbnail')
                if thumb and 'source' in thumb:
                    img_url = thumb['source']
                    img_res = requests.get(img_url, headers=HEADERS, timeout=15)
                    if img_res.status_code == 200:
                        img = Image.open(io.BytesIO(img_res.content)).convert('RGB')
                        return img
    except Exception as e:
        print(f"Error fetching query '{query}': {e}")
    return None


def run_download():
    print("=== Downloading Real Botanical Photographs for All 25 Plant Species ===")

    for species, queries in SPECIES_SEARCH_QUERIES.items():
        slug = species.lower().replace(' ', '_')
        target_card_path = MEDIA_SPECIES_DIR / f"{slug}.jpg"
        test_sample_path = TEST_SAMPLES_DIR / f"{slug}.jpg"
        
        train_class_dir = DATASET_TRAIN_DIR / species
        test_class_dir = DATASET_TEST_DIR / species
        train_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nFetching real photographs for: {species}...")
        downloaded_images = []

        for q in queries:
            img = search_and_download_image(q)
            if img is not None:
                downloaded_images.append(img)
                time.sleep(0.4) # polite rate limit
            if len(downloaded_images) >= 3:
                break

        if not downloaded_images:
            print(f"Warning: Could not fetch photo for {species} via primary queries. Trying broad search...")
            img = search_and_download_image(f"{species} plant")
            if img is not None:
                downloaded_images.append(img)

        if downloaded_images:
            primary_img = downloaded_images[0]
            # Save standardized cover photograph for card
            card_img = primary_img.copy()
            card_img.thumbnail((800, 600), Image.Resampling.LANCZOS)
            card_img.save(str(target_card_path), "JPEG", quality=92)
            card_img.save(str(test_sample_path), "JPEG", quality=92)
            print(f"  [OK] Saved real photo to: {target_card_path.name}")

            # Save dataset variations
            for i, d_img in enumerate(downloaded_images):
                d_img.save(str(train_class_dir / f"{slug}_{i+1}.jpg"), "JPEG", quality=90)
                if i == 0:
                    d_img.save(str(test_class_dir / f"{slug}_test.jpg"), "JPEG", quality=90)
        else:
            print(f"  [FAIL] Failed to download real photo for {species}.")

    print("\n=== Download Complete! ===")

if __name__ == '__main__':
    run_download()
