"""
Fetch authentic photographs for missing and under-represented plant species:
Dandelion (leaves + flowers), Money Plant (leaves),
Coriander, Mint, Spinach, Fenugreek, Marigold, Cotton, Potato, Brinjal, Chilli.
Sources: Wikimedia Commons / Wikipedia API & verified botanical sources.
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

HEADERS = {
    'User-Agent': 'BotanicalClassifierBot/2.0 (student-academic-research; contact: plantresearch@edu.org)'
}

QUERIES_BY_SPECIES = {
    "Dandelion": [
        "Taraxacum officinale leaf",
        "Dandelion leaves foliage",
        "Taraxacum officinale foliage",
        "Common dandelion leaf rosette",
        "Taraxacum officinale flower",
        "Dandelion plant green",
        "Taraxacum officinale"
    ],
    "Money Plant": [
        "Epipremnum aureum leaf",
        "Epipremnum aureum foliage",
        "Golden pothos leaf",
        "Devil's ivy money plant",
        "Epipremnum aureum vine",
        "Pothos plant indoor"
    ],
    "Coriander": [
        "Coriandrum sativum leaf",
        "Coriander leaves herb",
        "Coriandrum sativum plant",
        "Cilantro leaf green",
        "Coriander foliage herb"
    ],
    "Mint": [
        "Mentha spicata leaf",
        "Mint leaves herbal",
        "Mentha arvensis foliage",
        "Spearmint plant leaf",
        "Peppermint leaf green"
    ],
    "Spinach": [
        "Spinacia oleracea leaf",
        "Spinach leaves vegetable",
        "Spinacia oleracea plant",
        "Spinach foliage fresh",
        "Spinach vegetable garden"
    ],
    "Fenugreek": [
        "Trigonella foenum-graecum leaf",
        "Fenugreek leaves methi",
        "Trigonella foenum-graecum plant",
        "Fenugreek foliage herb",
        "Methi plant greens"
    ],
    "Marigold": [
        "Tagetes erecta flower",
        "African marigold plant",
        "Tagetes patula bloom",
        "Marigold foliage and flower",
        "Tagetes flower blossom"
    ],
    "Cotton": [
        "Gossypium hirsutum plant",
        "Cotton plant leaf",
        "Cotton boll and leaves",
        "Gossypium leaf foliage",
        "Upland cotton plant"
    ],
    "Potato": [
        "Solanum tuberosum leaf",
        "Potato plant foliage",
        "Solanum tuberosum plant",
        "Potato leaves field",
        "Potato vine green leaf"
    ],
    "Brinjal": [
        "Solanum melongena leaf",
        "Eggplant plant foliage",
        "Solanum melongena flower and leaf",
        "Brinjal plant leaves",
        "Aubergine plant leaf"
    ],
    "Chilli": [
        "Capsicum annuum leaf",
        "Chilli pepper plant foliage",
        "Capsicum annuum plant green",
        "Chili pepper leaves",
        "Capsicum leaf green"
    ]
}

def search_wikimedia_images(query, max_count=5):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'generator': 'search',
        'gsrsearch': query,
        'gsrlimit': max_count * 2,
        'prop': 'pageimages',
        'pithumbsize': 800,
        'format': 'json'
    }
    images = []
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=12)
        if r.status_code == 200:
            data = r.json()
            pages = data.get('query', {}).get('pages', {})
            for pid, pdata in pages.items():
                thumb = pdata.get('thumbnail')
                if thumb and 'source' in thumb:
                    img_url = thumb['source']
                    try:
                        res = requests.get(img_url, headers=HEADERS, timeout=12)
                        if res.status_code == 200:
                            img = Image.open(io.BytesIO(res.content)).convert('RGB')
                            if img.size[0] >= 150 and img.size[1] >= 150:
                                images.append(img)
                                if len(images) >= max_count:
                                    break
                    except Exception:
                        continue
    except Exception as e:
        print(f"Error querying '{query}': {e}")
    return images

def main():
    print("=== Downloading Botanical Images for Missing and Priority Species ===")
    for species, queries in QUERIES_BY_SPECIES.items():
        slug = species.lower().replace(' ', '_')
        train_folder = DATASET_TRAIN_DIR / species
        test_folder = DATASET_TEST_DIR / species
        train_folder.mkdir(parents=True, exist_ok=True)
        test_folder.mkdir(parents=True, exist_ok=True)

        collected = []
        for q in queries:
            print(f"[{species}] Searching: {q}...")
            imgs = search_wikimedia_images(q, max_count=4)
            for im in imgs:
                collected.append(im)
            time.sleep(0.3)
            if len(collected) >= 8:
                break

        print(f"[{species}] Total images collected: {len(collected)}")
        if collected:
            # Save representative card image
            card_path = MEDIA_SPECIES_DIR / f"{slug}.jpg"
            test_card_path = TEST_SAMPLES_DIR / f"{slug}.jpg"
            primary = collected[0]
            card_img = primary.copy()
            card_img.thumbnail((800, 600), Image.Resampling.LANCZOS)
            card_img.save(str(card_path), "JPEG", quality=92)
            card_img.save(str(test_card_path), "JPEG", quality=92)
            print(f"  -> Saved cover image to {card_path.name}")

            # Save train images
            for i, im in enumerate(collected):
                if i < len(collected) - 2 or len(collected) <= 3:
                    im.save(str(train_folder / f"{slug}_{i+1}.jpg"), "JPEG", quality=90)
                else:
                    im.save(str(test_folder / f"{slug}_test_{i+1}.jpg"), "JPEG", quality=90)
        else:
            print(f"  -> Warning: No images found for {species}")

if __name__ == '__main__':
    main()
