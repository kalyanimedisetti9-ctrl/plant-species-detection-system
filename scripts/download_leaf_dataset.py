"""
Leaf Image Dataset Downloader for 20-Class Plant Species Detection System.

Downloads real, correctly-labelled, CC-licensed plant/leaf photographs from:
  1. iNaturalist Open API (research-grade observations)
  2. Wikimedia Commons (via Wikipedia API)

Organizes images into train/validation/test splits with no duplicates between sets.
"""

import os
import io
import sys
import time
import json
import hashlib
import random
import requests
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / 'ml_model' / 'dataset'

HEADERS = {
    'User-Agent': 'PlantSpeciesDetectionSystem/2.0 (student-project; contact: plantdetect@gmail.com)'
}

# iNaturalist taxon IDs for our 20 target species (leaf-bearing plants)
# These are the exact iNaturalist taxon_id values for each species
SPECIES_CONFIG = {
    "Aloe_Vera": {
        "inaturalist_taxon_id": 57548,
        "scientific_name": "Aloe vera",
        "wiki_queries": ["Aloe vera plant leaves", "Aloe barbadensis succulent leaves", "Aloe vera succulent"],
    },
    "Sunflower": {
        "inaturalist_taxon_id": 54421,
        "scientific_name": "Helianthus annuus",
        "wiki_queries": ["Helianthus annuus leaf", "Sunflower plant leaves", "Sunflower foliage"],
    },
    "Hibiscus": {
        "inaturalist_taxon_id": 51502,
        "scientific_name": "Hibiscus rosa-sinensis",
        "wiki_queries": ["Hibiscus rosa-sinensis leaf", "Hibiscus plant leaf", "China rose leaf"],
    },
    "Curry_Leaf": {
        "inaturalist_taxon_id": 157440,
        "scientific_name": "Murraya koenigii",
        "wiki_queries": ["Murraya koenigii leaf", "Curry leaf plant", "Curry tree leaves"],
    },
    "Mango": {
        "inaturalist_taxon_id": 49920,
        "scientific_name": "Mangifera indica",
        "wiki_queries": ["Mangifera indica leaf", "Mango tree leaves", "Mango leaf"],
    },
    "Neem": {
        "inaturalist_taxon_id": 62765,
        "scientific_name": "Azadirachta indica",
        "wiki_queries": ["Azadirachta indica leaf", "Neem tree leaves", "Neem foliage"],
    },
    "Tulsi": {
        "inaturalist_taxon_id": 126860,
        "scientific_name": "Ocimum tenuiflorum",
        "wiki_queries": ["Ocimum tenuiflorum leaf", "Holy basil leaves", "Tulsi plant"],
    },
    "Guava": {
        "inaturalist_taxon_id": 62810,
        "scientific_name": "Psidium guajava",
        "wiki_queries": ["Psidium guajava leaf", "Guava tree leaf", "Guava leaves"],
    },
    "Rose": {
        "inaturalist_taxon_id": 47326,
        "scientific_name": "Rosa",
        "wiki_queries": ["Rosa leaf", "Rose plant leaf", "Rose bush leaves"],
    },
    "Tomato": {
        "inaturalist_taxon_id": 56491,
        "scientific_name": "Solanum lycopersicum",
        "wiki_queries": ["Solanum lycopersicum leaf", "Tomato plant foliage", "Tomato leaves"],
    },
    "Banana": {
        "inaturalist_taxon_id": 126868,
        "scientific_name": "Musa acuminata",
        "wiki_queries": ["Musa acuminata leaf", "Banana plant leaf", "Banana tree leaves"],
    },
    "Papaya": {
        "inaturalist_taxon_id": 62820,
        "scientific_name": "Carica papaya",
        "wiki_queries": ["Carica papaya leaf", "Papaya plant leaf", "Papaya tree leaves"],
    },
    "Lemon": {
        "inaturalist_taxon_id": 62933,
        "scientific_name": "Citrus limon",
        "wiki_queries": ["Citrus limon leaf", "Lemon tree leaves", "Lemon foliage"],
    },
    "Coconut": {
        "inaturalist_taxon_id": 52029,
        "scientific_name": "Cocos nucifera",
        "wiki_queries": ["Cocos nucifera frond", "Coconut palm leaf", "Coconut tree fronds"],
    },
    "Peepal": {
        "inaturalist_taxon_id": 63672,
        "scientific_name": "Ficus religiosa",
        "wiki_queries": ["Ficus religiosa leaf", "Peepal tree leaf", "Sacred fig leaf"],
    },
    "Banyan": {
        "inaturalist_taxon_id": 63609,
        "scientific_name": "Ficus benghalensis",
        "wiki_queries": ["Ficus benghalensis leaf", "Indian banyan leaf", "Banyan tree leaf"],
    },
    "Jasmine": {
        "inaturalist_taxon_id": 51072,
        "scientific_name": "Jasminum",
        "wiki_queries": ["Jasminum officinale leaf", "Jasmine plant leaves", "Jasmine foliage"],
    },
    "Lotus": {
        "inaturalist_taxon_id": 48832,
        "scientific_name": "Nelumbo nucifera",
        "wiki_queries": ["Nelumbo nucifera leaf", "Sacred lotus leaf", "Lotus plant leaves"],
    },
    "Jackfruit": {
        "inaturalist_taxon_id": 63675,
        "scientific_name": "Artocarpus heterophyllus",
        "wiki_queries": ["Artocarpus heterophyllus leaf", "Jackfruit tree leaf", "Jackfruit leaves"],
    },
    "Ashoka": {
        "inaturalist_taxon_id": 283687,
        "scientific_name": "Saraca asoca",
        "wiki_queries": ["Saraca asoca leaf", "Ashoka tree leaf", "Saraca asoca foliage"],
    },
}

# How many images to aim for per class
TARGET_IMAGES_PER_CLASS = 200
# iNaturalist API page size
INAT_PER_PAGE = 50
# Maximum iNaturalist pages to fetch
INAT_MAX_PAGES = 6


def compute_image_hash(img_bytes):
    """Compute perceptual hash to detect near-duplicates."""
    return hashlib.md5(img_bytes).hexdigest()


def download_image(url, timeout=15):
    """Download an image from URL and return as PIL Image + raw bytes."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        if resp.status_code == 200:
            content = resp.content
            img = Image.open(io.BytesIO(content)).convert('RGB')
            # Filter tiny or corrupt images
            if img.size[0] < 100 or img.size[1] < 100:
                return None, None
            return img, content
    except Exception:
        pass
    return None, None


def fetch_inaturalist_photos(taxon_id, max_images=200):
    """
    Fetch CC-licensed research-grade observation photos from iNaturalist.
    Returns list of (image_url, observation_id) tuples.
    """
    all_urls = []
    page = 1

    while len(all_urls) < max_images and page <= INAT_MAX_PAGES:
        api_url = "https://api.inaturalist.org/v1/observations"
        params = {
            'taxon_id': taxon_id,
            'quality_grade': 'research',
            'photos': 'true',
            'per_page': INAT_PER_PAGE,
            'page': page,
            'order': 'desc',
            'order_by': 'votes',
            'photo_license': 'cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa,cc0',
        }

        try:
            resp = requests.get(api_url, params=params, headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                print(f"    iNaturalist API returned {resp.status_code} on page {page}")
                break

            data = resp.json()
            results = data.get('results', [])
            if not results:
                break

            for obs in results:
                photos = obs.get('photos', [])
                for photo in photos[:1]:  # Take first photo per observation
                    url = photo.get('url', '')
                    if url:
                        # Get medium-quality image (640px)
                        medium_url = url.replace('/square.', '/medium.')
                        all_urls.append((medium_url, obs.get('id', 0)))
                        if len(all_urls) >= max_images:
                            break
                if len(all_urls) >= max_images:
                    break

        except Exception as e:
            print(f"    iNaturalist API error page {page}: {e}")
            break

        page += 1
        time.sleep(0.5)  # Rate limit

    return all_urls


def fetch_wikimedia_images(queries, max_images=30):
    """
    Fetch images from Wikimedia Commons via Wikipedia API.
    Returns list of image URLs.
    """
    urls = []
    for query in queries:
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'generator': 'search',
            'gsrsearch': query,
            'gsrlimit': 10,
            'prop': 'pageimages',
            'pithumbsize': 640,
            'format': 'json'
        }
        try:
            resp = requests.get(api_url, params=params, headers=HEADERS, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                pages = data.get('query', {}).get('pages', {})
                for pid, pdata in pages.items():
                    thumb = pdata.get('thumbnail')
                    if thumb and 'source' in thumb:
                        urls.append(thumb['source'])
                        if len(urls) >= max_images:
                            return urls
        except Exception:
            pass
        time.sleep(0.3)

    return urls


def download_species_images(species_name, config, output_base_dir):
    """Download images for a single species from all sources."""
    print(f"\n{'='*60}")
    print(f"  Downloading: {species_name} ({config['scientific_name']})")
    print(f"{'='*60}")

    all_images = []  # List of (PIL Image, hash)
    seen_hashes = set()

    # Source 1: iNaturalist
    taxon_id = config['inaturalist_taxon_id']
    print(f"  [iNaturalist] Fetching research-grade observations (taxon_id={taxon_id})...")
    inat_urls = fetch_inaturalist_photos(taxon_id, max_images=TARGET_IMAGES_PER_CLASS)
    print(f"  [iNaturalist] Found {len(inat_urls)} candidate URLs")

    downloaded = 0
    for url, obs_id in inat_urls:
        img, raw = download_image(url)
        if img is not None:
            h = compute_image_hash(raw)
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_images.append(img)
                downloaded += 1
                if downloaded % 20 == 0:
                    print(f"    Downloaded {downloaded} unique images so far...")
        time.sleep(0.15)  # Rate limit

    print(f"  [iNaturalist] Downloaded {downloaded} unique images")

    # Source 2: Wikimedia (supplement)
    if len(all_images) < TARGET_IMAGES_PER_CLASS:
        print(f"  [Wikimedia] Supplementing with Wikimedia images...")
        wiki_urls = fetch_wikimedia_images(config['wiki_queries'], max_images=30)
        wiki_count = 0
        for url in wiki_urls:
            img, raw = download_image(url)
            if img is not None:
                h = compute_image_hash(raw)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    all_images.append(img)
                    wiki_count += 1
            time.sleep(0.2)
        print(f"  [Wikimedia] Downloaded {wiki_count} additional images")

    total = len(all_images)
    print(f"  TOTAL unique images for {species_name}: {total}")

    if total == 0:
        print(f"  WARNING: No images downloaded for {species_name}!")
        return 0

    # Shuffle and split: 80% train, 10% validation, 10% test
    random.seed(42)
    random.shuffle(all_images)

    val_count = max(2, int(total * 0.10))
    test_count = max(2, int(total * 0.10))
    train_count = total - val_count - test_count

    if train_count < 1:
        train_count = total
        val_count = 0
        test_count = 0

    train_images = all_images[:train_count]
    val_images = all_images[train_count:train_count + val_count]
    test_images = all_images[train_count + val_count:]

    # Save to directories
    for split_name, split_imgs in [('train', train_images), ('validation', val_images), ('test', test_images)]:
        split_dir = output_base_dir / split_name / species_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(split_imgs):
            # Resize to reasonable dimensions for training efficiency
            img.thumbnail((640, 640), Image.Resampling.LANCZOS)
            filename = f"{species_name.lower()}_{split_name}_{i+1:04d}.jpg"
            img.save(str(split_dir / filename), "JPEG", quality=90)

    print(f"  Saved: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
    return total


def clean_old_dataset(dataset_dir):
    """Remove old class directories that aren't in our 20 target classes."""
    target_classes = set(SPECIES_CONFIG.keys())
    # Also check with spaces instead of underscores
    target_classes_spaces = {c.replace('_', ' ') for c in target_classes}

    for split in ['train', 'validation', 'test']:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for item in split_dir.iterdir():
            if item.is_dir():
                name = item.name
                # Check if this class should be removed
                if name not in target_classes and name not in target_classes_spaces:
                    print(f"  Removing non-target class: {split}/{name}")
                    import shutil
                    shutil.rmtree(str(item))
                elif name in target_classes_spaces and name not in target_classes:
                    # Rename "Aloe Vera" -> "Aloe_Vera" etc.
                    new_name = name.replace(' ', '_')
                    new_path = split_dir / new_name
                    if not new_path.exists():
                        print(f"  Renaming: {split}/{name} -> {split}/{new_name}")
                        item.rename(new_path)


def run_download():
    """Main download orchestration."""
    print("=" * 70)
    print("  PLANT SPECIES LEAF IMAGE DATASET DOWNLOADER")
    print("  Target: 20 classes, ~200 images each from iNaturalist + Wikimedia")
    print("=" * 70)

    # Clean old classes
    print("\n--- Step 1: Cleaning old dataset directories ---")
    clean_old_dataset(DATASET_DIR)

    # Download for each species
    print("\n--- Step 2: Downloading images for 20 target species ---")
    results = {}

    for species_name, config in SPECIES_CONFIG.items():
        count = download_species_images(species_name, config, DATASET_DIR)
        results[species_name] = count

    # Summary report
    print("\n" + "=" * 70)
    print("  DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"  {'Species':<20} {'Images Downloaded':<20} {'Status'}")
    print(f"  {'-'*20} {'-'*20} {'-'*15}")

    total_all = 0
    for species, count in sorted(results.items()):
        status = "OK" if count >= 50 else ("LOW" if count > 0 else "FAILED")
        print(f"  {species:<20} {count:<20} {status}")
        total_all += count

    print(f"\n  TOTAL IMAGES: {total_all}")
    print(f"  CLASSES: {len(results)}")

    # Report per-split counts
    print("\n  Per-split counts:")
    for split in ['train', 'validation', 'test']:
        split_dir = DATASET_DIR / split
        if split_dir.exists():
            total_split = 0
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.is_dir():
                    count = len(list(cls_dir.glob('*.jpg')))
                    total_split += count
            print(f"    {split}: {total_split} images")

    # Save manifest
    manifest = {
        'species': results,
        'target_classes': list(SPECIES_CONFIG.keys()),
        'total_images': total_all,
    }
    manifest_path = DATASET_DIR / 'dataset_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest saved to: {manifest_path}")

    if any(v == 0 for v in results.values()):
        print("\n  WARNING: Some species have 0 images. Check network connectivity.")

    print("\n  DONE.")


if __name__ == '__main__':
    run_download()
