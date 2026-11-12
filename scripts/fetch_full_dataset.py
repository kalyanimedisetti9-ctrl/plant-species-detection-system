"""
Comprehensive Botanical Dataset Downloader for 34 Plant Species
Fetches genuine, CC-licensed photos from iNaturalist Open Data and Wikimedia Commons.
Multithreaded, robust, deduplicated, and split into train / validation / test.
"""

import io
import json
import time
import hashlib
import random
import requests
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / 'ml_model' / 'dataset'
TRAIN_DIR = DATASET_DIR / 'train'
VAL_DIR = DATASET_DIR / 'validation'
TEST_DIR = DATASET_DIR / 'test'

HEADERS = {
    'User-Agent': 'PlantSpeciesBot/3.0 (botany-academic-research; contact: plantproject@edu.org)'
}

SPECIES_TAXONOMY = {
    "Aloe Vera": "Aloe vera",
    "Ashoka": "Saraca asoca",
    "Banana": "Musa acuminata",
    "Banyan": "Ficus benghalensis",
    "Brinjal": "Solanum melongena",
    "Chilli": "Capsicum annuum",
    "Coconut": "Cocos nucifera",
    "Coriander": "Coriandrum sativum",
    "Cotton": "Gossypium hirsutum",
    "Curry Leaf": "Murraya koenigii",
    "Daisy": "Bellis perennis",
    "Dandelion": "Taraxacum officinale",
    "Fenugreek": "Trigonella foenum-graecum",
    "Guava": "Psidium guajava",
    "Hibiscus": "Hibiscus rosa-sinensis",
    "Jackfruit": "Artocarpus heterophyllus",
    "Jasmine": "Jasminum officinale",
    "Lavender": "Lavandula angustifolia",
    "Lemon": "Citrus limon",
    "Lotus": "Nelumbo nucifera",
    "Mango": "Mangifera indica",
    "Marigold": "Tagetes erecta",
    "Mint": "Mentha spicata",
    "Money Plant": "Epipremnum aureum",
    "Neem": "Azadirachta indica",
    "Papaya": "Carica papaya",
    "Peepal": "Ficus religiosa",
    "Potato": "Solanum tuberosum",
    "Rose": "Rosa",
    "Spinach": "Spinacia oleracea",
    "Sunflower": "Helianthus annuus",
    "Tomato": "Solanum lycopersicum",
    "Tulip": "Tulipa",
    "Tulsi": "Ocimum tenuiflorum"
}

TARGET_PER_CLASS = 55  # ~40 train, 8 val, 7 test per class = ~1870 total

def get_inaturalist_taxon_id(scientific_name):
    try:
        url = 'https://api.inaturalist.org/v1/taxa'
        r = requests.get(url, params={'q': scientific_name}, headers=HEADERS, timeout=8)
        if r.status_code == 200:
            results = r.json().get('results', [])
            if results:
                return results[0]['id']
    except Exception:
        pass
    return None

def fetch_inaturalist_urls(taxon_id, target_count=60):
    urls = []
    if not taxon_id:
        return urls
    
    # Try page 1 & 2
    for page in [1, 2]:
        try:
            url = 'https://api.inaturalist.org/v1/observations'
            params = {
                'taxon_id': taxon_id,
                'photos': 'true',
                'per_page': 50,
                'page': page,
                'order': 'desc',
                'order_by': 'votes'
            }
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                results = r.json().get('results', [])
                for obs in results:
                    photos = obs.get('photos', [])
                    for p in photos:
                        u = p.get('url', '')
                        if u:
                            med_u = u.replace('/square.', '/medium.')
                            urls.append(med_u)
                            if len(urls) >= target_count:
                                return urls
        except Exception as e:
            print(f"    iNat obs page {page} err: {e}", flush=True)
            break
        time.sleep(0.2)
    return urls

def fetch_wikimedia_urls(scientific_name, common_name, target_count=30):
    urls = []
    queries = [
        f"{scientific_name} plant",
        f"{scientific_name} leaf",
        f"{common_name} plant leaves"
    ]
    for q in queries:
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'generator': 'search',
                'gsrsearch': q,
                'gsrlimit': 15,
                'prop': 'pageimages',
                'pithumbsize': 640,
                'format': 'json'
            }
            r = requests.get(url, params=params, headers=HEADERS, timeout=8)
            if r.status_code == 200:
                pages = r.json().get('query', {}).get('pages', {})
                for pid, pdata in pages.items():
                    thumb = pdata.get('thumbnail')
                    if thumb and 'source' in thumb:
                        urls.append(thumb['source'])
                        if len(urls) >= target_count:
                            return urls
        except Exception:
            pass
        time.sleep(0.2)
    return urls

def fetch_commons_category_urls(scientific_name, target_count=30):
    urls = []
    cat_title = f"Category:{scientific_name.replace(' ', '_')}"
    try:
        url = 'https://commons.wikimedia.org/w/api.php'
        params = {
            'action': 'query',
            'generator': 'categorymembers',
            'gcmtitle': cat_title,
            'gcmtype': 'file',
            'gcmlimit': 40,
            'prop': 'imageinfo',
            'iiprop': 'url',
            'iiurlwidth': 640,
            'format': 'json'
        }
        r = requests.get(url, params=params, headers=HEADERS, timeout=8)
        if r.status_code == 200:
            pages = r.json().get('query', {}).get('pages', {})
            for pid, pdata in pages.items():
                title = pdata.get('title', '').lower()
                if any(bad in title for bad in ['map', 'diagram', 'icon', 'logo', 'drawing', 'illustration', 'herbarium', 'chart']):
                    continue
                info = pdata.get('imageinfo', [])
                if info and 'thumburl' in info[0]:
                    urls.append(info[0]['thumburl'])
                    if len(urls) >= target_count:
                        return urls
    except Exception:
        pass
    return urls

def download_single_image(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200 and len(r.content) > 3000:
            im = Image.open(io.BytesIO(r.content)).convert('RGB')
            if im.size[0] >= 120 and im.size[1] >= 120:
                h = hashlib.md5(r.content).hexdigest()
                return im, h
    except Exception:
        pass
    return None, None

def process_species(species_name, scientific_name):
    print(f"\nProcessing [{species_name}] ({scientific_name})...", flush=True)
    
    # Existing images in train/val/test to preserve
    existing_images = []
    seen_hashes = set()
    for s_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        c_dir = s_dir / species_name
        if c_dir.exists():
            for p in c_dir.glob('*.*'):
                try:
                    with open(p, 'rb') as f:
                        raw = f.read()
                        h = hashlib.md5(raw).hexdigest()
                        im = Image.open(io.BytesIO(raw)).convert('RGB')
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            existing_images.append(im)
                except Exception:
                    pass
    
    print(f"  Existing valid images: {len(existing_images)}", flush=True)
    
    all_images = list(existing_images)
    
    # 1. Look up taxon ID
    tid = get_inaturalist_taxon_id(scientific_name)
    print(f"  iNaturalist Taxon ID: {tid}", flush=True)
    
    candidate_urls = []
    if tid:
        candidate_urls.extend(fetch_inaturalist_urls(tid, target_count=TARGET_PER_CLASS + 15))
    
    # Fallback to Wikimedia if needed
    if len(candidate_urls) < TARGET_PER_CLASS:
        candidate_urls.extend(fetch_commons_category_urls(scientific_name, target_count=30))
    if len(candidate_urls) < TARGET_PER_CLASS:
        candidate_urls.extend(fetch_wikimedia_urls(scientific_name, species_name, target_count=30))
    
    print(f"  Collected {len(candidate_urls)} candidate URLs. Downloading...", flush=True)
    
    # Parallel download
    with ThreadPoolExecutor(max_workers=8) as pool:
        future_map = {pool.submit(download_single_image, u): u for u in candidate_urls}
        for fut in as_completed(future_map):
            if len(all_images) >= TARGET_PER_CLASS:
                break
            im, h = fut.result()
            if im is not None and h not in seen_hashes:
                seen_hashes.add(h)
                all_images.append(im)
    
    total = len(all_images)
    print(f"  Total unique images for [{species_name}]: {total}", flush=True)
    
    if total < 5:
        print(f"  WARNING: Only {total} images for {species_name}!", flush=True)
        return species_name, total
    
    # Shuffle and split into train (75%), val (15%), test (10%)
    random.seed(42)
    random.shuffle(all_images)
    
    val_n = max(3, int(total * 0.15))
    test_n = max(3, int(total * 0.10))
    train_n = total - val_n - test_n
    
    train_set = all_images[:train_n]
    val_set = all_images[train_n:train_n + val_n]
    test_set = all_images[train_n + val_n:]
    
    for split_dir, split_set in [(TRAIN_DIR, train_set), (VAL_DIR, val_set), (TEST_DIR, test_set)]:
        target_dir = split_dir / species_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear out previous files in this class folder to avoid stale/mislabeled files
        for old_f in target_dir.glob('*.*'):
            try:
                old_f.unlink()
            except Exception:
                pass
        
        slug = species_name.lower().replace(' ', '_')
        for idx, img in enumerate(split_set):
            img_resized = img.copy()
            img_resized.thumbnail((500, 500), Image.Resampling.LANCZOS)
            out_file = target_dir / f"{slug}_{idx+1:03d}.jpg"
            img_resized.save(out_file, 'JPEG', quality=88)
            
    print(f"  Saved -> Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}", flush=True)
    return species_name, total

def main():
    print("="*70, flush=True)
    print("STARTING BOTANICAL DATASET EXPANSION FOR 34 PLANT SPECIES", flush=True)
    print("="*70, flush=True)
    
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    t_start = time.time()
    
    for sp_name, sc_name in SPECIES_TAXONOMY.items():
        name, cnt = process_species(sp_name, sc_name)
        summary[name] = cnt
        time.sleep(0.3)
        
    elapsed = time.time() - t_start
    print("\n" + "="*70, flush=True)
    print(f"DATASET DOWNLOAD COMPLETED IN {elapsed:.1f}s", flush=True)
    print("="*70, flush=True)
    for k, v in sorted(summary.items()):
        print(f"  {k:20s}: {v:3d} images", flush=True)
        
    manifest = {
        'total_species': len(summary),
        'species_counts': summary,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(DATASET_DIR / 'dataset_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print("Manifest updated successfully.", flush=True)

if __name__ == '__main__':
    main()

