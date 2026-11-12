"""
Fetch genuine, verified botanical photographs directly from Wikimedia Commons Categories:
Category:Taraxacum_officinale (Dandelion)
Category:Epipremnum_aureum (Money Plant)
Category:Solanum_melongena (Brinjal)
Category:Capsicum_annuum (Chilli)
Category:Spinacia_oleracea (Spinach)
Category:Solanum_tuberosum (Potato)
Category:Mentha_spicata (Mint)
Category:Tagetes_erecta (Marigold)
Category:Coriandrum_sativum (Coriander)
Category:Gossypium_hirsutum (Cotton)
Category:Trigonella_foenum-graecum (Fenugreek)
Category:Lavandula_angustifolia (Lavender)
"""

import io
import time
import requests
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / 'ml_model' / 'dataset' / 'train'
TEST_DIR = PROJECT_ROOT / 'ml_model' / 'dataset' / 'test'
SPECIES_DIR = PROJECT_ROOT / 'media' / 'species_images'
TEST_SAMPLES_DIR = PROJECT_ROOT / 'media' / 'test_samples'

HEADERS = {'User-Agent': 'BotanyDatasetValidator/2.0 (student-research; botanical-project)'}

TARGET_CATEGORIES = {
    "Dandelion": "Category:Taraxacum_officinale",
    "Money Plant": "Category:Epipremnum_aureum",
    "Brinjal": "Category:Solanum_melongena",
    "Chilli": "Category:Capsicum_annuum",
    "Spinach": "Category:Spinacia_oleracea",
    "Potato": "Category:Solanum_tuberosum",
    "Mint": "Category:Mentha_spicata",
    "Marigold": "Category:Tagetes_erecta",
    "Coriander": "Category:Coriandrum_sativum",
    "Cotton": "Category:Gossypium_hirsutum",
    "Fenugreek": "Category:Trigonella_foenum-graecum",
    "Lavender": "Category:Lavandula_angustifolia",
}

def fetch_category_images(category, max_images=12):
    url = 'https://commons.wikimedia.org/w/api.php'
    params = {
        'action': 'query',
        'generator': 'categorymembers',
        'gcmtitle': category,
        'gcmtype': 'file',
        'gcmlimit': max_images * 2,
        'prop': 'imageinfo',
        'iiprop': 'url',
        'iiurlwidth': 800,
        'format': 'json'
    }
    images = []
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=12)
        if r.status_code == 200:
            pages = r.json().get('query', {}).get('pages', {})
            for pid, pdata in pages.items():
                title = pdata.get('title', '').lower()
                # Exclude maps, diagrams, book pages, distribution maps
                if any(bad in title for bad in ['map', 'diagram', 'icon', 'logo', 'drawing', 'illustration', 'herbarium', 'page']):
                    continue
                info = pdata.get('imageinfo', [])
                if info and 'thumburl' in info[0]:
                    thumb_url = info[0]['thumburl']
                    try:
                        res = requests.get(thumb_url, headers=HEADERS, timeout=12)
                        if res.status_code == 200:
                            img = Image.open(io.BytesIO(res.content)).convert('RGB')
                            if img.size[0] >= 180 and img.size[1] >= 180:
                                images.append(img)
                                if len(images) >= max_images:
                                    break
                    except Exception:
                        continue
    except Exception as e:
        print(f"Error fetching {category}: {e}", flush=True)
    return images

def main():
    print("=== Fetching Verified Botanical Category Photos ===", flush=True)
    for cls_name, category in TARGET_CATEGORIES.items():
        slug = cls_name.lower().replace(' ', '_')
        train_folder = TRAIN_DIR / cls_name
        test_folder = TEST_DIR / cls_name
        train_folder.mkdir(parents=True, exist_ok=True)
        test_folder.mkdir(parents=True, exist_ok=True)

        print(f"Fetching from {category} for {cls_name}...", flush=True)
        imgs = fetch_category_images(category, max_images=12)
        if not imgs:
            print(f"  Warning: No category images found for {category}", flush=True)
            continue

        print(f"  Downloaded {len(imgs)} genuine photos for {cls_name}", flush=True)

        # Clear existing unverified images in this class folder
        for old_f in list(train_folder.glob('*.*')) + list(test_folder.glob('*.*')):
            try: old_f.unlink()
            except Exception: pass

        # Save cover photo
        card_img = imgs[0].copy()
        card_img.thumbnail((800, 600), Image.Resampling.LANCZOS)
        card_path = SPECIES_DIR / f"{slug}.jpg"
        test_sample_path = TEST_SAMPLES_DIR / f"{slug}.jpg"
        card_img.save(str(card_path), "JPEG", quality=92)
        card_img.save(str(test_sample_path), "JPEG", quality=92)

        # Distribute: last 3 are test, others are train
        for i, im in enumerate(imgs):
            if i >= len(imgs) - 3:
                im.save(str(test_folder / f"{slug}_test_{i+1}.jpg"), "JPEG", quality=90)
            else:
                im.save(str(train_folder / f"{slug}_{i+1}.jpg"), "JPEG", quality=90)

    print("\n=== Dataset Cleaned & Re-populated with Authentic Photos! ===", flush=True)

if __name__ == '__main__':
    main()
