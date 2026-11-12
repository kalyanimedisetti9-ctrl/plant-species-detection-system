import io
import requests
from PIL import Image
from pathlib import Path

DEST_DIR = Path('media/test_non_plant_samples')
DEST_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {'User-Agent': 'PlantSpeciesBot/1.0 (academic; medis@example.com)'}

QUERIES = {
    'hand': 'human hand palm',
    'dog': 'dog domestic animal',
    'vehicle': 'automobile car street'
}

api_url = 'https://commons.wikimedia.org/w/api.php'

for name, search_term in QUERIES.items():
    dest_file = DEST_DIR / f"{name}.jpg"
    try:
        params = {
            'action': 'query',
            'generator': 'search',
            'gsrsearch': search_term,
            'gsrnamespace': 6,
            'gsrlimit': 5,
            'prop': 'imageinfo',
            'iiprop': 'url',
            'iiurlwidth': 640,
            'format': 'json'
        }
        res = requests.get(api_url, params=params, headers=HEADERS, timeout=15).json()
        pages = res.get('query', {}).get('pages', {})
        for pid, page in pages.items():
            imageinfo = page.get('imageinfo', [{}])[0]
            thumb_url = imageinfo.get('thumburl') or imageinfo.get('url')
            if thumb_url and (thumb_url.endswith('.jpg') or thumb_url.endswith('.jpeg') or thumb_url.endswith('.png')):
                img_resp = requests.get(thumb_url, headers=HEADERS, timeout=15)
                if img_resp.status_code == 200:
                    img = Image.open(io.BytesIO(img_resp.content)).convert('RGB')
                    img.save(dest_file, 'JPEG')
                    print(f"Downloaded {name} ({dest_file}): {img.size}")
                    break
    except Exception as e:
        print(f"Error {name}: {e}")

