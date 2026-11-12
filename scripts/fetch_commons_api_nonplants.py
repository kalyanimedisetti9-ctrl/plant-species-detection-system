import io
import requests
from PIL import Image
from pathlib import Path

DEST_DIR = Path('media/test_non_plant_samples')
DEST_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {'User-Agent': 'PlantSpeciesBot/1.0 (academic; medis@example.com)'}

QUERIES = {
    'hand': 'File:Hand.jpg',
    'person': 'File:Woman_1.jpg',
    'dog': 'File:Golden_Retriever_Hund_-_Profile.jpg',
    'phone': 'File:Smartphone.jpg',
    'vehicle': 'File:Red_Car.jpg',
    'building': 'File:Eiffel_Tower_from_the_Tour_Montparnasse_3,_Paris_May_2014.jpg'
}

api_url = 'https://commons.wikimedia.org/w/api.php'

for name, title in QUERIES.items():
    dest_file = DEST_DIR / f"{name}.jpg"
    try:
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'imageinfo',
            'iiprop': 'url',
            'iiurlwidth': 640,
            'format': 'json'
        }
        res = requests.get(api_url, params=params, headers=HEADERS, timeout=15).json()
        pages = res.get('query', {}).get('pages', {})
        page = list(pages.values())[0]
        imageinfo = page.get('imageinfo', [{}])[0]
        thumb_url = imageinfo.get('thumburl') or imageinfo.get('url')
        if thumb_url:
            img_resp = requests.get(thumb_url, headers=HEADERS, timeout=15)
            if img_resp.status_code == 200:
                img = Image.open(io.BytesIO(img_resp.content)).convert('RGB')
                img.save(dest_file, 'JPEG')
                print(f"Downloaded {name} ({dest_file}): {img.size}")
                continue
        print(f"Could not get url for {name}")
    except Exception as e:
        print(f"Error {name}: {e}")

