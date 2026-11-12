import io, requests
from PIL import Image
from pathlib import Path

DEST_DIR = Path('media/test_non_plant_samples')
DEST_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {'User-Agent': 'PlantTest/1.0 (student; testing)'}

for query, name in [('dog sitting photo', 'dog'), ('sports car road photo', 'vehicle')]:
    try:
        sr = requests.get(
            f'https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={query}&srnamespace=6&format=json',
            headers=HEADERS, timeout=15
        ).json()
        title = sr['query']['search'][0]['title']
        r = requests.get(
            f'https://commons.wikimedia.org/w/api.php?action=query&titles={title}&prop=imageinfo&iiprop=url&iiurlwidth=640&format=json',
            headers=HEADERS, timeout=15
        ).json()
        pages = r['query']['pages']
        page = list(pages.values())[0]
        url = page['imageinfo'][0]['thumburl']
        img_bytes = requests.get(url, headers=HEADERS, timeout=15).content
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        dest = DEST_DIR / f"{name}.jpg"
        img.save(dest, 'JPEG')
        print(f"Successfully saved {name} ({title}) to {dest}: {img.size}")
    except Exception as e:
        print(f"Failed {name}: {e}")

