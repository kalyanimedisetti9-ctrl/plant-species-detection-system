import io
import requests
from PIL import Image
from pathlib import Path

DEST_DIR = Path('media/test_non_plant_samples')
DEST_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {'User-Agent': 'BotanyDatasetValidator/2.0 (student-research; botanical-project)'}

TARGETS = {
    'hand': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Hand_model_showing_open_palm.jpg/640px-Hand_model_showing_open_palm.jpg',
    'person': 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Young_woman_smiling_portrait.jpg/640px-Young_woman_smiling_portrait.jpg',
    'dog': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/640px-YellowLabradorLooking_new.jpg',
    'phone': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Smartphone_display_front.jpg/640px-Smartphone_display_front.jpg',
    'vehicle': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/2018_Tesla_Model_3_front_left.jpg/640px-2018_Tesla_Model_3_front_left.jpg',
    'building': 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/Empire_State_Building_from_the_Top_of_the_Rock.jpg/640px-Empire_State_Building_from_the_Top_of_the_Rock.jpg'
}

for name, url in TARGETS.items():
    dest_file = DEST_DIR / f"{name}.jpg"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            img = Image.open(io.BytesIO(r.content)).convert('RGB')
            img.save(dest_file, 'JPEG')
            print(f"Downloaded {name}: {img.size}")
        else:
            print(f"Failed {name}: status {r.status_code}")
    except Exception as e:
        print(f"Error {name}: {e}")

