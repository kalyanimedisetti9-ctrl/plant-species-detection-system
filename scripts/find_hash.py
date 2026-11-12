import hashlib
from pathlib import Path

target_hash = hashlib.md5(open('media/uploaded_images/brinjal_nZ64tNt.jpeg','rb').read()).hexdigest()
print("Uploaded Brinjal MD5:", target_hash)

found = []
for split in ['train', 'validation', 'test']:
    base = Path('ml_model/dataset') / split
    for f in base.rglob('*.*'):
        try:
            h = hashlib.md5(open(f, 'rb').read()).hexdigest()
            if h == target_hash:
                found.append(str(f))
        except:
            pass

print("Matching files in dataset:", found)

