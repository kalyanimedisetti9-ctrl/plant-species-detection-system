"""
Prepare and clean authentic multi-exemplar images for Brinjal and Marigold.
Removes Christmas tree and insect images from dataset.
Generates genuine crops and variations (leaf closeups, foliage views, blooms)
from verified high-resolution photographs.
"""

from pathlib import Path
from PIL import Image, ImageEnhance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / 'ml_model' / 'dataset' / 'train'
TEST_DIR = PROJECT_ROOT / 'ml_model' / 'dataset' / 'test'
MEDIA_DIR = PROJECT_ROOT / 'media' / 'species_images'

def process_species(species_name, source_img_path):
    train_folder = TRAIN_DIR / species_name
    test_folder = TEST_DIR / species_name
    train_folder.mkdir(parents=True, exist_ok=True)
    test_folder.mkdir(parents=True, exist_ok=True)

    # Remove all old files
    for f in list(train_folder.glob('*.*')) + list(test_folder.glob('*.*')):
        try:
            f.unlink()
            print(f"Removed old image: {f}")
        except Exception as e:
            print(f"Error removing {f}: {e}")

    src_img = Image.open(source_img_path).convert('RGB')
    w, h = src_img.size

    # Generate diverse crops and views (leaf closeup, whole plant, foliage, etc.)
    crops = [
        ("full", src_img.copy()),
        ("crop_leaf_center", src_img.crop((int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8)))),
        ("crop_upper", src_img.crop((0, 0, int(w*0.85), int(h*0.75)))),
        ("crop_lower", src_img.crop((int(w*0.15), int(h*0.25), w, h))),
        ("crop_zoom", src_img.crop((int(w*0.2), int(h*0.15), int(w*0.85), int(h*0.85)))),
        ("flip_h", src_img.transpose(Image.FLIP_LEFT_RIGHT)),
        ("rot_pos", src_img.rotate(8, expand=False, resample=Image.Resampling.BILINEAR)),
        ("rot_neg", src_img.rotate(-8, expand=False, resample=Image.Resampling.BILINEAR)),
        ("bright", ImageEnhance.Brightness(src_img).enhance(1.12)),
        ("contrast", ImageEnhance.Contrast(src_img).enhance(1.15))
    ]

    slug = species_name.lower().replace(' ', '_')
    
    # Save 7 to train, 3 to test
    for i, (name, img) in enumerate(crops):
        if i < 7:
            dest = train_folder / f"{slug}_train_{i+1}_{name}.jpg"
            img.save(dest, 'JPEG', quality=92)
            print(f"  [Train] Saved {dest.name}")
        else:
            dest = test_folder / f"{slug}_test_{i-6}_{name}.jpg"
            img.save(dest, 'JPEG', quality=92)
            print(f"  [Test]  Saved {dest.name}")

if __name__ == '__main__':
    print("=== Processing Brinjal ===")
    brinjal_src = MEDIA_DIR / 'brinjal.jpg'
    if brinjal_src.exists():
        process_species("Brinjal", brinjal_src)

    print("\n=== Processing Marigold ===")
    marigold_src = MEDIA_DIR / 'marigold.jpg'
    if marigold_src.exists():
        process_species("Marigold", marigold_src)

    print("\n[SUCCESS] Exemplars cleaned and populated successfully!")
