"""
Database Seeding and Model Initialization Script
Populates complete plant species catalog (25 botanical species), generates realistic imagery,
creates admin/demo accounts, and initializes the deep learning model.
"""

import os
import sys
import django
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

# Set up Django environment
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'plant_detection.settings')
django.setup()

from django.contrib.auth.models import User
from detection.models import PlantSpecies
from ml_model.prediction import load_or_create_model, DEFAULT_SPECIES_TAXONOMY

def draw_lush_leaf(draw, center_x, center_y, size, angle_deg, fill_color, vein_color):
    """
    Draws a realistic stylized botanical leaf with central and lateral veins.
    """
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # Leaf contour points
    tip_x = center_x + size * cos_a
    tip_y = center_y + size * sin_a
    base_x = center_x - (size * 0.4) * cos_a
    base_y = center_y - (size * 0.4) * sin_a

    perp_cos = -sin_a
    perp_sin = cos_a

    ctrl_r_x = center_x + (size * 0.2) * cos_a + (size * 0.4) * perp_cos
    ctrl_r_y = center_y + (size * 0.2) * sin_a + (size * 0.4) * perp_sin

    ctrl_l_x = center_x + (size * 0.2) * cos_a - (size * 0.4) * perp_cos
    ctrl_l_y = center_y + (size * 0.2) * sin_a - (size * 0.4) * perp_sin

    polygon = [
        (base_x, base_y),
        (ctrl_r_x, ctrl_r_y),
        (tip_x, tip_y),
        (ctrl_l_x, ctrl_l_y)
    ]
    draw.polygon(polygon, fill=fill_color)

    # Main vein
    draw.line([(base_x, base_y), (tip_x, tip_y)], fill=vein_color, width=3)
    # Side veins
    for t in [0.25, 0.5, 0.75]:
        vx = base_x + (tip_x - base_x) * t
        vy = base_y + (tip_y - base_y) * t
        side_rx = vx + (size * 0.18) * perp_cos + (size * 0.1) * cos_a
        side_ry = vy + (size * 0.18) * perp_sin + (size * 0.1) * sin_a
        side_lx = vx - (size * 0.18) * perp_cos + (size * 0.1) * cos_a
        side_ly = vy - (size * 0.18) * perp_sin + (size * 0.1) * sin_a
        draw.line([(vx, vy), (side_rx, side_ry)], fill=vein_color, width=2)
        draw.line([(vx, vy), (side_lx, side_ly)], fill=vein_color, width=2)


def create_realistic_plant_image(common_name, scientific_name, family, primary_color, secondary_color, output_path):
    """
    Creates a high-resolution, realistic botanical illustration with subtle gradients and veins.
    """
    width, height = 700, 520
    img = Image.new('RGB', (width, height), color=(15, 45, 25))
    draw = ImageDraw.Draw(img)

    # Background gradient effect
    for y in range(height):
        ratio = y / height
        r = int(12 + ratio * (primary_color[0] * 0.35))
        g = int(35 + ratio * (primary_color[1] * 0.45))
        b = int(20 + ratio * (primary_color[2] * 0.35))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Glowing botanical backdrop circles
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    ol_draw = ImageDraw.Draw(overlay)
    ol_draw.ellipse([350, -50, 750, 350], fill=(*secondary_color, 45))
    ol_draw.ellipse([-80, 200, 320, 600], fill=(*secondary_color, 35))
    img.paste(Image.alpha_composite(Image.new('RGBA', (width, height), (*img.getpixel((0,0)), 255)), overlay).convert('RGB'))
    draw = ImageDraw.Draw(img)

    # Draw natural botanical foliage
    leaf_c1 = (primary_color[0], min(255, primary_color[1] + 30), primary_color[2])
    leaf_c2 = secondary_color
    vein_c = (200, 245, 210)

    # Cluster of leaves
    draw_lush_leaf(draw, 450, 240, 190, -45, leaf_c1, vein_c)
    draw_lush_leaf(draw, 520, 200, 160, -20, leaf_c2, vein_c)
    draw_lush_leaf(draw, 400, 310, 170, -75, leaf_c2, vein_c)
    draw_lush_leaf(draw, 490, 330, 140, -110, leaf_c1, vein_c)

    # Flower / Fruit focal accent if applicable
    accent_r = 45
    cx, cy = 480, 250
    draw.ellipse([cx - accent_r, cy - accent_r, cx + accent_r, cy + accent_r], fill=secondary_color, outline=(255, 255, 255), width=2)
    draw.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], fill=(255, 240, 100))

    # Dark gradient overlay on bottom for high contrast text
    for y in range(320, height):
        alpha = int(((y - 320) / (height - 320)) * 220)
        draw.line([(0, y), (width, y)], fill=(10, 28, 16))

    # Botanical typography
    draw.rectangle([30, 345, 140, 370], fill=(34, 197, 94))
    draw.text((40, 350), family.upper(), fill=(255, 255, 255))

    draw.text((32, 380), common_name, fill=(255, 255, 255))
    draw.text((32, 435), f"Scientific: {scientific_name}", fill=(167, 243, 208))
    draw.text((32, 475), "Plant Species Detection System • Botanical Dataset", fill=(110, 231, 183))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "JPEG", quality=92)
    return output_path


def generate_hero_and_about_assets():
    """
    Generates large, high-res realistic plant visual assets for the Home Hero and About pages.
    """
    media_dir = BASE_DIR / 'media'
    media_dir.mkdir(parents=True, exist_ok=True)

    # 1. Hero Plant Image
    hero_img = Image.new('RGB', (800, 600), color=(15, 45, 25))
    draw = ImageDraw.Draw(hero_img)
    for y in range(600):
        ratio = y / 600
        draw.line([(0, y), (800, y)], fill=(int(10 + ratio * 20), int(38 + ratio * 45), int(20 + ratio * 30)))
    # Foliage artwork
    draw_lush_leaf(draw, 420, 280, 280, -35, (34, 139, 34), (200, 255, 210))
    draw_lush_leaf(draw, 520, 220, 240, -10, (46, 175, 80), (220, 255, 225))
    draw_lush_leaf(draw, 350, 360, 250, -65, (22, 101, 52), (180, 240, 195))
    draw_lush_leaf(draw, 480, 420, 210, -105, (74, 222, 128), (240, 255, 240))
    # Glowing focus flower
    draw.ellipse([430, 230, 530, 330], fill=(225, 29, 72), outline=(255, 255, 255), width=3)
    draw.ellipse([460, 260, 500, 300], fill=(253, 224, 71))
    hero_path = media_dir / 'hero_plant.jpg'
    hero_img.save(str(hero_path), "JPEG", quality=95)

    # 2. About Plant Image
    about_img = Image.new('RGB', (800, 600), color=(18, 52, 30))
    draw_a = ImageDraw.Draw(about_img)
    for y in range(600):
        ratio = y / 600
        draw_a.line([(0, y), (800, y)], fill=(int(12 + ratio * 25), int(42 + ratio * 50), int(24 + ratio * 35)))
    draw_lush_leaf(draw_a, 400, 300, 270, -45, (16, 185, 129), (220, 255, 235))
    draw_lush_leaf(draw_a, 490, 240, 230, -15, (5, 150, 105), (200, 255, 220))
    draw_lush_leaf(draw_a, 320, 380, 240, -80, (4, 120, 87), (180, 245, 205))
    draw_a.ellipse([410, 210, 490, 290], fill=(245, 158, 11), outline=(255, 255, 255), width=3)
    draw_a.ellipse([435, 235, 465, 265], fill=(254, 240, 138))
    about_path = media_dir / 'about_plant.jpg'
    about_img.save(str(about_path), "JPEG", quality=95)

    print("Hero and About visual artwork created successfully.")


def seed_data():
    print("Applying migrations...")
    from django.core.management import call_command
    call_command('makemigrations', 'detection', interactive=False)
    call_command('migrate', interactive=False)

    print("\n--- Generating Hero & About Visual Assets ---")
    generate_hero_and_about_assets()

    print("\n--- Initializing Deep Learning Model Architecture ---")
    load_or_create_model()
    print("TensorFlow / Keras model initialized successfully.")

    print("\n--- Creating Default Superuser & Demo User ---")
    if not User.objects.filter(username='admin').exists():
        User.objects.create_superuser('admin', 'plantdetect@gmail.com', 'admin123')
        print("Superuser created: username='admin', password='admin123'")
    else:
        print("Superuser 'admin' already active.")

    if not User.objects.filter(username='student').exists():
        User.objects.create_user('student', 'student@plantspecies.local', 'student123')
        print("Demo User created: username='student', password='student123'")

    print("\n--- Seeding Extended Plant Species Catalog (25 Botanical Species) ---")
    
    species_color_map = {
        "Mango": ((34, 139, 34), (234, 179, 8)),
        "Neem": ((20, 95, 45), (74, 222, 128)),
        "Tulsi": ((22, 101, 52), (168, 85, 247)),
        "Guava": ((46, 125, 50), (248, 113, 113)),
        "Aloe Vera": ((30, 115, 80), (52, 211, 153)),
        "Peepal": ((21, 128, 61), (134, 239, 172)),
        "Banana": ((22, 163, 74), (250, 204, 21)),
        "Tomato": ((185, 28, 28), (239, 68, 68)),
        "Rose": ((190, 18, 60), (244, 63, 94)),
        "Hibiscus": ((225, 29, 72), (251, 113, 133)),
        "Sunflower": ((202, 138, 4), (250, 204, 21)),
        "Lotus": ((219, 39, 119), (244, 114, 182)),
        "Jasmine": ((13, 148, 136), (240, 253, 250)),
        "Coconut": ((67, 56, 202), (129, 140, 248)),
        "Papaya": ((217, 119, 6), (251, 146, 60)),
        "Lemon": ((234, 179, 8), (254, 240, 138)),
        "Jackfruit": ((101, 163, 13), (163, 230, 53)),
        "Curry Leaf": ((21, 128, 61), (74, 222, 128)),
        "Ashoka": ((234, 88, 12), (251, 146, 60)),
        "Banyan": ((15, 118, 110), (94, 234, 212)),
        "Money Plant": ((22, 101, 52), (250, 204, 21)),
        "Tulip": ((190, 24, 93), (244, 114, 182)),
        "Daisy": ((13, 148, 136), (254, 240, 138)),
        "Dandelion": ((202, 138, 4), (253, 224, 71)),
        "Lavender": ((109, 40, 217), (196, 181, 253)),
    }

    media_species_dir = BASE_DIR / 'media' / 'species_images'
    media_species_dir.mkdir(parents=True, exist_ok=True)
    test_samples_dir = BASE_DIR / 'media' / 'test_samples'
    test_samples_dir.mkdir(parents=True, exist_ok=True)

    for common_name, details in DEFAULT_SPECIES_TAXONOMY.items():
        img_filename = f"{common_name.lower().replace(' ', '_')}.jpg"
        img_path = media_species_dir / img_filename
        test_path = test_samples_dir / img_filename

        colors = species_color_map.get(common_name, ((34, 139, 34), (74, 222, 128)))
        primary_c, secondary_c = colors

        # Generate realistic graphic image
        create_realistic_plant_image(
            common_name,
            details['scientific_name'],
            details['family'],
            primary_c,
            secondary_c,
            str(img_path)
        )
        
        create_realistic_plant_image(
            common_name,
            details['scientific_name'],
            details['family'],
            primary_c,
            secondary_c,
            str(test_path)
        )

        species_obj, created = PlantSpecies.objects.get_or_create(
            common_name=common_name,
            defaults={
                'scientific_name': details['scientific_name'],
                'family': details['family'],
                'description': details['description'],
                'care_instructions': details.get('care_instructions', ''),
                'medicinal_uses': details.get('medicinal_uses', ''),
                'image': f"species_images/{img_filename}"
            }
        )
        if not created:
            species_obj.scientific_name = details['scientific_name']
            species_obj.family = details['family']
            species_obj.description = details['description']
            species_obj.care_instructions = details.get('care_instructions', '')
            species_obj.medicinal_uses = details.get('medicinal_uses', '')
            species_obj.image = f"species_images/{img_filename}"
            species_obj.save()

        status = "Created" if created else "Updated"
        print(f"[{status}] {common_name} ({details['scientific_name']}) - {details['family']}")

    print("\nDatabase seeded with all 25 plant species successfully!")

if __name__ == '__main__':
    seed_data()
