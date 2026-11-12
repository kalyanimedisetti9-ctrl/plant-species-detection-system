"""
Update Django PlantSpecies database with complete, accurate taxonomy for all 36 botanical species.
Ensures every card has high-resolution photograph, scientific name, family, and description.
"""

import os
import sys
import django
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'plant_detection.settings')
django.setup()

from detection.models import PlantSpecies

ALL_SPECIES_DATA = {
    "Aloe Vera": {
        "scientific_name": "Aloe vera",
        "family": "Asphodelaceae",
        "description": "Aloe Vera is a stemless succulent perennial plant known for its thick, fleshy, lance-shaped leaves with serrated margins containing therapeutic gel.",
        "care_instructions": "Requires bright direct to indirect sunlight, gritty well-draining cactus soil, and infrequent deep watering.",
        "medicinal_uses": "Soothes thermal burns, accelerates wound re-epithelialization, moisturizes skin, and supports digestive comfort.",
        "image": "species_images/aloe_vera.jpg"
    },
    "Ashoka": {
        "scientific_name": "Saraca asoca",
        "family": "Fabaceae",
        "description": "A sacred evergreen rainforest tree valued for its dense foliage, fragrant orange-yellow floral clusters, and cultural significance.",
        "care_instructions": "Prefers warm humid climates, rich loamy soil, and partial shade to moderate sunlight.",
        "medicinal_uses": "Bark tannins and flavonoids are extensively utilized in traditional remedies for uterine and circulatory balance.",
        "image": "species_images/ashoka.jpg"
    },
    "Banana": {
        "scientific_name": "Musa acuminata",
        "family": "Musaceae",
        "description": "A giant herbaceous perennial with spirally arranged overlapping leaf sheaths forming a robust pseudostem bearing hanging fruit bunches.",
        "care_instructions": "Requires rich humus soil, abundant moisture, shelter from high winds, and full sun exposure.",
        "medicinal_uses": "Rich source of potassium and dietary fiber; pseudostem juice and flowers support glycemic and urinary health.",
        "image": "species_images/banana.jpg"
    },
    "Banyan": {
        "scientific_name": "Ficus benghalensis",
        "family": "Moraceae",
        "description": "The monumental national tree of India, characterized by extensive aerial prop roots forming vast secondary pillar canopies.",
        "care_instructions": "Needs vast root space, full sun, and adaptable to diverse soils with moderate moisture.",
        "medicinal_uses": "Latex and prop roots are used in traditional medicine for oral hygiene, gum strengthening, and skin inflammation.",
        "image": "species_images/banyan.jpg"
    },
    "Basil": {
        "scientific_name": "Ocimum basilicum",
        "family": "Lamiaceae",
        "description": "A culinary herb in the mint family with aromatic, tender green oval leaves widely celebrated in global gastronomy.",
        "care_instructions": "Requires 6-8 hours of direct sunshine, warm temperatures, and well-drained fertile garden soil.",
        "medicinal_uses": "Rich in essential oils (eugenol, linalool) with antimicrobial, antioxidant, and digestive carminative properties.",
        "image": "species_images/basil.jpg"
    },
    "Brinjal": {
        "scientific_name": "Solanum melongena",
        "family": "Solanaceae",
        "description": "Also known as Eggplant or Aubergine, a warm-season perennial herb grown as an annual crop with large lobed leaves and glossy fruit.",
        "care_instructions": "Requires full sunlight, warm soil temperatures, rich organic compost, and steady moisture.",
        "medicinal_uses": "High in nasunin (potent anthocyanin antioxidant) in peel, dietary fiber, and polyphenols supporting cholesterol management.",
        "image": "species_images/brinjal.jpg"
    },
    "Chilli": {
        "scientific_name": "Capsicum annuum",
        "family": "Solanaceae",
        "description": "A branching shrub-like plant with dark green ovate leaves and small white flowers yielding pungent, capsaicin-rich pods.",
        "care_instructions": "Thrives in sunny, warm locations with well-aerated sandy loam and moderate consistent watering.",
        "medicinal_uses": "Capsaicin provides topical neuropathic pain relief, stimulates metabolism, and delivers rich concentrations of vitamins A and C.",
        "image": "species_images/chilli.jpg"
    },
    "Coconut": {
        "scientific_name": "Cocos nucifera",
        "family": "Arecaceae",
        "description": "The quintessential tropical palm reaching up to 30 meters with pinnate leaves and versatile fruits known as the Tree of Life.",
        "care_instructions": "Requires bright tropical sun, high humidity, sandy saline-tolerant soil, and regular watering.",
        "medicinal_uses": "Electrolyte-rich coconut water for rehydration; lauric acid in virgin oil provides antimicrobial benefits.",
        "image": "species_images/coconut.jpg"
    },
    "Coriander": {
        "scientific_name": "Coriandrum sativum",
        "family": "Apiaceae",
        "description": "An annual culinary and medicinal herb with delicately dissected feathery leaves and aromatic globular seed fruits.",
        "care_instructions": "Grows best in cool seasons with well-drained fertile loam and moderate morning sunlight.",
        "medicinal_uses": "Possesses linalool and flavonoid antioxidants aiding digestion, heavy metal detox, and blood sugar balance.",
        "image": "species_images/coriander.jpg"
    },
    "Cotton": {
        "scientific_name": "Gossypium hirsutum",
        "family": "Malvaceae",
        "description": "An economically essential shrub with palmately lobed leaves and creamy blossoms yielding protective boll capsules filled with seed fibers.",
        "care_instructions": "Demands high sunlight, long frost-free periods, and deep, fertile, moisture-retentive soils.",
        "medicinal_uses": "Cottonseed extracts contain polyphenols; historically root bark infusions were studied for reproductive phytochemistry.",
        "image": "species_images/cotton.jpg"
    },
    "Curry Leaf": {
        "scientific_name": "Murraya koenigii",
        "family": "Rutaceae",
        "description": "An aromatic small tree native to India, bearing fragrant pinnate leaves indispensable in South Asian cooking.",
        "care_instructions": "Requires warm tropical temperatures, rich slightly acidic soil, and full morning sun.",
        "medicinal_uses": "Contains carbazole alkaloids with hepatoprotective, hair nourishing, antidiabetic, and antioxidant qualities.",
        "image": "species_images/curry_leaf.jpg"
    },
    "Daisy": {
        "scientific_name": "Bellis perennis",
        "family": "Asteraceae",
        "description": "A classic European perennial herbaceous plant producing basal rosettes of spoon-shaped leaves and composite flowers with white ray florets.",
        "care_instructions": "Prefers cool, temperate conditions, moderate moisture, and sunny to partially shaded meadows.",
        "medicinal_uses": "Historically used in herbal compresses for wound healing, bruising relief, and respiratory soothe.",
        "image": "species_images/daisy.jpg"
    },
    "Dandelion": {
        "scientific_name": "Taraxacum officinale",
        "family": "Asteraceae",
        "description": "A resilient perennial herb featuring deeply lobed toothed basal leaves (dent-de-lion) and bright yellow floral heads that form wind-borne seed globes.",
        "care_instructions": "Extremely hardy, adapts to almost all soil types, and thrives in full sun to partial shade.",
        "medicinal_uses": "Potent herbal diuretic, liver and gallbladder tonic, inulin-rich prebiotic, and packed with vitamins A, C, and K.",
        "image": "species_images/dandelion.jpg"
    },
    "Fenugreek": {
        "scientific_name": "Trigonella foenum-graecum",
        "family": "Fabaceae",
        "description": "A semi-arid annual legume with trifoliate clover-like leaves (methi) and slender pods containing golden-brown aromatic seeds.",
        "care_instructions": "Prefers cool sunny weather, loamy well-drained soil, and moderate regular hydration.",
        "medicinal_uses": "High in soluble galactomannan fiber and 4-hydroxyisoleucine, enhancing insulin sensitivity and lactation support.",
        "image": "species_images/fenugreek.jpg"
    },
    "Guava": {
        "scientific_name": "Psidium guajava",
        "family": "Myrtaceae",
        "description": "An evergreen tropical tree noted for its smooth mottled copper bark, prominent pinnate leaf veins, and fragrant white blossoms.",
        "care_instructions": "Thrives in subtropical to tropical climates with full sun and well-draining loamy soil.",
        "medicinal_uses": "Leaves contain astringent flavonoids brewed into therapeutic teas for gastrointestinal health and glycemic regulation.",
        "image": "species_images/guava.jpg"
    },
    "Hibiscus": {
        "scientific_name": "Hibiscus rosa-sinensis",
        "family": "Malvaceae",
        "description": "A striking ornamental tropical shrub famous for its vibrant trumpet-shaped flowers featuring an elongated central staminal column.",
        "care_instructions": "Requires warm temperatures, 6+ hours of daily sun, and evenly moist, nutrient-rich soil.",
        "medicinal_uses": "Anthocyanin-rich petals support cardiovascular tone, blood pressure modulation, and scalp nourishment.",
        "image": "species_images/hibiscus.jpg"
    },
    "Jackfruit": {
        "scientific_name": "Artocarpus heterophyllus",
        "family": "Moraceae",
        "description": "The largest tree-borne fruit in nature, Jackfruit features dark green leathery leaves and massive knobby cauliflower-bearing fruits.",
        "care_instructions": "Requires deep well-drained loamy soil, tropical temperatures, and ample space for massive canopy development.",
        "medicinal_uses": "Rich in potassium, carotenoids, and low-glycemic dietary resistant starch aiding gut microbiota.",
        "image": "species_images/jackfruit.jpg"
    },
    "Jasmine": {
        "scientific_name": "Jasminum officinale",
        "family": "Oleaceae",
        "description": "A fragrant vining or scrambling shrub renowned globally for its intensely sweet-scented white star-like blossoms.",
        "care_instructions": "Enjoys full sun to partial shade, trellis climbing support, and consistently moist fertile loam.",
        "medicinal_uses": "Volatile oils (benzyl acetate, linalool) reduce stress, elevate mood, and provide relaxing herbal infusions.",
        "image": "species_images/jasmine.jpg"
    },
    "Lavender": {
        "scientific_name": "Lavandula angustifolia",
        "family": "Lamiaceae",
        "description": "A fragrant semi-shrub with narrow silvery-gray foliage and slender spikes of vibrant violet-purple blooms.",
        "care_instructions": "Demands full direct sun and dry, well-draining alkaline sandy soil. Avoid excessive moisture.",
        "medicinal_uses": "Renowned for calming linalool terpenes promoting deep restorative sleep and stress alleviation.",
        "image": "species_images/lavender.jpg"
    },
    "Lemon": {
        "scientific_name": "Citrus limon",
        "family": "Rutaceae",
        "description": "A compact citrus tree with aromatic serrated evergreen leaves, thorny twigs, and fragrant blossoms yielding acidic yellow fruit.",
        "care_instructions": "Requires 8 hours of direct sunshine, slightly acidic well-drained soil, and balanced citrus micro-nutrients.",
        "medicinal_uses": "Concentrated source of vitamin C (ascorbic acid) and bioflavonoids enhancing immune vigor and iron absorption.",
        "image": "species_images/lemon.jpg"
    },
    "Lotus": {
        "scientific_name": "Nelumbo nucifera",
        "family": "Nelumbonaceae",
        "description": "An iconic aquatic perennial with large peltate hydrophobic floating leaves and radiant multi-petaled blossoms rising above water.",
        "care_instructions": "Needs warm water (20°C–35°C), nutrient-rich pond clay, and at least 6 hours of full unshaded sun.",
        "medicinal_uses": "Rhizomes and seeds provide dietary fiber; floral polyphenols offer calming neuro-protective properties.",
        "image": "species_images/lotus.jpg"
    },
    "Mango": {
        "scientific_name": "Mangifera indica",
        "family": "Anacardiaceae",
        "description": "A grand tropical fruit tree with a dense spreading canopy of leathery oblong lanceolate leaves and prized sweet drupe fruits.",
        "care_instructions": "Requires full tropical sunlight (8+ hours), deep well-draining soil, and minimal watering once established.",
        "medicinal_uses": "Leaves contain mangiferin, a unique xanthoid polyphenol showing strong anti-diabetic and cellular protection efficacy.",
        "image": "species_images/mango.jpg"
    },
    "Marigold": {
        "scientific_name": "Tagetes erecta",
        "family": "Asteraceae",
        "description": "An aromatic annual flowering plant featuring finely divided pinnatisect leaves and pompon-like orange and yellow flower heads.",
        "care_instructions": "Prefers full sunshine, moderately fertile garden soil, and light watering once established.",
        "medicinal_uses": "Prime commercial source of lutein and zeaxanthin carotenoids critical for macula retinal eye protection.",
        "image": "species_images/marigold.jpg"
    },
    "Mint": {
        "scientific_name": "Mentha spicata",
        "family": "Lamiaceae",
        "description": "A fast-spreading aromatic perennial herb characterized by square stems, opposite serrated leaves, and refreshing menthol aroma.",
        "care_instructions": "Requires rich moist soil and partial shade to morning sun. Spreads vigorously via underground runners.",
        "medicinal_uses": "Menthol relaxes gastrointestinal smooth muscle, eases irritable bowel spasms, and clarifies respiratory passages.",
        "image": "species_images/mint.jpg"
    },
    "Money Plant": {
        "scientific_name": "Epipremnum aureum",
        "family": "Araceae",
        "description": "An evergreen climbing vine featuring glossy, heart-shaped variegated leaves blotched with bright yellow and creamy white patterns.",
        "care_instructions": "Thrives in moderate indirect light and aerated potting mix. Water when upper substrate feels dry.",
        "medicinal_uses": "Certified in NASA clean air research for removing indoor toxins including formaldehyde, xylene, and benzene.",
        "image": "species_images/money_plant.jpg"
    },
    "Neem": {
        "scientific_name": "Azadirachta indica",
        "family": "Meliaceae",
        "description": "A hardy evergreen tree famed across the tropics for its pinnate serrated leaves and profound bioactive limonoid compounds.",
        "care_instructions": "Extremely drought-tolerant, demands full sun and porous sandy loam. Thrives in arid environments.",
        "medicinal_uses": "Ayurvedic panacea providing potent antibacterial, antifungal, anti-inflammatory, and natural pest-repellent benefits.",
        "image": "species_images/neem.jpg"
    },
    "Papaya": {
        "scientific_name": "Carica papaya",
        "family": "Caricaceae",
        "description": "A fast-growing soft-wooded tree-like plant with an unbranched trunk topped by a crown of deeply palmately lobed leaves.",
        "care_instructions": "Requires maximum sunshine, warm weather, and excellent soil drainage; highly sensitive to root waterlogging.",
        "medicinal_uses": "Produces papain proteolytic enzyme; leaf infusions are used clinically to support platelet count during viral episodes.",
        "image": "species_images/papaya.jpg"
    },
    "Peepal": {
        "scientific_name": "Ficus religiosa",
        "family": "Moraceae",
        "description": "The sacred fig or Bodhi tree, identifiable by its iconic heart-shaped leaves with an elongated distinct dripping tail tip.",
        "care_instructions": "Requires expansive space, full sunshine, and deep loamy soil. Highly resilient and long-lived.",
        "medicinal_uses": "Releases oxygen round-the-clock; leaf extracts are used in traditional therapy for asthma and skin soothing.",
        "image": "species_images/peepal.jpg"
    },
    "Potato": {
        "scientific_name": "Solanum tuberosum",
        "family": "Solanaceae",
        "description": "An herbaceous annual crop with compound leaves and star-shaped white-to-purple flowers developing nutrient-storing subterranean stem tubers.",
        "care_instructions": "Needs cool growing season, loose well-aerated hilled soil, and moderate moisture.",
        "medicinal_uses": "Raw potato juice extract contains resistant starch and soothing alkaline potassium compounds for gastric lining support.",
        "image": "species_images/potato.jpg"
    },
    "Rose": {
        "scientific_name": "Rosa rubiginosa",
        "family": "Rosaceae",
        "description": "A prickled woody perennial shrub celebrated across cultures for its exquisitely layered aromatic blossoms and serrated pinnate leaves.",
        "care_instructions": "Needs at least 6 hours of full morning sunshine, rich compost, and regular root-level irrigation.",
        "medicinal_uses": "Rose hips are extraordinarily rich in natural Vitamin C; floral water delivers astringent and mood-soothing effects.",
        "image": "species_images/rose.jpg"
    },
    "Spinach": {
        "scientific_name": "Spinacia oleracea",
        "family": "Amaranthaceae",
        "description": "An edible annual flowering leafy vegetable producing dense rosettes of fleshy, crisp ovate-to-triangular dark green leaves.",
        "care_instructions": "Prefers cool weather, moisture-retentive fertile loam, and full sun to light shade.",
        "medicinal_uses": "Exceptional dietary source of lutein, folate, plant iron, magnesium, and vitamin K1 supporting bone and vascular health.",
        "image": "species_images/spinach.jpg"
    },
    "Sunflower": {
        "scientific_name": "Helianthus annuus",
        "family": "Asteraceae",
        "description": "A magnificent annual plant with tall coarse hairy stems, broad cordate leaves, and massive heliotropic golden flower disks.",
        "care_instructions": "Requires 6-8 hours of direct intense sunlight and deep, moderately fertile, moisture-retentive soil.",
        "medicinal_uses": "Seeds provide heart-healthy unsaturated fatty acids and vitamin E; herbal infusions ease respiratory discomfort.",
        "image": "species_images/sunflower.jpg"
    },
    "Tomato": {
        "scientific_name": "Solanum lycopersicum",
        "family": "Solanaceae",
        "description": "A hairy glandular vine-like herb with distinctive pungent aroma and serrated compound leaves producing succulent edible berries.",
        "care_instructions": "Requires warm temperatures, 8 hours of sunshine, consistent deep hydration, and structural staking support.",
        "medicinal_uses": "Premier source of lycopene, a carotenoid antioxidant proven to reduce cardiovascular oxidative stress.",
        "image": "species_images/tomato.jpg"
    },
    "Tomato Leaf": {
        "scientific_name": "Solanum lycopersicum",
        "family": "Solanaceae",
        "description": "The pinnate compound foliage of the tomato plant, characterized by distinct trichomes (hairs) releasing a sharp aromatic solanaceous scent.",
        "care_instructions": "Ensure good air circulation, avoid wetting leaves during watering to prevent fungal blights.",
        "medicinal_uses": "Contains tomatine glycoalkaloid, investigated for its natural antimicrobial and bio-insecticidal defense mechanisms.",
        "image": "species_images/tomato_leaf.jpg"
    },
    "Tulip": {
        "scientific_name": "Tulipa gesneriana",
        "family": "Liliaceae",
        "description": "A spring-blooming bulbous geophyte producing cup-shaped ornamental blooms in luminous colors above glaucous lanceolate foliage.",
        "care_instructions": "Requires winter cold vernalization, sharp soil drainage, and full morning to midday sunlight.",
        "medicinal_uses": "Cultivated for aesthetic wellness and studied for specialized bioactive flavonoids and anthocyanins.",
        "image": "species_images/tulip.jpg"
    },
    "Tulsi": {
        "scientific_name": "Ocimum tenuiflorum",
        "family": "Lamiaceae",
        "description": "Revered as Holy Basil, an aromatic sacred perennial subshrub with green or purple serrated leaves renowned in Ayurveda as the Queen of Herbs.",
        "care_instructions": "Requires warm tropical sun, well-aerated fertile soil, and regular watering without water stagnation.",
        "medicinal_uses": "Supreme adaptogen containing eugenol and rosmarinic acid that modulates cortisol, enhances immunity, and supports respiratory clarity.",
        "image": "species_images/tulsi.jpg"
    }
}

def seed_plants():
    print(f"Updating botanical species catalog ({len(ALL_SPECIES_DATA)} species)...")
    for common_name, data in ALL_SPECIES_DATA.items():
        obj, created = PlantSpecies.objects.update_or_create(
            common_name=common_name,
            defaults={
                'scientific_name': data['scientific_name'],
                'family': data['family'],
                'description': data['description'],
                'care_instructions': data['care_instructions'],
                'medicinal_uses': data['medicinal_uses'],
                'image': data['image']
            }
        )
        status = "Created" if created else "Updated"
        print(f"  [{status}] {common_name} ({data['scientific_name']}) -> {data['image']}")

    total = PlantSpecies.objects.count()
    print(f"\n[SUCCESS] PlantSpecies database table now contains {total} species!")

if __name__ == '__main__':
    seed_plants()
