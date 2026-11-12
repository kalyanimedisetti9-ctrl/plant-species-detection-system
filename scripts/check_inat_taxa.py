import requests

classes_taxa = {
    'Aloe Vera': 'Aloe vera',
    'Ashoka': 'Saraca asoca',
    'Banana': 'Musa acuminata',
    'Banyan': 'Ficus benghalensis',
    'Brinjal': 'Solanum melongena',
    'Chilli': 'Capsicum annuum',
    'Coconut': 'Cocos nucifera',
    'Coriander': 'Coriandrum sativum',
    'Cotton': 'Gossypium hirsutum',
    'Curry Leaf': 'Murraya koenigii',
    'Daisy': 'Bellis perennis',
    'Dandelion': 'Taraxacum officinale',
    'Fenugreek': 'Trigonella foenum-graecum',
    'Guava': 'Psidium guajava',
    'Hibiscus': 'Hibiscus rosa-sinensis',
    'Jackfruit': 'Artocarpus heterophyllus',
    'Jasmine': 'Jasminum officinale',
    'Lavender': 'Lavandula angustifolia',
    'Lemon': 'Citrus limon',
    'Lotus': 'Nelumbo nucifera',
    'Mango': 'Mangifera indica',
    'Marigold': 'Tagetes erecta',
    'Mint': 'Mentha spicata',
    'Money Plant': 'Epipremnum aureum',
    'Neem': 'Azadirachta indica',
    'Papaya': 'Carica papaya',
    'Peepal': 'Ficus religiosa',
    'Potato': 'Solanum tuberosum',
    'Rose': 'Rosa',
    'Spinach': 'Spinacia oleracea',
    'Sunflower': 'Helianthus annuus',
    'Tomato': 'Solanum lycopersicum',
    'Tulip': 'Tulipa gesneriana',
    'Tulsi': 'Ocimum tenuiflorum'
}

for name, sciname in classes_taxa.items():
    try:
        r = requests.get('https://api.inaturalist.org/v1/taxa', params={'q': sciname}, timeout=6)
        results = r.json().get('results', [])
        if results:
            t = results[0]
            print(f"{name:15s} | {sciname:25s} | Taxon ID: {t['id']:7d} | Observations: {t.get('observations_count', 0)}")
        else:
            print(f"{name:15s} | {sciname:25s} | NOT FOUND")
    except Exception as e:
        print(f"{name:15s} | ERROR: {e}")

