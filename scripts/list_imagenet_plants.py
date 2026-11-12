import json

idx = json.load(open('ml_model/imagenet_class_index.json'))
classes = [v[1] for v in idx.values()]

# Search for plant/floral/fruit/veggie classes in ImageNet
plant_like = []
for i, name in enumerate(classes):
    n = name.lower()
    for w in ['plant', 'flower', 'tree', 'leaf', 'fruit', 'vegetable', 'daisy', 'sunflower', 'rose', 'tulip', 'pot', 'herb', 'grass', 'mushroom', 'fungus', 'corn', 'cabbage', 'broccoli', 'squash', 'cucumber', 'pepper', 'orange', 'lemon', 'banana', 'apple', 'fig', 'pineapple', 'jackfruit', 'pomegranate', 'strawberry', 'acorn']:
        if w in n:
            plant_like.append((i, name))
            break

print("Identified plant-like ImageNet classes count:", len(plant_like))
for p in plant_like:
    print(f"{p[0]}: {p[1]}")

