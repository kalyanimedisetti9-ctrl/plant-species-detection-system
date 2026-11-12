import os
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

os.makedirs('media/test_validation_samples', exist_ok=True)

# 1. Synthesize Human Hand / Skin Tone patch
hand_img = Image.new('RGB', (300, 300), color=(225, 172, 131)) # skin tone
d = ImageDraw.Draw(hand_img)
d.rectangle([100, 80, 160, 260], fill=(210, 155, 115))
hand_img.save('media/test_validation_samples/hand_sample.jpg')

# 2. Synthesize Vehicle / Car patch (metal silver & wheels)
car_img = Image.new('RGB', (300, 300), color=(180, 180, 185))
d2 = ImageDraw.Draw(car_img)
d2.rectangle([40, 100, 260, 200], fill=(40, 80, 160)) # car body
d2.ellipse([60, 180, 100, 220], fill=(20, 20, 20)) # wheel
d2.ellipse([200, 180, 240, 220], fill=(20, 20, 20)) # wheel
car_img.save('media/test_validation_samples/vehicle_sample.jpg')

# 3. Synthesize Phone patch
phone_img = Image.new('RGB', (300, 300), color=(240, 240, 240))
d3 = ImageDraw.Draw(phone_img)
d3.rectangle([80, 40, 220, 260], fill=(20, 20, 20)) # phone bezel
d3.rectangle([90, 60, 210, 240], fill=(70, 130, 180)) # screen
phone_img.save('media/test_validation_samples/phone_sample.jpg')

print("Created test non-plant synthetic samples successfully.")

