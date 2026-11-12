import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'plant_detection.settings')
os.environ['KERAS_HOME'] = str(PROJECT_ROOT / '.keras')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import django
django.setup()

from django.test import Client
from django.core.files.uploadedfile import SimpleUploadedFile
from detection.models import Detection

client = Client()

test_cases = [
    # 4 Plants
    ("Plant: Brinjal", "media/uploaded_images/brinjal_nZ64tNt.jpeg", True, "Brinjal"),
    ("Plant: Mango", "media/species_images/mango.jpg", True, "Mango"),
    ("Plant: Hibiscus", "media/uploaded_images/hibiscus_c1hiAku.jpeg", True, "Hibiscus"),
    ("Plant: Aloe Vera", "media/uploaded_images/alovera_pC3w3ZJ.jpeg", True, "Aloe Vera"),
    # 6 Non-Plants
    ("Non-Plant: Hand", "media/test_non_plant_samples/hand.jpg", False, "Not a Plant \u2013 Please upload a valid plant image."),
    ("Non-Plant: Person", "media/test_non_plant_samples/person.jpg", False, "Not a Plant \u2013 Please upload a valid plant image."),
    ("Non-Plant: Dog", "media/test_non_plant_samples/dog.jpg", False, "Plant Not Detected! Please upload a valid plant or leaf image."),
    ("Non-Plant: Phone", "media/test_non_plant_samples/phone.jpg", False, "Plant Not Detected! Please upload a valid plant or leaf image."),
    ("Non-Plant: Vehicle", "media/test_non_plant_samples/vehicle.jpg", False, "Plant Not Detected! Please upload a valid plant or leaf image."),
    ("Non-Plant: Building", "media/test_non_plant_samples/building.jpg", False, "Plant Not Detected! Please upload a valid plant or leaf image."),
]

print("================================================================================")
print("COMPREHENSIVE END-TO-END VERIFICATION: DJANGO VIEW & REST API")
print("================================================================================\n")

all_passed = True
initial_count = Detection.objects.count()

for label, rel_path, should_accept, expected_text in test_cases:
    full_path = PROJECT_ROOT / rel_path
    with open(full_path, 'rb') as f:
        file_content = f.read()
    
    # --- 1. Test Browser Upload View (/detect/) ---
    uploaded = SimpleUploadedFile(full_path.name, file_content, content_type='image/jpeg')
    response = client.post('/detect/', {'uploaded_image': uploaded}, follow=True)
    html_content = response.content.decode('utf-8', errors='ignore')
    
    # --- 2. Test REST API Endpoint (/api/detect/) ---
    uploaded_api = SimpleUploadedFile(full_path.name, file_content, content_type='image/jpeg')
    api_response = client.post('/api/detect/', {'image': uploaded_api})
    api_data = api_response.json()
    
    if should_accept:
        # Should redirect to result and show predicted species
        passed = (response.status_code == 200 and 
                  api_response.status_code == 200 and 
                  api_data.get('status') == 'success' and
                  expected_text.lower() in api_data.get('predicted_species', '').lower())
        status_str = "PASS [ACCEPTED]" if passed else "FAIL [UNEXPECTED REJECTION]"
        detail = f"Predicted: {api_data.get('predicted_species')} ({api_data.get('confidence')}%)"
    else:
        # Non-plant should be rejected:
        # Web UI: redirected back to /detect/ with the error message in the HTML alert banner
        # REST API: HTTP 400 with the exact error message in JSON
        ui_rejected = (len(response.redirect_chain) > 0 and 
                       response.redirect_chain[0][0] == '/detect/' and 
                       expected_text in html_content)
        api_rejected = (api_response.status_code == 400 and 
                        api_data.get('status') == 'error' and 
                        api_data.get('is_plant') is False and 
                        api_data.get('message') == expected_text)
        
        passed = ui_rejected and api_rejected
        status_str = "PASS [REJECTED]" if passed else "FAIL [MISCLASSIFIED AS PLANT]"
        detail = f"UI Banner: OK | API 400: \"{api_data.get('message')}\""

    if not passed:
        all_passed = False
        
    print(f"[{status_str}] {label:20} -> {detail}")

final_count = Detection.objects.count()
# For the 4 plants, 4 records from detect view + 4 from api view = +8
print(f"\nDetection Records Created: {final_count - initial_count} (Expected: exactly 8 for the 4 plants across 2 test methods, 0 for non-plants)")
# Clean up test detection records
Detection.objects.filter(id__gt=initial_count).delete()

print(f"\nFINAL SYSTEM STATUS: {'ALL CHECKS PASSED PERFECTLY' if all_passed else 'SOME CHECKS FAILED'}")
