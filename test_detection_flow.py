"""
End-to-End Verification Test Script
Tests OpenCV Preprocessing, TensorFlow Prediction, Django URL Routing, and Database Flow.
"""

import os
import django
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'plant_detection.settings')
django.setup()

from django.test import Client
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from detection.models import PlantSpecies, Detection
from ml_model.prediction import predict_plant

BASE_DIR = Path(__file__).resolve().parent

def run_tests():
    print("=== STARTING COMPREHENSIVE END-TO-END SYSTEM TEST ===")
    
    # 1. Test ML & OpenCV Preprocessing
    sample_rose_path = BASE_DIR / 'media' / 'species_images' / 'rose.jpg'
    print(f"\n1. Testing ML & OpenCV Inference on: {sample_rose_path}")
    assert sample_rose_path.exists(), "Sample test image does not exist!"

    res = predict_plant(str(sample_rose_path))
    print(f"Prediction Output: Species='{res['predicted_species']}', Scientific='{res['scientific_name']}', Confidence={res['confidence']}%")
    assert 'predicted_species' in res
    assert 'scientific_name' in res
    assert res['confidence'] > 0
    print("[PASS] ML Prediction and OpenCV Preprocessing Pipeline Passed.")

    # 2. Test Django Client Routes
    client = Client()
    routes = [
        ('/', 200, "Home Page"),
        ('/about/', 200, "About Page"),
        ('/how-it-works/', 200, "How It Works Page"),
        ('/species/', 200, "Plant Species Catalog"),
        ('/contact/', 200, "Contact Page"),
        ('/login/', 200, "Login Page"),
        ('/register/', 200, "Register Page"),
        ('/detect/', 200, "Detect Upload Page"),
    ]

    print("\n2. Testing Public Web Endpoints:")
    for url, expected_code, name in routes:
        response = client.get(url)
        print(f"Testing {name} ({url}) -> Status: {response.status_code}")
        assert response.status_code == expected_code, f"Failed on {name}: expected {expected_code}, got {response.status_code}"
    print("[PASS] All Public URL Endpoints returned 200 OK.")

    # 3. Test Detection Upload & Result Redirection
    print("\n3. Testing Plant Detection Upload Flow via POST:")
    with open(sample_rose_path, 'rb') as f:
        uploaded = SimpleUploadedFile("test_rose.jpg", f.read(), content_type="image/jpeg")
        post_response = client.post('/detect/', {'uploaded_image': uploaded}, follow=True)
        print(f"POST /detect/ status code: {post_response.status_code}")
        assert post_response.status_code == 200
        # Check that result page was rendered
        assert "Prediction Confidence" in post_response.content.decode('utf-8')
    print("[PASS] Upload, OpenCV processing, ML inference, and result rendering passed.")

    # 4. Test User Authentication & Protected Routes
    print("\n4. Testing User Login, Dashboard, and History:")
    login_success = client.login(username='student', password='student123')
    assert login_success, "Student login failed!"
    print("User 'student' successfully authenticated.")

    dash_response = client.get('/dashboard/')
    assert dash_response.status_code == 200
    print(f"Dashboard status: {dash_response.status_code}")

    hist_response = client.get('/history/')
    assert hist_response.status_code == 200
    print(f"History status: {hist_response.status_code}")

    csv_response = client.get('/history/export-csv/')
    assert csv_response.status_code == 200
    assert csv_response['Content-Type'] == 'text/csv'
    print(f"CSV Export status: {csv_response.status_code}")
    print("[PASS] User Authentication, Dashboard, History & CSV export passed.")

    # 5. Test REST API Endpoints
    print("\n5. Testing REST API Endpoints:")
    api_species_res = client.get('/api/species/')
    assert api_species_res.status_code == 200
    api_data = api_species_res.json()
    assert api_data['status'] == 'success'
    print(f"GET /api/species/ -> Found {api_data['count']} botanical species.")

    with open(sample_rose_path, 'rb') as f:
        api_upload_file = SimpleUploadedFile("api_test_rose.jpg", f.read(), content_type="image/jpeg")
        api_detect_res = client.post('/api/detect/', {'image': api_upload_file})
        assert api_detect_res.status_code == 200
        detect_json = api_detect_res.json()
        assert detect_json['status'] == 'success'
        print(f"POST /api/detect/ -> Predicted: {detect_json['predicted_species']} ({detect_json['scientific_name']}) at {detect_json['confidence']}%")
    print("[PASS] REST API Endpoints passed.")

    # 6. Test Admin portal
    print("\n6. Testing Django Admin Portal:")
    admin_client = Client()
    admin_login = admin_client.login(username='admin', password='admin123')
    assert admin_login, "Admin login failed!"
    admin_response = admin_client.get('/admin/')
    assert admin_response.status_code == 200
    print(f"Admin Dashboard status: {admin_response.status_code}")
    print("[PASS] Admin Portal is fully accessible and responsive.")

    print("\n=== ALL SYSTEM TESTS COMPLETED SUCCESSFULLY (100% PASS) ===")

if __name__ == '__main__':
    run_tests()
