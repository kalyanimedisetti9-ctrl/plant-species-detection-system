"""
Simulates an end-to-end browser form upload of brinjal.jpg to /detect/
and asserts the result page displays Brinjal with correct scientific name.
"""

import re
import requests

s = requests.Session()
r = s.get('http://localhost:8000/detect/')
csrf = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text).group(1)

with open('media/test_samples/brinjal.jpg', 'rb') as f:
    res = s.post(
        'http://localhost:8000/detect/',
        data={'csrfmiddlewaretoken': csrf},
        files={'uploaded_image': ('brinjal.jpg', f, 'image/jpeg')},
        allow_redirects=True
    )

print("Status:", res.status_code)
print("Final URL:", res.url)
assert "Brinjal" in res.text, "Brinjal not found in result page!"
assert "Solanum melongena" in res.text, "Scientific name Solanum melongena not found!"
print("[SUCCESS] End-to-end upload test passed: Brinjal (Solanum melongena) correctly identified and displayed!")
