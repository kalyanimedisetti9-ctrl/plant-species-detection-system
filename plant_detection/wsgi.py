"""
WSGI config for plant_detection project.
"""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'plant_detection.settings')
application = get_wsgi_application()
