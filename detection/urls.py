from django.urls import path
from . import views

urlpatterns = [
    # Core public pages
    path('', views.home_view, name='home'),
    path('about/', views.about_view, name='about'),
    path('how-it-works/', views.how_it_works_view, name='how_it_works'),
    path('detect/', views.detect_view, name='detect_plant'),
    path('result/<int:pk>/', views.result_view, name='detection_result'),
    path('species/', views.species_list_view, name='species_list'),
    path('species/<int:pk>/', views.species_detail_view, name='species_detail'),
    path('contact/', views.contact_view, name='contact'),

    # User Authentication & Management
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('history/', views.history_view, name='history'),
    path('history/export-csv/', views.export_history_csv_view, name='export_history_csv'),
    path('history/delete/<int:pk>/', views.delete_detection_view, name='delete_detection'),

    # REST API Endpoints
    path('api/detect/', views.api_detect_view, name='api_detect'),
    path('api/species/', views.api_species_list_view, name='api_species_list'),
]
