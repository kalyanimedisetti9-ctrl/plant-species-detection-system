import os
import csv
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.db.models import Q
from django.conf import settings

from .models import PlantSpecies, Detection, ContactMessage
from .forms import ImageUploadForm, UserRegistrationForm, ContactForm
from ml_model.prediction import predict_plant

def home_view(request):
    """
    Renders the homepage featuring hero section, AI highlights,
    visual workflow preview, recent statistics, and call-to-action.
    """
    total_species = PlantSpecies.objects.count()
    total_detections = Detection.objects.count()
    featured_species = PlantSpecies.objects.all()[:6]

    context = {
        'total_species': total_species,
        'total_detections': total_detections,
        'featured_species': featured_species,
    }
    return render(request, 'detection/home.html', context)


def about_view(request):
    """
    Renders the About page explaining project background, diploma objectives,
    and technological stack (Python, Django, TensorFlow, OpenCV, MySQL, Bootstrap).
    """
    return render(request, 'detection/about.html')


def how_it_works_view(request):
    """
    Renders the system workflow page explaining image acquisition,
    OpenCV preprocessing, deep learning inference, and botanical output.
    """
    return render(request, 'detection/how_it_works.html')


def detect_view(request):
    """
    Main detection interface. Accepts plant image upload, validates format,
    triggers OpenCV preprocessing and TensorFlow/Keras prediction, records history,
    and forwards to the result view.
    """
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Save uploaded image initially
                detection_instance = form.save(commit=False)
                if request.user.is_authenticated:
                    detection_instance.user = request.user
                
                # We need the file on disk to run OpenCV preprocessing
                detection_instance.save()
                
                image_path = detection_instance.uploaded_image.path

                # Execute TensorFlow / Keras prediction pipeline (includes Plant vs Non-Plant validation)
                prediction_result = predict_plant(image_path)

                # Reject non-plant images (hands, humans, vehicles, phones, buildings, animals, etc.)
                if not prediction_result.get('is_plant', True):
                    try:
                        if os.path.exists(image_path):
                            os.remove(image_path)
                    except Exception:
                        pass
                    detection_instance.delete()

                    reject_msg = prediction_result.get('message') or "Plant Not Detected! Please upload a valid plant or leaf image."
                    messages.error(request, reject_msg)
                    return redirect('detect_plant')

                predicted_name = prediction_result.get('predicted_species', 'Unknown Plant')
                scientific_name = prediction_result.get('scientific_name', '')
                confidence = prediction_result.get('confidence', 0.0)

                # Find associated PlantSpecies database record if exists
                species_obj = PlantSpecies.objects.filter(common_name__iexact=predicted_name).first()
                if not species_obj:
                    species_obj = PlantSpecies.objects.filter(scientific_name__iexact=scientific_name).first()

                # Update detection instance with prediction results
                detection_instance.predicted_species = predicted_name
                detection_instance.scientific_name = scientific_name or (species_obj.scientific_name if species_obj else "")
                detection_instance.confidence = confidence
                detection_instance.plant_species_ref = species_obj
                detection_instance.save()

                # Store prediction metadata in session for detailed display
                request.session['last_top_predictions'] = prediction_result.get('top_predictions', [])

                is_confident = prediction_result.get('is_confident', True)
                if is_confident:
                    messages.success(request, f"Successfully identified: {predicted_name} with {confidence}% confidence!")
                else:
                    messages.warning(request, "Unable to identify this plant confidently. Please upload a clear plant or leaf image.")
                return redirect('detection_result', pk=detection_instance.pk)

            except Exception as e:
                messages.error(request, f"Detection error: Unable to process image. {str(e)}")
                return redirect('detect_plant')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
    else:
        form = ImageUploadForm()

    return render(request, 'detection/upload.html', {'form': form})


def result_view(request, pk):
    """
    Displays the prediction result card with image, common name, scientific name,
    confidence progress bar, botanical characteristics, and top candidates.
    """
    detection = get_object_or_404(Detection, pk=pk)
    top_predictions = request.session.get('last_top_predictions', [])

    # Retrieve botanical species details from database
    species_info = detection.plant_species_ref
    if not species_info:
        species_info = PlantSpecies.objects.filter(common_name__iexact=detection.predicted_species).first()

    context = {
        'detection': detection,
        'species_info': species_info,
        'top_predictions': top_predictions,
    }
    return render(request, 'detection/result.html', context)


def species_list_view(request):
    """
    Displays the botanical plant species catalog with search and family filtering.
    """
    query = request.GET.get('q', '').strip()
    family_filter = request.GET.get('family', '').strip()

    species_qs = PlantSpecies.objects.all()

    if query:
        species_qs = species_qs.filter(
            Q(common_name__icontains=query) |
            Q(scientific_name__icontains=query) |
            Q(description__icontains=query) |
            Q(family__icontains=query)
        )

    if family_filter:
        species_qs = species_qs.filter(family__iexact=family_filter)

    families = PlantSpecies.objects.values_list('family', flat=True).distinct().exclude(family='')

    paginator = Paginator(species_qs, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'query': query,
        'family_filter': family_filter,
        'families': sorted(list(families)),
    }
    return render(request, 'detection/species.html', context)


def species_detail_view(request, pk):
    """
    Detailed botanical profile for a single plant species.
    """
    species = get_object_or_404(PlantSpecies, pk=pk)
    related_species = PlantSpecies.objects.filter(family=species.family).exclude(pk=species.pk)[:3]

    context = {
        'species': species,
        'related_species': related_species,
    }
    return render(request, 'detection/species_detail.html', context)


def contact_view(request):
    """
    Handles user inquiries, research questions, and contact messages.
    """
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Thank you! Your message has been sent successfully. We will get back to you soon.")
            return redirect('contact')
        else:
            messages.error(request, "Please correct the errors in the form below.")
    else:
        form = ContactForm()

    return render(request, 'detection/contact.html', {'form': form})


def register_view(request):
    """
    Registers a new student, researcher, or user account.
    """
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            login(request, user)
            messages.success(request, f"Welcome, {user.username}! Your account has been created successfully.")
            return redirect('dashboard')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field.capitalize()}: {error}")
    else:
        form = UserRegistrationForm()

    return render(request, 'detection/register.html', {'form': form})


def login_view(request):
    """
    User login view with clean error feedback.
    """
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {user.username}!")
                next_url = request.GET.get('next', 'dashboard')
                return redirect(next_url)
        else:
            messages.error(request, "Invalid username or password. Please try again.")
    else:
        form = AuthenticationForm()

    return render(request, 'detection/login.html', {'form': form})


def logout_view(request):
    """
    Logs out the user safely.
    """
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('home')


@login_required
def dashboard_view(request):
    """
    User Dashboard showing personal stats, total detections, recent detections, and shortcuts.
    """
    user_detections = Detection.objects.filter(user=request.user)
    total_user_detections = user_detections.count()
    recent_detections = user_detections[:5]

    # Calculate average confidence
    avg_confidence = 0
    if total_user_detections > 0:
        total_conf = sum(d.confidence for d in user_detections)
        avg_confidence = round(total_conf / total_user_detections, 1)

    context = {
        'total_user_detections': total_user_detections,
        'recent_detections': recent_detections,
        'avg_confidence': avg_confidence,
    }
    return render(request, 'detection/dashboard.html', context)


@login_required
def history_view(request):
    """
    Detection History view displaying all scans performed by the authenticated user.
    """
    history_list = Detection.objects.filter(user=request.user)
    paginator = Paginator(history_list, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'detection/history.html', {'page_obj': page_obj})


@login_required
def delete_detection_view(request, pk):
    """
    Deletes a specific detection record from user's history.
    """
    detection = get_object_or_404(Detection, pk=pk, user=request.user)
    if request.method == 'POST':
        # Remove media file from disk if exists
        try:
            if detection.uploaded_image and os.path.exists(detection.uploaded_image.path):
                os.remove(detection.uploaded_image.path)
        except Exception:
            pass
        detection.delete()
        messages.success(request, "Detection record removed successfully.")
    return redirect('history')


@login_required
def export_history_csv_view(request):
    """
    Exports the authenticated user's detection history as a downloadable CSV spreadsheet.
    """
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="plant_detection_history_{request.user.username}.csv"'

    writer = csv.writer(response)
    writer.writerow(['ID', 'Common Name', 'Scientific Name', 'Confidence (%)', 'Date & Time'])

    detections = Detection.objects.filter(user=request.user)
    for d in detections:
        writer.writerow([
            d.id,
            d.predicted_species,
            d.scientific_name,
            f"{d.confidence:.2f}",
            d.detected_at.strftime('%Y-%m-%d %H:%M:%S')
        ])

    return response


@csrf_exempt
def api_detect_view(request):
    """
    REST API endpoint for programmatic plant detection.
    Accepts: POST request with 'image' file field.
    Returns: JSON response containing prediction details.
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed.'}, status=405)

    if 'image' not in request.FILES and 'uploaded_image' not in request.FILES:
        return JsonResponse({'status': 'error', 'message': 'No image file provided in request.'}, status=400)

    uploaded_file = request.FILES.get('image') or request.FILES.get('uploaded_image')
    
    # Validate extension
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        return JsonResponse({'status': 'error', 'message': 'Invalid file format. Only JPG, JPEG, and PNG are supported.'}, status=400)

    try:
        # Create detection record
        detection = Detection(uploaded_image=uploaded_file)
        if request.user.is_authenticated:
            detection.user = request.user
        detection.save()

        # Run prediction (includes Plant vs Non-Plant validation)
        result = predict_plant(detection.uploaded_image.path)

        if not result.get('is_plant', True):
            try:
                if os.path.exists(detection.uploaded_image.path):
                    os.remove(detection.uploaded_image.path)
            except Exception:
                pass
            detection.delete()
            return JsonResponse({
                'status': 'error',
                'is_plant': False,
                'message': result.get('message', 'Plant Not Detected! Please upload a valid plant or leaf image.'),
                'reason': result.get('reason', 'non_plant')
            }, status=400)
        species_name = result.get('predicted_species', 'Unknown')
        scientific_name = result.get('scientific_name', '')
        confidence = result.get('confidence', 0.0)

        # Lookup DB reference
        species_obj = PlantSpecies.objects.filter(common_name__iexact=species_name).first()
        if not species_obj:
            species_obj = PlantSpecies.objects.filter(scientific_name__iexact=scientific_name).first()

        detection.predicted_species = species_name
        detection.scientific_name = scientific_name or (species_obj.scientific_name if species_obj else "")
        detection.confidence = confidence
        detection.plant_species_ref = species_obj
        detection.save()

        return JsonResponse({
            'status': 'success',
            'detection_id': detection.id,
            'predicted_species': species_name,
            'scientific_name': detection.scientific_name,
            'confidence': confidence,
            'image_url': request.build_absolute_uri(detection.uploaded_image.url),
            'top_predictions': result.get('top_predictions', []),
            'botanical_info': {
                'family': species_obj.family if species_obj else '',
                'description': species_obj.description if species_obj else '',
                'care_instructions': species_obj.care_instructions if species_obj else '',
                'medicinal_uses': species_obj.medicinal_uses if species_obj else ''
            }
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def api_species_list_view(request):
    """
    REST API endpoint listing cataloged botanical species.
    """
    species = PlantSpecies.objects.all()
    data = []
    for s in species:
        data.append({
            'id': s.id,
            'common_name': s.common_name,
            'scientific_name': s.scientific_name,
            'family': s.family,
            'description': s.description,
            'care_instructions': s.care_instructions,
            'medicinal_uses': s.medicinal_uses,
            'image_url': request.build_absolute_uri(s.image.url) if s.image else None
        })
    return JsonResponse({'status': 'success', 'count': len(data), 'species': data})

