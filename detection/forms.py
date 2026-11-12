import os
from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from .models import PlantSpecies, Detection, ContactMessage

ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
MAX_IMAGE_SIZE_MB = 10

class ImageUploadForm(forms.ModelForm):
    """
    Form for validating and handling plant image uploads for AI detection.
    """
    class Meta:
        model = Detection
        fields = ['uploaded_image']
        widgets = {
            'uploaded_image': forms.FileInput(attrs={
                'class': 'form-control visually-hidden',
                'id': 'plant-image-input',
                'accept': 'image/jpeg,image/png,image/jpg'
            })
        }

    def clean_uploaded_image(self):
        image = self.cleaned_data.get('uploaded_image')
        if not image:
            raise ValidationError("Please select a plant image to upload.")

        # Validate file extension
        ext = os.path.splitext(image.name)[1].lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise ValidationError("Please upload a JPG, JPEG, or PNG image format.")

        # Validate file size
        if image.size > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise ValidationError(f"Image file size cannot exceed {MAX_IMAGE_SIZE_MB}MB.")

        return image


class UserRegistrationForm(forms.ModelForm):
    """
    User registration form with password confirmation and clean styling.
    """
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Enter your email address'})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Create a password (min. 6 characters)'}),
        min_length=6
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Confirm your password'}),
        min_length=6
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'password']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Choose a unique username'}),
        }

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if User.objects.filter(username__iexact=username).exists():
            raise ValidationError("This username is already taken. Please choose another one.")
        return username

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email__iexact=email).exists():
            raise ValidationError("An account with this email address already exists.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        p1 = cleaned_data.get('password')
        p2 = cleaned_data.get('confirm_password')

        if p1 and p2 and p1 != p2:
            self.add_error('confirm_password', "Passwords do not match.")

        return cleaned_data


class ContactForm(forms.ModelForm):
    """
    Contact form for feedback, research inquiries, and plant suggestions.
    """
    class Meta:
        model = ContactMessage
        fields = ['name', 'email', 'subject', 'message']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Your Full Name'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Your Email Address'}),
            'subject': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Inquiry Subject'}),
            'message': forms.Textarea(attrs={'class': 'form-control', 'rows': 5, 'placeholder': 'Type your message or inquiry here...'}),
        }
