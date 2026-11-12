from django.db import models
from django.contrib.auth.models import User

class PlantSpecies(models.Model):
    """
    Stores comprehensive information regarding cataloged plant species,
    including botanical taxonomy, common names, and descriptions.
    """
    common_name = models.CharField(max_length=150, unique=True, help_text="Common name of the plant (e.g., Rose)")
    scientific_name = models.CharField(max_length=200, help_text="Botanical scientific name in Latin (e.g., Rosa rubiginosa)")
    family = models.CharField(max_length=150, blank=True, help_text="Biological plant family (e.g., Rosaceae)")
    description = models.TextField(help_text="Detailed botanical description, habitat, and characteristics")
    care_instructions = models.TextField(blank=True, help_text="Care requirements: sunlight, watering, soil, temperature")
    medicinal_uses = models.TextField(blank=True, help_text="Medicinal, ecological, or economic benefits")
    image = models.ImageField(upload_to='species_images/', blank=True, null=True, help_text="Representative reference image")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Plant Species"
        verbose_name_plural = "Plant Species"
        ordering = ['common_name']

    def __str__(self):
        return f"{self.common_name} ({self.scientific_name})"


class Detection(models.Model):
    """
    Records each plant image uploaded and analyzed by the ML model.
    """
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='detections')
    uploaded_image = models.ImageField(upload_to='uploaded_images/')
    predicted_species = models.CharField(max_length=150, default='Pending')
    scientific_name = models.CharField(max_length=200, default='', blank=True)
    confidence = models.FloatField(default=0.0, help_text="Prediction confidence percentage (0 to 100)")
    plant_species_ref = models.ForeignKey(PlantSpecies, on_delete=models.SET_NULL, null=True, blank=True, related_name='detection_logs')
    detected_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Detection Record"
        verbose_name_plural = "Detection Records"
        ordering = ['-detected_at']

    def __str__(self):
        username = self.user.username if self.user else "Anonymous"
        return f"{self.predicted_species} ({self.confidence:.1f}%) - {username} on {self.detected_at.strftime('%Y-%m-%d %H:%M')}"


class ContactMessage(models.Model):
    """
    Stores contact messages submitted through the website contact form.
    """
    name = models.CharField(max_length=120)
    email = models.EmailField()
    subject = models.CharField(max_length=200)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_resolved = models.BooleanField(default=False)

    class Meta:
        verbose_name = "Contact Message"
        verbose_name_plural = "Contact Messages"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.subject} by {self.name} ({self.email})"
