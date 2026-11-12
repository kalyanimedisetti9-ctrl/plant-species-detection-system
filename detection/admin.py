from django.contrib import admin
from django.utils.html import format_html
from .models import PlantSpecies, Detection, ContactMessage

# Admin Site Customization
admin.site.site_header = "Plant Species Detection System - Admin Portal"
admin.site.site_title = "Plant Species Admin"
admin.site.index_title = "Botanical Dataset & Detection Management"

@admin.register(PlantSpecies)
class PlantSpeciesAdmin(admin.ModelAdmin):
    list_display = ('thumbnail_preview', 'common_name', 'scientific_name_display', 'family', 'created_at')
    list_filter = ('family', 'created_at')
    search_fields = ('common_name', 'scientific_name', 'family', 'description')
    ordering = ('common_name',)
    readonly_fields = ('thumbnail_preview', 'created_at', 'updated_at')

    def thumbnail_preview(self, obj):
        if obj.image:
            return format_html('<img src="{}" style="width: 50px; height: 50px; object-fit: cover; border-radius: 6px;" />', obj.image.url)
        return format_html('<span style="color: #999;">No Image</span>')
    thumbnail_preview.short_description = "Image"

    def scientific_name_display(self, obj):
        return format_html('<i>{}</i>', obj.scientific_name)
    scientific_name_display.short_description = "Scientific Name"


@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ('thumbnail_preview', 'predicted_species', 'scientific_name_display', 'confidence_badge', 'user_display', 'detected_at')
    list_filter = ('predicted_species', 'detected_at')
    search_fields = ('predicted_species', 'scientific_name', 'user__username', 'user__email')
    ordering = ('-detected_at',)
    readonly_fields = ('thumbnail_preview', 'detected_at')

    def thumbnail_preview(self, obj):
        if obj.uploaded_image:
            return format_html('<img src="{}" style="width: 50px; height: 50px; object-fit: cover; border-radius: 6px;" />', obj.uploaded_image.url)
        return "No Image"
    thumbnail_preview.short_description = "Uploaded Image"

    def scientific_name_display(self, obj):
        return format_html('<i>{}</i>', obj.scientific_name)
    scientific_name_display.short_description = "Scientific Name"

    def confidence_badge(self, obj):
        color = "#198754" if obj.confidence >= 80 else ("#ffc107" if obj.confidence >= 50 else "#dc3545")
        return format_html('<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 12px; font-weight: bold; font-size: 11px;">{:.1f}%</span>', color, obj.confidence)
    confidence_badge.short_description = "Confidence"

    def user_display(self, obj):
        return obj.user.username if obj.user else "Anonymous User"
    user_display.short_description = "User"


@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ('subject', 'name', 'email', 'created_at', 'is_resolved')
    list_filter = ('is_resolved', 'created_at')
    search_fields = ('name', 'email', 'subject', 'message')
    ordering = ('-created_at',)
    actions = ['mark_as_resolved']

    def mark_as_resolved(self, request, queryset):
        queryset.update(is_resolved=True)
        self.message_user(request, "Selected messages marked as resolved.")
    mark_as_resolved.short_description = "Mark selected messages as Resolved"
