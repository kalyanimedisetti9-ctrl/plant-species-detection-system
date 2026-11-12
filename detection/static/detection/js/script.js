/**
 * Plant Species Detection System - Frontend Interactions
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Upload & Drag-and-Drop Handler
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('plant-image-input');
    const previewWrapper = document.getElementById('image-preview-wrapper');
    const previewImg = document.getElementById('preview-image');
    const detectForm = document.getElementById('detect-upload-form');
    const detectButton = document.getElementById('detect-submit-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const fileNameDisplay = document.getElementById('selected-file-name');
    const removeBtn = document.getElementById('remove-image-btn');

    const allowedExtensions = ['jpg', 'jpeg', 'png'];

    if (dropzone && fileInput) {
        // Click dropzone to trigger input
        dropzone.addEventListener('click', (e) => {
            if (e.target !== removeBtn && !removeBtn?.contains(e.target)) {
                fileInput.click();
            }
        });

        // Drag & Drop events
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            });
        });

        // Handle file dropped
        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelection(files[0]);
            }
        });

        // Handle file selected through browser picker
        fileInput.addEventListener('change', (e) => {
            if (fileInput.files.length > 0) {
                handleFileSelection(fileInput.files[0]);
            }
        });

        // Remove image button
        if (removeBtn) {
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                fileInput.value = '';
                if (previewWrapper) previewWrapper.style.display = 'none';
                if (detectButton) detectButton.disabled = true;
                if (fileNameDisplay) fileNameDisplay.textContent = '';
            });
        }
    }

    function handleFileSelection(file) {
        const fileExt = file.name.split('.').pop().toLowerCase();
        
        if (!allowedExtensions.includes(fileExt)) {
            alert('Please select a valid image format (JPG, JPEG, PNG).');
            fileInput.value = '';
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            alert('File size exceeds 10MB limit.');
            fileInput.value = '';
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewImg) {
                previewImg.src = e.target.result;
            }
            if (previewWrapper) {
                previewWrapper.style.display = 'block';
            }
            if (fileNameDisplay) {
                fileNameDisplay.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
            }
            if (detectButton) {
                detectButton.disabled = false;
            }
        };
        reader.readAsDataURL(file);
    }

    // Show loading overlay during detection processing
    if (detectForm) {
        detectForm.addEventListener('submit', (e) => {
            if (!fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                alert('Please select a plant image before clicking Detect.');
                return;
            }
            if (loadingOverlay) {
                loadingOverlay.style.display = 'flex';
            }
        });
    }

    // Auto-dismiss Bootstrap alert messages after 5 seconds
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(alertEl => {
        setTimeout(() => {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alertEl);
            if (bsAlert) bsAlert.close();
        }, 5000);
    });
});
