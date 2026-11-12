"""
Image Preprocessing Module for Plant Species Detection System
Uses OpenCV (cv2) and NumPy to prepare input images for TensorFlow/Keras deep learning model.
"""

import os
import cv2
import numpy as np

# Standard input dimensions for CNN / MobileNetV2 architecture
TARGET_IMAGE_SIZE = (224, 224)

def preprocess_image_from_path(image_path, target_size=TARGET_IMAGE_SIZE):
    """
    Loads an image from disk using OpenCV, preprocesses it, and converts it
    into a normalized tensor ready for model inference.

    Steps:
    1. Read Image using OpenCV (cv2.imread)
    2. Convert Color from BGR to RGB
    3. Resize Image to target dimensions (e.g. 224x224)
    4. Normalize pixel intensity values to [0.0, 1.0]
    5. Expand batch dimensions: shape (1, 224, 224, 3)

    Args:
        image_path (str): File system path to the image.
        target_size (tuple): Target width and height (width, height).

    Returns:
        np.ndarray: Preprocessed image batch tensor of shape (1, target_height, target_width, 3).

    Raises:
        ValueError: If image file cannot be read or is corrupted.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")

    # Step 1: Read image via OpenCV
    # cv2.imdecode is used if path has non-ascii characters, cv2.imread for standard
    try:
        # Read raw byte buffer to support Windows unicode path strings reliably
        with open(image_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception:
        img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Unable to decode or read the uploaded image. The file may be corrupted.")

    # Step 2: Convert color space BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 3: Resize image to target input dimensions
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)

    # Step 4: Convert to float32 but keep pixel values in [0, 255] range.
    # The model has a built-in Rescaling layer that converts [0,255] -> [-1,1]
    # for MobileNetV2. Do NOT divide by 255 here — that causes double-normalization.
    img_float = img_resized.astype(np.float32)

    # Step 5: Expand dimensions to match model input batch shape: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_float, axis=0)

    return img_batch


def extract_image_metadata(image_path):
    """
    Extracts basic image properties (resolution, channels, aspect ratio) using OpenCV.
    """
    try:
        with open(image_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is not None:
            height, width, channels = img.shape
            return {
                'width': width,
                'height': height,
                'channels': channels,
                'aspect_ratio': round(width / height, 2)
            }
    except Exception:
        pass
    return None
