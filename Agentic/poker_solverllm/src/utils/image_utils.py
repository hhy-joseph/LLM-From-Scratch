"""
Utility functions for image processing.
"""

import base64
from PIL import Image
import io

def process_image(uploaded_file, max_size=(1024, 1024), quality=70):
    """
    Resizes and encodes the uploaded image to base64.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        max_size: Maximum dimensions for resizing
        quality: JPEG quality (0-100)
        
    Returns:
        Base64 encoded string of the processed image
    """
    try:
        img = Image.open(uploaded_file)

        # Convert to RGB if it's RGBA or P to avoid potential issues
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        img.thumbnail(max_size)  # Resize while maintaining aspect ratio
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)  # Save as JPEG for size efficiency
        encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded_string
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")