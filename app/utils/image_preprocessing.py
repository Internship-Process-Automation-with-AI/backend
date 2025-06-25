import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing to improve OCR accuracy."""
    
    def __init__(self):
        self.kernel = np.ones((1, 1), np.uint8)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply comprehensive preprocessing to improve OCR accuracy.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Convert to grayscale if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply preprocessing steps
            processed = self._apply_preprocessing_pipeline(gray)
            
            logger.info("Image preprocessing completed successfully")
            return processed
            
        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}")
            # Return original image if preprocessing fails
            return image
    
    def _apply_preprocessing_pipeline(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Apply a series of preprocessing steps to enhance OCR accuracy.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Preprocessed image
        """
        # 1. Noise reduction
        denoised = cv2.medianBlur(gray_image, 3)
        
        # 2. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # 4. Binarization using adaptive thresholding
        binary = cv2.adaptiveThreshold(
            cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 5. Remove small noise
        kernel = np.ones((1, 1), np.uint8)
        final = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return final
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Alternative preprocessing for low-quality images.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to PIL Image for enhancement
        pil_image = Image.fromarray(image)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        sharpened = enhancer.enhance(1.5)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(sharpened)
        contrasted = enhancer.enhance(1.3)
        
        # Convert back to numpy array
        return np.array(contrasted)
    
    def resize_image(self, image: np.ndarray, target_width: int = 2000) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            target_width: Target width in pixels
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        
        if width <= target_width:
            return image
        
        # Calculate new height maintaining aspect ratio
        aspect_ratio = width / height
        new_height = int(target_width / aspect_ratio)
        
        # Resize using INTER_CUBIC for better quality
        resized = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return resized
    
    def detect_skew(self, image: np.ndarray) -> float:
        """
        Detect and return the skew angle of the image.
        
        Args:
            image: Input image
            
        Returns:
            Skew angle in degrees
        """
        # Convert to binary if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Find the largest contour (assumed to be the main text area)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit a rectangle to the contour
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Normalize angle to -45 to 45 degrees
        if angle < -45:
            angle = 90 + angle
        
        return angle
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew the image based on detected text orientation.
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        angle = self.detect_skew(image)
        
        if abs(angle) < 0.5:  # Skip if angle is very small
            return image
        
        # Get image dimensions
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        deskewed = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return deskewed 