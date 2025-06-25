import cv2
import numpy as np
from PIL import Image, ImageEnhance
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
            
            # Try multiple preprocessing strategies and return the best one
            strategies = [
                self._strategy_adaptive_threshold,
                self._strategy_otsu_threshold,
                self._strategy_enhanced_contrast,
                self._strategy_morphological_cleanup
            ]
            
            best_result = None
            best_score = 0
            
            for strategy in strategies:
                try:
                    processed = strategy(gray)
                    # Simple quality score based on edge density
                    score = self._calculate_image_quality(processed)
                    if score > best_score:
                        best_score = score
                        best_result = processed
                except Exception as e:
                    logger.debug(f"Preprocessing strategy failed: {e}")
                    continue
            
            # If all strategies fail, use the original
            if best_result is None:
                best_result = gray
            
            logger.info("Image preprocessing completed successfully")
            return best_result
            
        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}")
            # Return original image if preprocessing fails
            return image
    
    def _strategy_adaptive_threshold(self, gray_image: np.ndarray) -> np.ndarray:
        """Strategy 1: Adaptive thresholding for varying lighting conditions."""
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_image)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )
        
        return binary
    
    def _strategy_otsu_threshold(self, gray_image: np.ndarray) -> np.ndarray:
        """Strategy 2: Otsu thresholding for bimodal histograms."""
        # Denoise
        denoised = cv2.medianBlur(gray_image, 3)
        
        # Apply Otsu thresholding
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _strategy_enhanced_contrast(self, gray_image: np.ndarray) -> np.ndarray:
        """Strategy 3: Enhanced contrast and sharpening."""
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_image)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Apply threshold
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _strategy_morphological_cleanup(self, gray_image: np.ndarray) -> np.ndarray:
        """Strategy 4: Morphological operations for cleanup."""
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_image)
        
        # Apply threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        kernel = np.ones((1, 1), np.uint8)
        final = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return final
    
    def _calculate_image_quality(self, image: np.ndarray) -> float:
        """Calculate a simple quality score for the image."""
        # Calculate edge density as a quality measure
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        return edge_density
    
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