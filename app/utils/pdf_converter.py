from typing import List,  Tuple
import fitz  # PyMuPDF
from PIL import Image
import io
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PDFConverter:
    """Handles conversion of PDF files to text and images using PyMuPDF."""
    
    def __init__(self):
        """Initialize PDF converter with PyMuPDF."""
        logger.info("PDF Converter initialized with PyMuPDF")
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extract text directly from PDF using PyMuPDF.
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Extracted text as string
        """
        try:
            logger.info("Extracting text from PDF using PyMuPDF")
            
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text.strip():
                    text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
            
            doc.close()
            
            logger.info(f"Successfully extracted text from {len(doc)} pages")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def extract_text_from_pdf_file(self, pdf_path: str) -> str:
        """
        Extract text from PDF file on disk.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            logger.info(f"Extracting text from PDF file: {pdf_path}")
            
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            return self.extract_text_from_pdf(pdf_bytes)
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF file: {e}")
            raise
    
    def convert_pdf_to_images(self, pdf_bytes: bytes, dpi: int = 300) -> List[np.ndarray]:
        """
        Convert PDF to list of numpy arrays (images) for OCR processing.
        
        Args:
            pdf_bytes: PDF file as bytes
            dpi: Resolution for conversion (default: 300)
            
        Returns:
            List of numpy arrays representing images
        """
        try:
            logger.info("Converting PDF to images using PyMuPDF")
            
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []
            
            # Calculate zoom factor for desired DPI
            zoom = dpi / 72.0  # PyMuPDF uses 72 DPI as base
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Create transformation matrix for desired resolution
                mat = fitz.Matrix(zoom, zoom)
                
                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Convert to numpy array
                img_array = np.array(pil_image)
                
                # Convert RGBA to RGB if needed
                if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]
                
                images.append(img_array)
                
                logger.info(f"Converted page {page_num + 1} to image")
            
            doc.close()
            
            logger.info(f"Successfully converted PDF to {len(images)} images")
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise
    
    def convert_pdf_file_to_images(self, pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        """
        Convert PDF file on disk to list of numpy arrays.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for conversion (default: 300)
            
        Returns:
            List of numpy arrays representing images
        """
        try:
            logger.info(f"Converting PDF file to images: {pdf_path}")
            
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            return self.convert_pdf_to_images(pdf_bytes, dpi)
            
        except Exception as e:
            logger.error(f"Error converting PDF file to images: {e}")
            raise
    
    def get_pdf_page_count(self, pdf_bytes: bytes) -> int:
        """
        Get the number of pages in a PDF.
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Number of pages
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = len(doc)
            doc.close()
            return page_count
            
        except Exception as e:
            logger.error(f"Error getting PDF page count: {e}")
            raise
    
    def get_pdf_page_count_from_file(self, pdf_path: str) -> int:
        """
        Get the number of pages in a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of pages
        """
        try:
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            return self.get_pdf_page_count(pdf_bytes)
            
        except Exception as e:
            logger.error(f"Error getting PDF page count from file: {e}")
            raise
    
    def extract_text_with_quality_check(self, pdf_bytes: bytes) -> Tuple[str, float]:
        """
        Extract text from PDF and assess quality.
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Tuple of (extracted_text, quality_score)
        """
        doc = None
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            total_chars = 0
            pages_with_text = 0
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                if page_text.strip():
                    text += page_text + "\n"
                    total_chars += len(page_text.strip())
                    pages_with_text += 1
            
            # Calculate quality score based on text density
            quality_score = min(100.0, (total_chars / max(1, len(doc))) / 10.0)
            
            return text.strip(), quality_score
            
        except Exception as e:
            logger.error(f"Error in text quality check: {e}")
            raise
        finally:
            if doc:
                try:
                    doc.close()
                except Exception:
                    pass
    
    def is_pdf_text_based(self, pdf_bytes: bytes) -> bool:
        """
        Check if PDF contains extractable text (not just images).
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            True if PDF contains text, False if image-only
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(min(3, len(doc))):  # Check first 3 pages
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                if len(page_text.strip()) > 50:  # If we find substantial text
                    doc.close()
                    return True
            
            doc.close()
            return False
            
        except Exception as e:
            logger.error(f"Error checking PDF text content: {e}")
            return False
    
    def get_pdf_metadata(self, pdf_bytes: bytes) -> dict:
        """
        Extract PDF metadata.
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Dictionary containing PDF metadata
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            metadata = {
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", "")
            }
            
            doc.close()
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {e}")
            return {} 