from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger
from pdf2image import convert_from_path

from docext.core.file_converters.file_converter import FileConverter


class PDFConverter(FileConverter):
    """
    Converter for PDF files to images using pdf2image library.
    
    This class handles conversion of PDF documents to image files,
    with support for custom output directories and error handling.
    """
    
    def convert_to_images(self, file_path: str):
        """
        Convert PDF file to PIL Image objects.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of PIL Image objects, one per page
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF conversion fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        try:
            logger.debug(f"Converting PDF to images: {file_path}")
            images = convert_from_path(file_path)
            logger.debug(f"Successfully converted PDF to {len(images)} images")
            return images
        except Exception as e:
            logger.error(f"Failed to convert PDF {file_path}: {str(e)}")
            raise

    def convert_and_save_images(self, file_path: str, output_folder: str | None = None):
        """
        Convert PDF to images and save them to disk.
        
        Args:
            file_path: Path to the PDF file
            output_folder: Directory to save images. If None, uses system temp directory
            
        Returns:
            List of paths to saved image files
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            OSError: If output directory cannot be created
            Exception: If conversion or saving fails
        """
        images = self.convert_to_images(file_path)
        
        if not output_folder:
            # Create a unique subdirectory in temp folder to avoid conflicts
            pdf_name = Path(file_path).stem
            output_folder = os.path.join(tempfile.gettempdir(), f"docext_pdf_{pdf_name}")
            
        try:
            os.makedirs(output_folder, exist_ok=True)
            logger.debug(f"Saving {len(images)} images to: {output_folder}")
        except OSError as e:
            logger.error(f"Failed to create output directory {output_folder}: {str(e)}")
            raise
            
        output_file_paths = []
        
        for i, image in enumerate(images):
            try:
                output_file_path = os.path.join(output_folder, f"page_{i}.png")
                image.save(output_file_path, "PNG", optimize=True)
                output_file_paths.append(output_file_path)
                logger.debug(f"Saved page {i+1} to: {output_file_path}")
            except Exception as e:
                logger.error(f"Failed to save page {i+1}: {str(e)}")
                # Clean up partially saved files
                for saved_path in output_file_paths:
                    try:
                        os.remove(saved_path)
                    except OSError:
                        pass
                raise
                
        logger.info(f"Successfully converted PDF to {len(output_file_paths)} images in {output_folder}")
        return output_file_paths
