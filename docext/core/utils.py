"""
Utility functions for DocExt core functionality.

This module provides common utilities including:
- Image processing and validation
- File format conversion
- Input validation
- Configuration helpers
"""

from __future__ import annotations

import base64
import io
import os
from typing import Union, List, Dict, Any
from pathlib import Path

import pandas as pd
from PIL import Image
from loguru import logger

from docext.core.file_converters.pdf_converter import PDFConverter
from docext.core.resource_manager import resource_manager


def encode_image(image_path: Union[str, Path]) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Base64 encoded image string
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        Exception: If encoding fails
    """
    try:
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
            logger.debug(f"Encoded image: {image_path}")
            return encoded
            
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        raise


def validate_fields_and_tables(fields_and_tables: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, List[Dict]]:
    """
    Validate and normalize fields and tables configuration.
    
    Args:
        fields_and_tables: Configuration dict or DataFrame
        
    Returns:
        dict: Normalized configuration with 'fields' and 'tables' keys
        
    Raises:
        ValueError: If validation fails
    """
    try:
        # Convert DataFrame to dict if needed
        if isinstance(fields_and_tables, pd.DataFrame):
            if "index" in fields_and_tables.columns:
                fields_and_tables.drop(columns=["index"], inplace=True)
            
            fields_df = fields_and_tables[fields_and_tables.type == "field"]
            tables_df = fields_and_tables[fields_and_tables.type == "table"]

            fields_and_tables = {
                "fields": fields_df.to_dict(orient="records"),
                "tables": tables_df.to_dict(orient="records"),
            }

        # Validate structure
        if not isinstance(fields_and_tables, dict):
            raise ValueError("fields_and_tables must be a dict or DataFrame")
        
        if "fields" not in fields_and_tables:
            raise ValueError("'fields' key must be present in configuration")
        
        if "tables" not in fields_and_tables:
            raise ValueError("'tables' key must be present in configuration")

        # Validate field entries
        for i, field_details in enumerate(fields_and_tables["fields"]):
            if not isinstance(field_details, dict):
                raise ValueError(f"Field {i} must be a dictionary")
            if "name" not in field_details:
                raise ValueError(f"Field {i} must have a 'name' key")

        # Validate table entries
        for i, table_details in enumerate(fields_and_tables["tables"]):
            if not isinstance(table_details, dict):
                raise ValueError(f"Table {i} must be a dictionary")
            if "name" not in table_details:
                raise ValueError(f"Table {i} must have a 'name' key")

        logger.debug(f"Validated configuration: {len(fields_and_tables['fields'])} fields, {len(fields_and_tables['tables'])} tables")
        return fields_and_tables
        
    except Exception as e:
        logger.error(f"Failed to validate fields and tables configuration: {e}")
        raise ValueError(f"Invalid configuration: {str(e)}")


def resize_images(file_paths: List[str], max_img_size: int) -> None:
    """
    Resize images to maximum size while maintaining aspect ratio.
    
    Args:
        file_paths: List of image file paths
        max_img_size: Maximum size for the larger dimension
        
    Raises:
        Exception: If image processing fails
    """
    for file_path in file_paths:
        try:
            img = Image.open(file_path)
            original_size = img.size
            
            # Calculate new size maintaining aspect ratio
            if max(original_size) > max_img_size:
                ratio = max_img_size / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                img.save(file_path, optimize=True, quality=95)
                logger.debug(f"Resized image {file_path}: {original_size} -> {new_size}")
            else:
                logger.debug(f"Image {file_path} already within size limit: {original_size}")
                
        except Exception as e:
            logger.error(f"Failed to resize image {file_path}: {e}")
            raise


def validate_file_paths(file_paths: List[str]) -> None:
    """
    Validate that file paths exist and are supported formats.
    
    Args:
        file_paths: List of file paths to validate
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    supported_extensions = {
        ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif", ".webp", ".pdf"
    }
    
    for file_path in file_paths:
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        extension = path.suffix.lower()
        if extension not in supported_extensions:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {', '.join(sorted(supported_extensions))}"
            )
    
    logger.debug(f"Validated {len(file_paths)} file paths")


def file_is_supported_image(file_path: str) -> bool:
    """
    Check if file is a supported image format.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if file is a supported image format
    """
    supported_image_extensions = {
        ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif", ".webp"
    }
    extension = Path(file_path).suffix.lower()
    return extension in supported_image_extensions


def convert_files_to_images(file_paths: List[str]) -> List[str]:
    """
    Convert files to images, handling PDF conversion.
    
    Args:
        file_paths: List of file paths to convert
        
    Returns:
        List[str]: List of converted image file paths
        
    Raises:
        Exception: If conversion fails
    """
    converted_file_paths = []
    pdf_converter = PDFConverter()
    
    for file_path in file_paths:
        try:
            path = Path(file_path)
            
            if path.suffix.lower() == ".pdf":
                logger.debug(f"Converting PDF to images: {file_path}")
                images = pdf_converter.convert_to_images(file_path)
                
                for i, image in enumerate(images):
                    output_path = f"{file_path.replace('.pdf', '')}_{i}.jpg"
                    image.save(output_path, optimize=True, quality=95)
                    converted_file_paths.append(output_path)
                    
                    # Track the converted image for cleanup
                    resource_manager.track_resource(output_path)
                
                logger.info(f"Converted PDF to {len(images)} images")
                
            elif file_is_supported_image(file_path):
                converted_file_paths.append(file_path)
            else:
                logger.warning(f"Skipping unsupported file: {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to convert file {file_path}: {e}")
            raise
    
    logger.debug(f"Converted {len(file_paths)} files to {len(converted_file_paths)} images")
    return converted_file_paths


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get detailed information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        dict: File information including size, type, etc.
    """
    path = Path(file_path)
    
    if not path.exists():
        return {"exists": False, "path": str(path)}
    
    stat = path.stat()
    
    info = {
        "exists": True,
        "path": str(path),
        "name": path.name,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "extension": path.suffix.lower(),
        "is_image": file_is_supported_image(str(path)),
        "is_pdf": path.suffix.lower() == ".pdf",
        "modified_time": stat.st_mtime,
    }
    
    # Add image-specific info if it's an image
    if info["is_image"]:
        try:
            with Image.open(path) as img:
                info["image_size"] = img.size
                info["image_mode"] = img.mode
                info["image_format"] = img.format
        except Exception:
            pass
    
    return info
