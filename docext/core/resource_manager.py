"""
Resource management utilities for DocExt.

This module provides centralized resource management including:
- Temporary file/directory cleanup
- Resource tracking and automatic cleanup
- Context managers for safe resource handling
"""

from __future__ import annotations

import os
import shutil
import tempfile
import atexit
from pathlib import Path
from typing import List, Set, Optional, Union
from contextlib import contextmanager
from loguru import logger

from .config import config


class ResourceManager:
    """Manages temporary files and directories with automatic cleanup."""
    
    def __init__(self):
        """Initialize the resource manager."""
        self._tracked_resources: Set[Path] = set()
        self._cleanup_enabled = config.get('CLEANUP_TEMP_FILES', True)
        
        # Register cleanup function to run on exit
        atexit.register(self.cleanup_all)
        
        logger.debug("Resource manager initialized")
    
    def create_temp_directory(self, prefix: str = "docext_", suffix: str = "") -> Path:
        """
        Create a temporary directory and track it for cleanup.
        
        Args:
            prefix: Prefix for the directory name
            suffix: Suffix for the directory name
            
        Returns:
            Path: Path to the created temporary directory
        """
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix, suffix=suffix))
        self._tracked_resources.add(temp_dir)
        logger.debug(f"Created temporary directory: {temp_dir}")
        return temp_dir
    
    def create_temp_file(
        self, 
        suffix: str = "", 
        prefix: str = "docext_", 
        dir: Optional[Union[str, Path]] = None,
        delete: bool = False
    ) -> Path:
        """
        Create a temporary file and track it for cleanup.
        
        Args:
            suffix: File suffix/extension
            prefix: File prefix
            dir: Directory to create the file in
            delete: Whether to auto-delete (if False, will be tracked for manual cleanup)
            
        Returns:
            Path: Path to the created temporary file
        """
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
        os.close(fd)  # Close the file descriptor
        
        temp_file = Path(temp_path)
        if not delete:
            self._tracked_resources.add(temp_file)
        
        logger.debug(f"Created temporary file: {temp_file}")
        return temp_file
    
    def track_resource(self, resource_path: Union[str, Path]) -> Path:
        """
        Track an existing resource for cleanup.
        
        Args:
            resource_path: Path to the resource to track
            
        Returns:
            Path: Path object of the tracked resource
        """
        path = Path(resource_path)
        self._tracked_resources.add(path)
        logger.debug(f"Tracking resource: {path}")
        return path
    
    def untrack_resource(self, resource_path: Union[str, Path]) -> bool:
        """
        Stop tracking a resource (without deleting it).
        
        Args:
            resource_path: Path to the resource to untrack
            
        Returns:
            bool: True if resource was being tracked, False otherwise
        """
        path = Path(resource_path)
        if path in self._tracked_resources:
            self._tracked_resources.remove(path)
            logger.debug(f"Untracked resource: {path}")
            return True
        return False
    
    def cleanup_resource(self, resource_path: Union[str, Path]) -> bool:
        """
        Clean up a specific resource.
        
        Args:
            resource_path: Path to the resource to clean up
            
        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        path = Path(resource_path)
        
        try:
            if path.exists():
                if path.is_file():
                    path.unlink()
                    logger.debug(f"Deleted file: {path}")
                elif path.is_dir():
                    shutil.rmtree(path)
                    logger.debug(f"Deleted directory: {path}")
                else:
                    logger.warning(f"Unknown resource type: {path}")
                    return False
            
            # Remove from tracking
            self._tracked_resources.discard(path)
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup resource {path}: {e}")
            return False
    
    def cleanup_all(self) -> int:
        """
        Clean up all tracked resources.
        
        Returns:
            int: Number of resources successfully cleaned up
        """
        if not self._cleanup_enabled:
            logger.debug("Resource cleanup is disabled")
            return 0
        
        cleaned_count = 0
        resources_to_clean = list(self._tracked_resources)  # Copy to avoid modification during iteration
        
        for resource in resources_to_clean:
            if self.cleanup_resource(resource):
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} resources")
        return cleaned_count
    
    def list_tracked_resources(self) -> List[Path]:
        """
        Get a list of all tracked resources.
        
        Returns:
            List[Path]: List of tracked resource paths
        """
        return list(self._tracked_resources)
    
    def enable_cleanup(self):
        """Enable automatic cleanup."""
        self._cleanup_enabled = True
        logger.debug("Resource cleanup enabled")
    
    def disable_cleanup(self):
        """Disable automatic cleanup."""
        self._cleanup_enabled = False
        logger.debug("Resource cleanup disabled")


# Global resource manager instance
resource_manager = ResourceManager()


@contextmanager
def temp_directory(prefix: str = "docext_", suffix: str = "", cleanup: bool = True):
    """
    Context manager for temporary directory with automatic cleanup.
    
    Args:
        prefix: Directory name prefix
        suffix: Directory name suffix
        cleanup: Whether to cleanup on exit
        
    Yields:
        Path: Path to the temporary directory
    """
    temp_dir = resource_manager.create_temp_directory(prefix=prefix, suffix=suffix)
    
    try:
        yield temp_dir
    finally:
        if cleanup:
            resource_manager.cleanup_resource(temp_dir)


@contextmanager
def temp_file(suffix: str = "", prefix: str = "docext_", cleanup: bool = True):
    """
    Context manager for temporary file with automatic cleanup.
    
    Args:
        suffix: File suffix/extension
        prefix: File name prefix
        cleanup: Whether to cleanup on exit
        
    Yields:
        Path: Path to the temporary file
    """
    temp_file_path = resource_manager.create_temp_file(suffix=suffix, prefix=prefix)
    
    try:
        yield temp_file_path
    finally:
        if cleanup:
            resource_manager.cleanup_resource(temp_file_path)


def get_temp_dir() -> Path:
    """
    Get the configured temporary directory.
    
    Returns:
        Path: Path to the temporary directory
    """
    temp_dir = Path(config.get('TEMP_DIR', tempfile.gettempdir()))
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def cleanup_old_temp_files(max_age_hours: int = 24) -> int:
    """
    Clean up old temporary files created by DocExt.
    
    Args:
        max_age_hours: Maximum age in hours for files to keep
        
    Returns:
        int: Number of files cleaned up
    """
    import time
    
    temp_dir = get_temp_dir()
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_count = 0
    
    try:
        for item in temp_dir.iterdir():
            if item.name.startswith("docext_"):
                item_age = current_time - item.stat().st_mtime
                if item_age > max_age_seconds:
                    try:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                        cleaned_count += 1
                        logger.debug(f"Cleaned up old temp resource: {item}")
                    except Exception as e:
                        logger.error(f"Failed to cleanup old temp resource {item}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old temporary files")
            
    except Exception as e:
        logger.error(f"Error during old temp file cleanup: {e}")
    
    return cleaned_count
