"""
Configuration management for DocExt.

This module provides centralized configuration management including:
- Environment variable handling
- Template definitions
- Default settings
- Configuration validation
"""

from __future__ import annotations

import os
from typing import Dict, List, Any, Optional
from loguru import logger


class ConfigManager:
    """Centralized configuration manager for DocExt."""
    
    def __init__(self):
        """Initialize configuration manager with environment variables."""
        self._config = {}
        self._load_environment_variables()
        self._validate_required_settings()
    
    def _load_environment_variables(self):
        """Load and validate environment variables."""
        # VLM Model configuration
        self._config['VLM_MODEL_URL'] = os.getenv('VLM_MODEL_URL')
        self._config['VLM_MODEL_API_KEY'] = os.getenv('VLM_MODEL_API_KEY')
        
        # API configuration
        self._config['API_BASE'] = os.getenv('API_BASE')
        self._config['API_KEY'] = os.getenv('API_KEY')
        
        # Model settings
        self._config['DEFAULT_MODEL'] = os.getenv('DEFAULT_MODEL', 'gpt-4o')
        self._config['MAX_IMAGE_SIZE'] = int(os.getenv('MAX_IMAGE_SIZE', '1024'))
        
        # Logging configuration
        self._config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'INFO')
        
        # Application settings
        self._config['TEMP_DIR'] = os.getenv('TEMP_DIR', os.path.join(os.getcwd(), 'temp'))
        self._config['CLEANUP_TEMP_FILES'] = os.getenv('CLEANUP_TEMP_FILES', 'true').lower() == 'true'
        
        logger.debug(f"Loaded configuration: {self._get_safe_config()}")
    
    def _validate_required_settings(self):
        """Validate that required configuration is present."""
        required_vars = ['VLM_MODEL_URL']
        missing_vars = []
        
        for var in required_vars:
            if not self._config.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.warning(f"Missing required environment variables: {missing_vars}")
    
    def _get_safe_config(self) -> Dict[str, Any]:
        """Get configuration dict with sensitive values masked."""
        safe_config = self._config.copy()
        sensitive_keys = ['VLM_MODEL_API_KEY', 'API_KEY']
        
        for key in sensitive_keys:
            if key in safe_config and safe_config[key]:
                safe_config[key] = '***masked***'
        
        return safe_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self._config[key] = value
    
    def get_required(self, key: str) -> Any:
        """Get required configuration value, raise error if missing."""
        value = self._config.get(key)
        if value is None:
            raise ValueError(f"Required configuration '{key}' is not set")
        return value
    
    def is_configured(self) -> bool:
        """Check if basic configuration is present."""
        return bool(self._config.get('VLM_MODEL_URL'))


# Global configuration instance
config = ConfigManager()


TEMPLATES_FIELDS = {
    "invoice colab demo 🧾": [
        {"field_name": "invoice_number", "description": "Invoice number"},
        {"field_name": "invoice_date", "description": "Invoice date"},
        {"field_name": "invoice_amount", "description": "Invoice amount"},
        {
            "field_name": "seller_name",
            "description": "Seller name. If not explicitly mentioned, return ''",
        },
    ],
    "invoice 🧾": [
        {"field_name": "invoice_number", "description": "Invoice number"},
        {"field_name": "invoice_date", "description": "Invoice date"},
        {"field_name": "invoice_amount", "description": "Invoice amount"},
        {
            "field_name": "invoice_currency",
            "description": "Invoice currency. If not explicitly mentioned, return ''",
        },
        {
            "field_name": "document_type",
            "description": "Document type. If not explicitly mentioned, return ''",
        },
        {
            "field_name": "seller_name",
            "description": "Seller name. If not explicitly mentioned, return ''",
        },
        {"field_name": "buyer_name", "description": "Buyer name"},
        {"field_name": "seller_address", "description": "Seller address"},
        {"field_name": "buyer_address", "description": "Buyer address"},
        {"field_name": "seller_tax_id", "description": "Seller tax id"},
        {"field_name": "buyer_tax_id", "description": "Buyer tax id"},
    ],
    "passport 🎫": [
        {"field_name": "full_name", "description": "Full name"},
        {
            "field_name": "date_of_birth",
            "description": "Date of birth. Return in format YYYY-MM-DD",
        },
        {"field_name": "passport_number", "description": "Passport number"},
        {"field_name": "passport_type", "description": "Passport type"},
        {
            "field_name": "date_of_issue",
            "description": "Date of issue. Return in format YYYY-MM-DD",
        },
        {
            "field_name": "date_of_expiry",
            "description": "Date of expiry. Return in format YYYY-MM-DD",
        },
        {"field_name": "place_of_birth", "description": "Place of birth"},
        {"field_name": "nationality", "description": "Nationality"},
        {"field_name": "gender", "description": "Gender"},
    ],
}

TEMPLATES_TABLES = {
    "invoice colab demo 🧾": [
        {
            "field_name": "items_description",
            "description": "Description of the product",
        },
        {"field_name": "Unit Price", "description": "Unit price of the product"},
    ],
    "invoice 🧾": [
        {"field_name": "Quantity", "description": "Total quantity of the product"},
        {
            "field_name": "items_description",
            "description": "Description of the product",
        },
        {"field_name": "Unit Price", "description": "Unit price of the product"},
        {"field_name": "Total Price", "description": "Total price of the product"},
        {"field_name": "tax", "description": "tax amount"},
    ],
}


def get_template_fields(template_name: str) -> List[Dict[str, str]]:
    """Get field template by name."""
    return TEMPLATES_FIELDS.get(template_name, [])


def get_template_tables(template_name: str) -> List[Dict[str, str]]:
    """Get table template by name."""
    return TEMPLATES_TABLES.get(template_name, [])


def list_available_templates() -> List[str]:
    """List all available template names."""
    field_templates = set(TEMPLATES_FIELDS.keys())
    table_templates = set(TEMPLATES_TABLES.keys())
    return sorted(field_templates.union(table_templates))
