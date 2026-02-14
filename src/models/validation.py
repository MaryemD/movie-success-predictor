"""
Input validation utilities for movie prediction API.
Provides error handling and data validation for prediction requests.
"""

from pydantic import BaseModel, validator, ValidationError
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates input data for prediction requests."""
    
    @staticmethod
    def validate_movie_record(record: Dict[str, Any], required_features: List[str]) -> Dict[str, Any]:
        """
        Validate a single movie record against required features.
        
        Args:
            record: Dictionary containing movie features
            required_features: List of feature names that must be present
            
        Returns:
            Validated record
            
        Raises:
            ValueError: If validation fails
        """
        missing = [f for f in required_features if f not in record]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Check for numeric values
        for feature in required_features:
            try:
                val = record[feature]
                if val is None:
                    raise ValueError(f"Feature '{feature}' cannot be None")
                # Try to convert to float to verify it's numeric
                float(val)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Feature '{feature}' must be numeric, got {val}")
        
        return record
    
    @staticmethod
    def validate_batch(
        records: List[Dict[str, Any]], 
        required_features: List[str],
        max_records: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Validate a batch of movie records.
        
        Args:
            records: List of movie record dictionaries
            required_features: Required feature names
            max_records: Maximum records allowed per batch
            
        Returns:
            Validated records
            
        Raises:
            ValueError: If batch validation fails
        """
        if not records:
            raise ValueError("Records list cannot be empty")
        
        if len(records) > max_records:
            raise ValueError(f"Batch size {len(records)} exceeds maximum of {max_records}")
        
        validated = []
        errors = []
        
        for i, record in enumerate(records):
            try:
                validated_record = InputValidator.validate_movie_record(record, required_features)
                validated.append(validated_record)
            except ValueError as e:
                errors.append(f"Record {i}: {str(e)}")
        
        if errors:
            error_msg = "\n".join(errors[:5])  # Show first 5 errors
            if len(errors) > 5:
                error_msg += f"\n... and {len(errors) - 5} more errors"
            raise ValueError(f"Batch validation failed:\n{error_msg}")
        
        return validated
    
    @staticmethod
    def validate_dataset_choice(dataset: str, available: List[str]) -> str:
        """
        Validate dataset choice.
        
        Args:
            dataset: Dataset name
            available: List of available datasets
            
        Returns:
            Validated dataset name
            
        Raises:
            ValueError: If dataset is not available
        """
        if dataset not in available:
            raise ValueError(
                f"Invalid dataset '{dataset}'. Available options: {', '.join(available)}"
            )
        return dataset


class DataTypeValidator:
    """Validates and coerces data types."""
    
    @staticmethod
    def coerce_numeric(value: Any, field_name: str, allow_none: bool = False) -> float:
        """
        Coerce value to float, with error handling.
        
        Args:
            value: Value to coerce
            field_name: Name of field (for error messages)
            allow_none: Whether None is acceptable
            
        Returns:
            Coerced numeric value
            
        Raises:
            ValueError: If coercion fails
        """
        if value is None:
            if allow_none:
                return None
            raise ValueError(f"Field '{field_name}' cannot be None")
        
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ValueError(
                f"Field '{field_name}' must be numeric, got {value} ({type(value).__name__})"
            )
    
    @staticmethod
    def validate_range(
        value: float, 
        field_name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> float:
        """
        Validate value is within specified range.
        
        Args:
            value: Numeric value
            field_name: Name of field
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Validated value
            
        Raises:
            ValueError: If value is out of range
        """
        if min_val is not None and value < min_val:
            raise ValueError(f"Field '{field_name}' value {value} is below minimum {min_val}")
        
        if max_val is not None and value > max_val:
            raise ValueError(f"Field '{field_name}' value {value} exceeds maximum {max_val}")
        
        return value
