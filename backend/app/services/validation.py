"""
Data validation service for quality checks.
"""
from typing import Dict, Any, List, Optional
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)


class DataValidationService:
    """Service for data quality validation."""
    
    def __init__(self):
        self.logger = logger
    
    def validate_schema(
        self,
        data: pd.DataFrame,
        expected_columns: Optional[List[str]] = None,
        required_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate data schema (columns).
        
        Args:
            data: DataFrame to validate
            expected_columns: Expected column names
            required_columns: Required column names
            
        Returns:
            Validation result dictionary
        """
        issues = []
        warnings = []
        
        # Check required columns
        if required_columns:
            missing_required = set(required_columns) - set(data.columns)
            if missing_required:
                issues.append(f"Missing required columns: {', '.join(missing_required)}")
        
        # Check expected columns
        if expected_columns:
            unexpected = set(data.columns) - set(expected_columns)
            if unexpected:
                warnings.append(f"Unexpected columns: {', '.join(unexpected)}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "columns_found": list(data.columns),
            "columns_expected": expected_columns or [],
            "columns_required": required_columns or []
        }
    
    def check_completeness(
        self,
        data: pd.DataFrame,
        threshold: float = 0.95
    ) -> Dict[str, Any]:
        """
        Check data completeness (non-null values).
        
        Args:
            data: DataFrame to check
            threshold: Minimum completeness threshold (0-1)
            
        Returns:
            Completeness check result
        """
        total_cells = data.size
        null_cells = data.isnull().sum().sum()
        completeness = 1 - (null_cells / total_cells) if total_cells > 0 else 0
        
        column_completeness = {}
        for col in data.columns:
            col_completeness = 1 - (data[col].isnull().sum() / len(data))
            column_completeness[col] = col_completeness
        
        issues = []
        if completeness < threshold:
            issues.append(
                f"Overall completeness ({completeness:.2%}) below threshold ({threshold:.2%})"
            )
        
        # Check individual columns
        low_completeness_cols = [
            col for col, comp in column_completeness.items()
            if comp < threshold
        ]
        if low_completeness_cols:
            issues.append(
                f"Columns with low completeness: {', '.join(low_completeness_cols)}"
            )
        
        return {
            "overall_completeness": completeness,
            "threshold": threshold,
            "meets_threshold": completeness >= threshold,
            "column_completeness": column_completeness,
            "issues": issues,
            "null_count": null_cells,
            "total_cells": total_cells
        }
    
    def check_consistency(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Check data consistency (duplicates, data types, etc.).
        
        Args:
            data: DataFrame to check
            
        Returns:
            Consistency check result
        """
        issues = []
        warnings = []
        
        # Check for duplicates
        duplicate_rows = data.duplicated().sum()
        if duplicate_rows > 0:
            warnings.append(f"Found {duplicate_rows} duplicate rows")
        
        # Check data types consistency
        type_issues = []
        for col in data.columns:
            # Check if numeric columns have non-numeric values (after removing nulls)
            if pd.api.types.is_numeric_dtype(data[col]):
                non_numeric = pd.to_numeric(data[col], errors='coerce').isnull().sum() - data[col].isnull().sum()
                if non_numeric > 0:
                    type_issues.append(f"Column '{col}' has {non_numeric} non-numeric values")
        
        if type_issues:
            issues.extend(type_issues)
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "duplicate_rows": int(duplicate_rows),
            "total_rows": len(data)
        }
    
    def check_validity(
        self,
        data: pd.DataFrame,
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check data validity based on custom rules.
        
        Args:
            data: DataFrame to check
            validation_rules: Dictionary of validation rules per column
                Example: {
                    "age": {"min": 0, "max": 120},
                    "email": {"pattern": r"^[^@]+@[^@]+\.[^@]+$"}
                }
            
        Returns:
            Validity check result
        """
        if not validation_rules:
            return {
                "valid": True,
                "issues": [],
                "warnings": []
            }
        
        issues = []
        
        for col, rules in validation_rules.items():
            if col not in data.columns:
                continue
            
            # Check min/max for numeric columns
            if "min" in rules or "max" in rules:
                if pd.api.types.is_numeric_dtype(data[col]):
                    if "min" in rules:
                        below_min = (data[col] < rules["min"]).sum()
                        if below_min > 0:
                            issues.append(
                                f"Column '{col}': {below_min} values below minimum ({rules['min']})"
                            )
                    if "max" in rules:
                        above_max = (data[col] > rules["max"]).sum()
                        if above_max > 0:
                            issues.append(
                                f"Column '{col}': {above_max} values above maximum ({rules['max']})"
                            )
            
            # Check pattern for string columns
            if "pattern" in rules:
                import re
                pattern = re.compile(rules["pattern"])
                non_matching = ~data[col].astype(str).str.match(pattern, na=False)
                non_matching_count = non_matching.sum()
                if non_matching_count > 0:
                    issues.append(
                        f"Column '{col}': {non_matching_count} values don't match pattern"
                    )
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": []
        }
    
    def comprehensive_validation(
        self,
        data: pd.DataFrame,
        expected_columns: Optional[List[str]] = None,
        required_columns: Optional[List[str]] = None,
        completeness_threshold: float = 0.95,
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data validation.
        
        Args:
            data: DataFrame to validate
            expected_columns: Expected column names
            required_columns: Required column names
            completeness_threshold: Minimum completeness threshold
            validation_rules: Custom validation rules
            
        Returns:
            Comprehensive validation result
        """
        schema_result = self.validate_schema(data, expected_columns, required_columns)
        completeness_result = self.check_completeness(data, completeness_threshold)
        consistency_result = self.check_consistency(data)
        validity_result = self.check_validity(data, validation_rules)
        
        all_valid = (
            schema_result["valid"] and
            completeness_result["meets_threshold"] and
            consistency_result["valid"] and
            validity_result["valid"]
        )
        
        all_issues = (
            schema_result["issues"] +
            completeness_result["issues"] +
            consistency_result["issues"] +
            validity_result["issues"]
        )
        
        all_warnings = (
            schema_result["warnings"] +
            consistency_result["warnings"] +
            validity_result["warnings"]
        )
        
        return {
            "valid": all_valid,
            "schema": schema_result,
            "completeness": completeness_result,
            "consistency": consistency_result,
            "validity": validity_result,
            "all_issues": all_issues,
            "all_warnings": all_warnings,
            "summary": {
                "total_rows": len(data),
                "total_columns": len(data.columns),
                "overall_completeness": completeness_result["overall_completeness"],
                "duplicate_rows": consistency_result["duplicate_rows"],
                "issue_count": len(all_issues),
                "warning_count": len(all_warnings)
            }
        }

