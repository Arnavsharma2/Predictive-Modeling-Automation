"""
Model explainability API endpoints.

Provides endpoints for SHAP, LIME, and feature importance explanations.
"""
import tempfile
import os
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.database.ml_models import MLModel, ModelStatus, ModelType
from app.models.database.data_points import DataPoint
from app.models.schemas.explainability import (
    ShapExplanationRequest,
    ShapExplanationResponse,
    ShapWaterfallData,
    LimeExplanationRequest,
    LimeExplanationResponse,
    FeatureImportanceRequest,
    FeatureImportanceResponse,
    ComprehensiveImportanceResponse,
    ExplanationSummaryRequest,
    ExplanationSummaryResponse,
)
from app.ml.explainability.shap_explainer import ShapExplainer
from app.ml.explainability.lime_explainer import LimeExplainer
from app.ml.explainability.feature_importance import FeatureImportanceAnalyzer
from app.ml.storage.model_storage import model_storage
from app.services.training_job import TrainingJobService

logger = get_logger(__name__)

router = APIRouter(prefix="/explain", tags=["Explainability"])


def _get_feature_name_mapping(model_record: MLModel, model_package: dict) -> dict:
    """
    Get feature name mapping from model record or package metadata.
    
    Args:
        model_record: Database model record
        model_package: Model package dictionary
        
    Returns:
        Dictionary mapping technical feature names to readable names
    """
    # Try to get from model package metadata first
    if isinstance(model_package, dict):
        metadata = model_package.get("metadata", {})
        if metadata and "feature_name_mapping" in metadata:
            mapping = metadata["feature_name_mapping"]
            logger.debug(f"Retrieved feature_name_mapping from model package metadata ({len(mapping)} mappings)")
            return mapping
    
    # Fall back to database record
    if model_record.feature_name_mapping:
        logger.debug(f"Retrieved feature_name_mapping from database record ({len(model_record.feature_name_mapping)} mappings)")
        return model_record.feature_name_mapping
    
    # If no mapping available, return empty dict (will use technical names as-is)
    logger.warning(f"No feature_name_mapping found for model {model_record.id}. Using technical names as-is.")
    return {}


def _get_original_column_mapping(model_record: MLModel, model_package: dict) -> dict:
    """
    Get original column mapping from model record or package metadata.
    This maps technical feature names directly to original column names from the dataset.
    
    Args:
        model_record: Database model record
        model_package: Model package dictionary
        
    Returns:
        Dictionary mapping technical feature names to original column names
    """
    # Try to get from model package metadata first
    if isinstance(model_package, dict):
        metadata = model_package.get("metadata", {})
        if metadata and "original_column_mapping" in metadata:
            mapping = metadata["original_column_mapping"]
            # Filter out None values and return only valid mappings
            mapping = {k: v for k, v in mapping.items() if v is not None}
            logger.debug(f"Retrieved original_column_mapping from model package metadata ({len(mapping)} mappings)")
            return mapping
    
    # If no mapping available, return empty dict
    logger.debug(f"No original_column_mapping found for model {model_record.id}.")
    return {}


def _apply_feature_name_mapping(
    importance_dict: dict,
    feature_name_mapping: dict,
    original_column_mapping: Optional[dict] = None
) -> dict:
    """
    Apply feature name mapping to importance dictionary.
    Prioritizes original column names when available, falls back to readable names.
    
    Args:
        importance_dict: Dictionary mapping technical feature names to importance scores
        feature_name_mapping: Dictionary mapping technical names to readable names
        original_column_mapping: Dictionary mapping technical names to original column names (optional)
        
    Returns:
        Dictionary with original column names (or readable names if original not available)
    """
    if not feature_name_mapping and not original_column_mapping:
        logger.debug("No feature_name_mapping or original_column_mapping provided, returning importance_dict as-is")
        return importance_dict
    
    mapped_dict = {}
    unmapped_count = 0
    for technical_name, importance_value in importance_dict.items():
        # Prioritize original column name if available
        display_name = None
        
        if original_column_mapping:
            original_col = original_column_mapping.get(technical_name)
            if original_col:
                display_name = original_col
        
        # Fall back to readable name if original column not available
        if not display_name and feature_name_mapping:
            display_name = feature_name_mapping.get(technical_name)
        
        # Final fallback to technical name
        if not display_name:
            display_name = technical_name
            unmapped_count += 1
        
        # Aggregate importance if multiple technical features map to same original column
        if display_name in mapped_dict:
            # If multiple features map to same original column, sum their importance
            mapped_dict[display_name] += importance_value
        else:
            mapped_dict[display_name] = importance_value
    
    if unmapped_count > 0:
        logger.warning(f"{unmapped_count} out of {len(importance_dict)} features were not found in mapping")
        # Log a few examples of unmapped features
        unmapped_examples = [
            tech_name for tech_name in importance_dict.keys()
            if tech_name not in (original_column_mapping or {}) and tech_name not in (feature_name_mapping or {})
        ][:5]
        logger.debug(f"Unmapped feature examples: {unmapped_examples}")
    
    return mapped_dict


def _apply_feature_name_mapping_to_list(
    feature_list: List[str],
    feature_name_mapping: dict,
    original_column_mapping: Optional[dict] = None
) -> List[str]:
    """
    Apply feature name mapping to a list of feature names.
    Prioritizes original column names when available, falls back to readable names.
    
    Args:
        feature_list: List of technical feature names
        feature_name_mapping: Dictionary mapping technical names to readable names
        original_column_mapping: Dictionary mapping technical names to original column names (optional)
        
    Returns:
        List of original column names (or readable names if original not available)
    """
    if not feature_name_mapping and not original_column_mapping:
        return feature_list
    
    result = []
    for technical_name in feature_list:
        # Prioritize original column name if available
        display_name = None
        
        if original_column_mapping:
            original_col = original_column_mapping.get(technical_name)
            if original_col:
                display_name = original_col
        
        # Fall back to readable name if original column not available
        if not display_name and feature_name_mapping:
            display_name = feature_name_mapping.get(technical_name)
        
        # Final fallback to technical name
        if not display_name:
            display_name = technical_name
        
        result.append(display_name)
    
    return result


async def _get_model_and_package(
    db: AsyncSession,
    model_id: int
) -> tuple[MLModel, dict]:
    """
    Get model record and load model package from storage.
    
    Args:
        db: Database session
        model_id: Model ID
        
    Returns:
        Tuple of (model_record, model_package)
    """
    model_record = await db.get(MLModel, model_id)
    
    if not model_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    if model_record.status not in [ModelStatus.TRAINED, ModelStatus.DEPLOYED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {model_id} is not ready for explanations (status: {model_record.status})"
        )
    
    if not model_record.model_path:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model path not found"
        )
    
    # Load model package
    model_package = await model_storage.load_model(model_record.model_path, return_package=True)
    
    if not isinstance(model_package, dict):
        # Legacy format - wrap in package format
        # Include metadata from database record if available
        metadata = {}
        if model_record.feature_name_mapping:
            metadata["feature_name_mapping"] = model_record.feature_name_mapping
        if model_record.original_columns:
            metadata["original_columns"] = model_record.original_columns
        if model_record.hyperparameters:
            metadata["hyperparameters"] = model_record.hyperparameters
        
        model_package = {
            "model": model_package,
            "preprocessor": None,
            "feature_names": model_record.features,
            "metadata": metadata if metadata else {}
        }
    
    return model_record, model_package


async def _save_model_temp(model_package: dict) -> str:
    """
    Save model to temporary file for explainers that need file path.
    
    Args:
        model_package: Model package dictionary
        
    Returns:
        Path to temporary model file
    """
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.joblib')
    temp_path = temp_file.name
    temp_file.close()
    
    # Save model to temp file (run in thread pool to avoid blocking event loop)
    import joblib
    import asyncio
    await asyncio.to_thread(joblib.dump, model_package["model"], temp_path)
    
    return temp_path


async def _transform_features(
    features: Dict[str, Any],
    model_record: MLModel,
    model_package: dict
) -> np.ndarray:
    """
    Transform original features to preprocessed format for model input.
    
    Args:
        features: Dictionary of original feature values
        model_record: Model record
        model_package: Model package with preprocessor
        
    Returns:
        Numpy array of preprocessed features
    """
    # Prepare features from user input (in original column format)
    features_df = pd.DataFrame([features])
    
    preprocessor = model_package.get("preprocessor")
    
    # If preprocessor is available, user should provide original columns
    if preprocessor:
        # Check that user provided original columns
        if model_record.original_columns:
            missing_cols = [col for col in model_record.original_columns if col not in features_df.columns]
            if missing_cols:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required columns: {missing_cols}. Please provide values for original dataset columns."
                )
            
            # Reorder to match training order
            features_df = features_df[model_record.original_columns]
        
        # Infer and convert data types (numeric columns to float, categorical stay as object/string)
        for col in features_df.columns:
            try:
                features_df[col] = pd.to_numeric(features_df[col], errors='ignore')
            except (ValueError, TypeError):
                pass  # Keep as string/object for categorical columns
        
        # Apply preprocessing transformation
        try:
            features_df = preprocessor.transform(features_df)
        except Exception as e:
            logger.error(f"Error applying preprocessing: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error preprocessing features: {str(e)}"
            )
    else:
        # Legacy: No preprocessor, user must provide transformed features
        if model_record.features:
            missing_features = [f for f in model_record.features if f not in features_df.columns]
            if missing_features:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required features: {missing_features}"
                )
            
            features_df = features_df[model_record.features]
    
    return features_df.values


def _map_explanation_to_original_features(
    explanation: Dict[str, Any],
    model_record: MLModel,
    model_package: dict
) -> Dict[str, Any]:
    """
    Map explanation from technical feature names back to original feature names.
    
    Args:
        explanation: Explanation dictionary with technical feature names
        model_record: Model record
        model_package: Model package
        
    Returns:
        Explanation dictionary with original feature names
    """
    # Get feature name mapping and original column mapping - use helper functions for consistency
    feature_name_mapping = _get_feature_name_mapping(model_record, model_package)
    original_column_mapping = _get_original_column_mapping(model_record, model_package)

    # Log mapping info for debugging
    logger.debug(f"Feature name mapping: {len(feature_name_mapping)} mappings available")
    logger.debug(f"Original column mapping: {len(original_column_mapping)} mappings available")
    if original_column_mapping:
        sample_orig = list(original_column_mapping.items())[:3]
        logger.debug(f"Sample original column mappings: {sample_orig}")

    # If we have original_columns, use them as fallback
    original_columns = model_record.original_columns or []
    
    # Map feature contributions
    mapped_contributions = []
    feature_aggregation = {}  # Aggregate contributions by original feature
    
    for contrib in explanation.get('feature_contributions', []):
        technical_name = contrib.get('feature', contrib.get('feature_name', ''))
        value = contrib.get('shap_value', contrib.get('value', contrib.get('weight', 0)))
        
        # Find original column - prioritize original_column_mapping
        original_col = None
        if original_column_mapping:
            original_col = original_column_mapping.get(technical_name)
        
        # Fallback: check if technical name is already an original column
        if not original_col and technical_name in original_columns:
            original_col = technical_name
        
        # Additional fallback: try to extract original column from readable name
        # If technical_name looks like "Option 0" or "Category: Option 0", try to find the original column
        if not original_col:
            # Check if technical_name is already a readable name (like "Option 0")
            # Try reverse lookup: find which technical name maps to this readable name
            if feature_name_mapping:
                # Find technical names that map to this readable name
                for tech_name, readable_name in feature_name_mapping.items():
                    if readable_name == technical_name:
                        # Found the technical name, now get its original column
                        if original_column_mapping:
                            original_col = original_column_mapping.get(tech_name)
                            if original_col:
                                break
            # Also try direct reverse lookup in original_column_mapping
            if not original_col and original_column_mapping:
                # Reverse lookup: find technical names that map to original columns
                for tech_name, orig_col in original_column_mapping.items():
                    # Check if technical_name matches tech_name or if it's the readable version
                    if tech_name == technical_name or feature_name_mapping.get(tech_name) == technical_name:
                        if orig_col:
                            original_col = orig_col
                            break
        
        # Use original column if found, otherwise use readable name, otherwise use technical name
        if original_col:
            display_name = original_col
            logger.debug(f"Mapped '{technical_name}' -> original column: '{original_col}'")
        else:
            # Use readable name if available, but avoid generic names like "Option 0"
            readable_name = feature_name_mapping.get(technical_name, technical_name)
            # If readable name is generic (starts with "Option" or "Feature"), prefer technical name
            if readable_name.startswith("Option ") or readable_name.startswith("Feature "):
                # Try to use technical name if it's more descriptive
                display_name = technical_name if technical_name and not technical_name.startswith("Option ") else readable_name
                logger.debug(f"No original column for '{technical_name}', using generic name: '{display_name}'")
            else:
                display_name = readable_name
                logger.debug(f"Mapped '{technical_name}' -> readable name: '{display_name}'")
        
        # Aggregate contributions by original feature
        if display_name not in feature_aggregation:
            feature_aggregation[display_name] = 0
        feature_aggregation[display_name] += float(value)
    
    # Convert aggregated contributions to list
    for feature_name, total_value in feature_aggregation.items():
        mapped_contributions.append({
            'feature': feature_name,
            'shap_value': total_value,
            'value': total_value
        })
    
    # Sort by absolute value (ensure value is numeric)
    def get_abs_value(x):
        val = x.get('shap_value', x.get('value', 0))
        if isinstance(val, (list, np.ndarray)):
            return abs(float(np.mean(val)))
        return abs(float(val))
    mapped_contributions.sort(key=get_abs_value, reverse=True)
    
    # Update explanation - preserve base_value and prediction
    result = explanation.copy()
    result['feature_contributions'] = mapped_contributions
    result['feature_names'] = [c['feature'] for c in mapped_contributions]
    # Preserve base_value and prediction if they exist
    if 'base_value' not in result:
        result['base_value'] = explanation.get('base_value', 0.0)
    if 'prediction' not in result:
        result['prediction'] = explanation.get('prediction', 0.0)
    
    return result


async def _get_training_data(
    db: AsyncSession,
    model_record: MLModel,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Get training data for the model (for LIME and SHAP background).
    
    Args:
        db: Database session
        model_record: Model record
        sample_size: Maximum number of samples to return
        
    Returns:
        DataFrame with training features (after preprocessing if applicable)
    """
    if not model_record.data_source_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model does not have associated data source"
        )
    
    # Get data points
    query = select(DataPoint).where(DataPoint.source_id == model_record.data_source_id)
    
    if sample_size:
        query = query.limit(sample_size)
    
    result = await db.execute(query)
    data_points = result.scalars().all()
    
    if not data_points:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No training data found for this model"
        )
    
    # Convert to DataFrame
    records = []
    for point in data_points:
        record = point.data.copy() if isinstance(point.data, dict) else point.data
        # Add timestamp from database column if it exists and is in original_columns
        if hasattr(point, 'timestamp') and point.timestamp is not None:
            if model_record.original_columns and 'timestamp' in model_record.original_columns:
                # Convert timestamp to string format that can be parsed
                if hasattr(point.timestamp, 'isoformat'):
                    record['timestamp'] = point.timestamp.isoformat()
                elif hasattr(point.timestamp, 'strftime'):
                    record['timestamp'] = point.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    record['timestamp'] = str(point.timestamp)
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Apply preprocessing if available
    model_package = await model_storage.load_model(model_record.model_path, return_package=True)
    if isinstance(model_package, dict) and model_package.get("preprocessor"):
        preprocessor = model_package["preprocessor"]
        
        # Use original columns if available
        if model_record.original_columns:
            # Check for missing columns and fill them
            missing_cols = [col for col in model_record.original_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in training data: {missing_cols}. Filling with defaults.")
                
                # Fill missing columns with appropriate defaults
                for col in missing_cols:
                    # For timestamp/date columns, use NaT (Not a Time)
                    if any(keyword in col.lower() for keyword in ['timestamp', 'time', 'date']):
                        df[col] = pd.NaT
                    # For numeric-looking columns, use NaN
                    elif any(c.isdigit() for c in col) or col.startswith('num_') or col.endswith('_num'):
                        df[col] = np.nan
                    # For categorical/string columns, use empty string
                    else:
                        df[col] = ''
            
            # Ensure all original_columns are present and in the correct order
            for col in model_record.original_columns:
                if col not in df.columns:
                    # This shouldn't happen after filling, but just in case
                    if any(keyword in col.lower() for keyword in ['timestamp', 'time', 'date']):
                        df[col] = pd.NaT
                    else:
                        df[col] = np.nan
            
            # Reorder to match original_columns order
            df = df[model_record.original_columns]
        
        # Convert data types appropriately
        for col in df.columns:
            try:
                # Handle datetime columns
                if any(keyword in col.lower() for keyword in ['timestamp', 'time', 'date']):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                # Try numeric conversion for other columns
                else:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception as e:
                logger.debug(f"Could not convert column {col} to expected type: {e}")
                pass
        
        # Apply preprocessing
        try:
            df = preprocessor.transform(df)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error preprocessing training data: {error_msg}", exc_info=True)
            
            # If the error is about missing columns, provide a helpful message
            if "columns are missing" in error_msg or "missing" in error_msg.lower():
                # Extract missing column names from error if possible
                import re
                missing_match = re.search(r"columns are missing: \{([^}]+)\}", error_msg)
                if missing_match:
                    missing_cols_str = missing_match.group(1)
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Missing required columns for preprocessing: {missing_cols_str}. "
                               f"Please ensure all original columns are present in the training data."
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Error preprocessing training data: {error_msg}. "
                               f"Some required columns may be missing from the training data."
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error preprocessing training data: {error_msg}"
                )
    else:
        # No preprocessor - use features directly
        if model_record.features:
            available_features = [f for f in model_record.features if f in df.columns]
            if available_features:
                df = df[available_features]
    
    return df


@router.post("/shap", response_model=ShapExplanationResponse)
async def explain_shap(
    request: ShapExplanationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate SHAP explanation. If features are provided, explains a single instance.
    If features are None, generates global explanation from training data.
    """
    try:
        # Get model and package
        model_record, model_package = await _get_model_and_package(db, request.model_id)
        
        # Save model to temp file
        temp_model_path = await _save_model_temp(model_package)
        
        try:
            # Get feature names
            feature_names = model_package.get("feature_names") or model_record.features
            if not feature_names:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Feature names not available for this model"
                )
            
            # Initialize SHAP explainer
            shap_explainer = ShapExplainer(
                model_path=temp_model_path,
                feature_names=feature_names
            )
            
            # Get training data for background or global explanation
            training_data = await _get_training_data(
                db, 
                model_record, 
                sample_size=request.sample_size or request.background_samples or 100
            )
            background_data = training_data.values
            
            if request.features:
                # Local explanation for specific instance
                instance_array = await _transform_features(request.features, model_record, model_package)
                
                # Generate explanation for instance
                explanation = shap_explainer.explain_instance(
                    instance=instance_array,
                    background_data=background_data
                )
            else:
                # Global explanation from training data
                # Get feature importance across the dataset
                feature_importance = shap_explainer.get_feature_importance(
                    X=training_data.values,
                    background_data=background_data
                )
                
                # Convert to explanation format
                # Handle case where value might be a list (multi-class) or a number
                feature_contributions = []
                for name, value in feature_importance.items():
                    # Convert to float, handling both single values and lists
                    if isinstance(value, (list, np.ndarray)):
                        shap_val = float(np.mean(np.abs(value)))
                    else:
                        shap_val = float(value)
                    feature_contributions.append({
                        'feature': name,
                        'shap_value': shap_val,
                        'value': shap_val
                    })
                feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
                
                # Calculate average prediction as base value
                model = model_package.get("model")
                predictions = model.predict(training_data.values)
                avg_prediction = float(np.mean(predictions))
                
                explanation = {
                    'shap_values': list(feature_importance.values()),
                    'base_value': avg_prediction,
                    'prediction': avg_prediction,  # For global, prediction is the average
                    'feature_contributions': feature_contributions,
                    'feature_names': list(feature_importance.keys())
                }
            
            # Map explanation back to original feature names
            mapped_explanation = _map_explanation_to_original_features(explanation, model_record, model_package)
            
            # Log for debugging
            logger.info(f"SHAP explanation generated: {len(mapped_explanation.get('feature_contributions', []))} feature contributions")
            if mapped_explanation.get('feature_contributions'):
                logger.debug(f"Sample contribution: {mapped_explanation['feature_contributions'][0]}")
            
            # Safely convert shap_values to list of floats
            shap_vals = explanation['shap_values']
            if isinstance(shap_vals, np.ndarray):
                shap_vals = shap_vals.tolist()
            elif isinstance(shap_vals, list):
                # Handle nested lists (e.g., multi-class models)
                flattened = []
                for val in shap_vals:
                    if isinstance(val, (list, np.ndarray)):
                        # For multi-class or multi-dimensional, take mean
                        flattened.append(float(np.mean(np.abs(val))))
                    else:
                        flattened.append(float(val))
                shap_vals = flattened
            else:
                shap_vals = [float(shap_vals)]

            # Safely extract base_value and prediction from mapped explanation
            base_val = mapped_explanation.get('base_value', explanation.get('base_value', 0.0))
            if isinstance(base_val, (list, np.ndarray)):
                base_val = float(base_val[0] if len(base_val) > 0 else 0)
            else:
                base_val = float(base_val)
            
            # Safely extract prediction
            pred_val = mapped_explanation.get('prediction', explanation.get('prediction', 0.0))
            if isinstance(pred_val, (list, np.ndarray)):
                pred_val = float(pred_val[0] if len(pred_val) > 0 else 0)
            else:
                pred_val = float(pred_val)

            return ShapExplanationResponse(
                model_id=request.model_id,
                shap_values=shap_vals,
                base_value=base_val,
                prediction=pred_val,
                feature_contributions=mapped_explanation['feature_contributions'],
                feature_names=mapped_explanation['feature_names']
            )
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_model_path):
                os.unlink(temp_model_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating SHAP explanation: {str(e)}"
        )


@router.post("/lime", response_model=LimeExplanationResponse)
async def explain_lime(
    request: LimeExplanationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate LIME explanation. If features are provided, explains a single instance.
    If features are None, generates global explanation from training data.
    """
    try:
        # Get model and package
        model_record, model_package = await _get_model_and_package(db, request.model_id)
        
        # Save model to temp file
        temp_model_path = await _save_model_temp(model_package)
        
        try:
            # Get training data for LIME
            training_data = await _get_training_data(
                db, 
                model_record, 
                sample_size=request.sample_size or 1000
            )
            
            # Get feature names
            feature_names = model_package.get("feature_names") or model_record.features
            if not feature_names:
                feature_names = [f'feature_{i}' for i in range(training_data.shape[1])]
            
            # Determine mode
            mode = 'classification' if model_record.type == ModelType.CLASSIFICATION else 'regression'
            
            # Initialize LIME explainer
            lime_explainer = LimeExplainer(
                model_path=temp_model_path,
                training_data=training_data.values,
                feature_names=feature_names,
                mode=mode
            )
            
            if request.features:
                # Local explanation for specific instance
                instance_array = await _transform_features(request.features, model_record, model_package)
                
                # Generate explanation
                explanation = lime_explainer.explain_instance(
                    instance=instance_array.flatten(),
                    num_features=request.num_features,
                    num_samples=request.num_samples
                )
            else:
                # Global explanation - aggregate LIME explanations across training data
                # Sample a subset for efficiency
                sample_indices = np.random.choice(
                    len(training_data), 
                    size=min(50, len(training_data)), 
                    replace=False
                )
                sample_data = training_data.values[sample_indices]
                
                # Get explanations for multiple instances
                explanations = []
                for instance in sample_data:
                    exp = lime_explainer.explain_instance(
                        instance=instance.flatten(),
                        num_features=request.num_features,
                        num_samples=request.num_samples
                    )
                    explanations.append(exp)
                
                # Aggregate feature importance
                feature_aggregation = {}
                for exp in explanations:
                    for contrib in exp['feature_contributions']:
                        feature = contrib['feature']
                        weight = abs(contrib['weight'])
                        if feature not in feature_aggregation:
                            feature_aggregation[feature] = []
                        feature_aggregation[feature].append(weight)
                
                # Calculate average importance
                feature_contributions = [
                    {
                        'feature': feature,
                        'weight': float(np.mean(weights)),
                        'value': float(np.mean(weights))
                    }
                    for feature, weights in feature_aggregation.items()
                ]
                feature_contributions.sort(key=lambda x: abs(x['weight']), reverse=True)
                
                # Get average prediction
                model = model_package.get("model")
                predictions = model.predict(sample_data)
                avg_prediction = float(np.mean(predictions))
                
                explanation = {
                    'prediction': avg_prediction,
                    'predicted_class': int(np.argmax(avg_prediction)) if mode == 'classification' and isinstance(avg_prediction, np.ndarray) else None,
                    'local_prediction': avg_prediction,
                    'feature_contributions': feature_contributions,
                    'intercept': 0.0,  # Not meaningful for aggregated explanation
                    'score': None,
                    'feature_names': [c['feature'] for c in feature_contributions],
                    'mode': mode
                }
            
            # Map explanation back to original feature names
            mapped_explanation = _map_explanation_to_original_features(explanation, model_record, model_package)
            
            # Build feature contributions, only including description if it exists
            feature_contributions = []
            for c in mapped_explanation['feature_contributions']:
                contrib = {
                    'feature': c['feature'],
                    'value': c.get('shap_value', c.get('value', c.get('weight', 0)))
                }
                # Only add description if it exists and is not None
                description = c.get('description')
                if description is not None:
                    contrib['description'] = description
                feature_contributions.append(contrib)
            
            return LimeExplanationResponse(
                model_id=request.model_id,
                prediction=explanation['prediction'],
                predicted_class=explanation.get('predicted_class'),
                local_prediction=explanation.get('local_prediction'),
                feature_contributions=feature_contributions,
                intercept=explanation['intercept'],
                score=explanation.get('score'),
                feature_names=mapped_explanation['feature_names'],
                mode=explanation['mode']
            )
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_model_path):
                os.unlink(temp_model_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating LIME explanation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating LIME explanation: {str(e)}"
        )


@router.post("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance_alt(
    request: FeatureImportanceRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get feature importance for a model (alternative endpoint name for frontend compatibility).
    """
    # Delegate to the main importance endpoint
    return await get_feature_importance(request, db)


@router.post("/importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    request: FeatureImportanceRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get feature importance for a model.
    """
    try:
        # Get model and package
        model_record, model_package = await _get_model_and_package(db, request.model_id)
        
        # Save model to temp file
        temp_model_path = await _save_model_temp(model_package)
        
        try:
            # Get feature names - prefer from package metadata, fall back to database
            feature_names = model_package.get("feature_names") or model_record.features
            if not feature_names:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Feature names not available for this model"
                )
            
            # Ensure feature_names is a list
            if not isinstance(feature_names, list):
                feature_names = list(feature_names) if hasattr(feature_names, '__iter__') else [feature_names]
            
            # Ensure we have feature names (not empty)
            if not feature_names or len(feature_names) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Feature names list is empty"
                )
            
            logger.info(f"Using {len(feature_names)} feature names for importance calculation")
            logger.debug(f"First few feature names: {feature_names[:5]}")
            
            # Initialize feature importance analyzer
            importance_analyzer = FeatureImportanceAnalyzer(
                model_path=temp_model_path,
                feature_names=feature_names
            )
            
            # Get feature name mapping and original column mapping
            feature_name_mapping = _get_feature_name_mapping(model_record, model_package)
            original_column_mapping = _get_original_column_mapping(model_record, model_package)
            
            if request.method == "model":
                # Model-based importance
                importance = importance_analyzer.get_model_importance()
                if importance is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Model does not support built-in feature importance"
                    )
                
                # Apply feature name mapping
                logger.info(f"Feature importance before mapping: {list(importance.keys())[:5]}...")
                logger.info(f"Feature name mapping available: {len(feature_name_mapping)} mappings")
                logger.info(f"Original column mapping available: {len(original_column_mapping)} mappings")
                if feature_name_mapping:
                    sample_mappings = list(feature_name_mapping.items())[:3]
                    logger.info(f"Sample mappings: {sample_mappings}")
                    # Check if mapping keys match importance keys
                    importance_keys = set(importance.keys())
                    mapping_keys = set(feature_name_mapping.keys())
                    if importance_keys != mapping_keys:
                        logger.warning(f"Feature name mismatch: {len(importance_keys)} importance keys vs {len(mapping_keys)} mapping keys")
                        missing_in_mapping = importance_keys - mapping_keys
                        if missing_in_mapping:
                            logger.warning(f"Features in importance but not in mapping: {list(missing_in_mapping)[:5]}")
                
                importance = _apply_feature_name_mapping(importance, feature_name_mapping, original_column_mapping)
                
                logger.info(f"Feature importance after mapping: {list(importance.keys())[:5]}...")
                
                return FeatureImportanceResponse(
                    model_id=request.model_id,
                    method="model",
                    importance=importance,
                    importance_std=None,
                    feature_names=list(importance.keys())
                )
            
            elif request.method == "permutation":
                # Permutation importance - need data
                training_data = await _get_training_data(db, model_record, sample_size=500)
                
                # Need target values - try to get from training job or estimate
                # For now, we'll use a dummy target (this is a limitation)
                # In practice, you might want to store target column info with the model
                y_dummy = np.zeros(len(training_data))
                
                result = importance_analyzer.get_permutation_importance(
                    X=training_data.values,
                    y=y_dummy,
                    n_repeats=request.n_repeats or 10
                )
                
                # Apply feature name mapping
                importance_mapped = _apply_feature_name_mapping(
                    result['importance_mean'],
                    feature_name_mapping,
                    original_column_mapping
                )
                importance_std_mapped = _apply_feature_name_mapping(
                    result['importance_std'],
                    feature_name_mapping,
                    original_column_mapping
                )
                feature_names_mapped = _apply_feature_name_mapping_to_list(
                    result['feature_names'],
                    feature_name_mapping,
                    original_column_mapping
                )
                
                return FeatureImportanceResponse(
                    model_id=request.model_id,
                    method="permutation",
                    importance=importance_mapped,
                    importance_std=importance_std_mapped,
                    feature_names=feature_names_mapped
                )
            
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown importance method: {request.method}. Use 'model' or 'permutation'. For comprehensive, use /explain/importance/comprehensive endpoint."
                )
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_model_path):
                os.unlink(temp_model_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating feature importance: {str(e)}"
        )


@router.post("/importance/comprehensive", response_model=ComprehensiveImportanceResponse)
async def get_comprehensive_importance(
    request: FeatureImportanceRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive feature importance using multiple methods.
    """
    try:
        # Get model and package
        model_record, model_package = await _get_model_and_package(db, request.model_id)
        
        # Save model to temp file
        temp_model_path = await _save_model_temp(model_package)
        
        try:
            # Get feature names
            feature_names = model_package.get("feature_names") or model_record.features
            if not feature_names:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Feature names not available for this model"
                )
            
            # Initialize feature importance analyzer
            importance_analyzer = FeatureImportanceAnalyzer(
                model_path=temp_model_path,
                feature_names=feature_names
            )
            
            # Get feature name mapping and original column mapping
            feature_name_mapping = _get_feature_name_mapping(model_record, model_package)
            original_column_mapping = _get_original_column_mapping(model_record, model_package)
            
            # Comprehensive importance - need data
            training_data = await _get_training_data(db, model_record, sample_size=500)
            y_dummy = np.zeros(len(training_data))
            
            result = importance_analyzer.get_comprehensive_importance(
                X=training_data.values,
                y=y_dummy,
                include_permutation=True,
                n_repeats=request.n_repeats or 10
            )
            
            # Apply feature name mapping to all importance dictionaries
            model_importance = result.get('model_importance')
            if model_importance:
                model_importance = _apply_feature_name_mapping(model_importance, feature_name_mapping, original_column_mapping)
            
            permutation_importance = result.get('permutation_importance')
            if permutation_importance:
                permutation_importance = _apply_feature_name_mapping(permutation_importance, feature_name_mapping, original_column_mapping)
            
            permutation_std = result.get('permutation_std')
            if permutation_std:
                permutation_std = _apply_feature_name_mapping(permutation_std, feature_name_mapping, original_column_mapping)
            
            return ComprehensiveImportanceResponse(
                model_id=request.model_id,
                model_importance=model_importance,
                permutation_importance=permutation_importance,
                permutation_std=permutation_std
            )
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_model_path):
                os.unlink(temp_model_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating feature importance: {str(e)}"
        )


@router.post("/waterfall", response_model=ShapWaterfallData)
async def get_waterfall_data(
    request: ShapExplanationRequest,
    max_display: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    """
    Get SHAP waterfall plot data for a single instance.
    """
    try:
        # Get model and package
        model_record, model_package = await _get_model_and_package(db, request.model_id)
        
        # Save model to temp file
        temp_model_path = await _save_model_temp(model_package)
        
        try:
            # Get feature names
            feature_names = model_package.get("feature_names") or model_record.features
            if not feature_names:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Feature names not available for this model"
                )
            
            # Transform original features to preprocessed format
            instance_array = await _transform_features(request.features, model_record, model_package)
            
            # Initialize SHAP explainer
            shap_explainer = ShapExplainer(
                model_path=temp_model_path,
                feature_names=feature_names
            )
            
            # Get background data if needed
            background_data = None
            if shap_explainer.explainer is None:
                training_data = await _get_training_data(db, model_record, sample_size=request.background_samples or 100)
                background_data = training_data.values
            
            # Generate waterfall data
            waterfall_data = shap_explainer.get_waterfall_data(
                instance=instance_array,
                background_data=background_data,
                max_display=max_display
            )
            
            # Map to original feature names
            mapped_explanation = _map_explanation_to_original_features(
                {
                    'feature_contributions': waterfall_data['contributions'],
                    'feature_names': waterfall_data['feature_names']
                },
                model_record,
                model_package
            )
            
            return ShapWaterfallData(
                model_id=request.model_id,
                base_value=waterfall_data['base_value'] if isinstance(waterfall_data['base_value'], (int, float)) else waterfall_data['base_value'][0],
                prediction=waterfall_data['prediction'],
                contributions=mapped_explanation['feature_contributions'][:max_display],
                feature_names=mapped_explanation['feature_names'][:max_display],
                shap_values=[c.get('shap_value', c.get('value', 0)) for c in mapped_explanation['feature_contributions'][:max_display]]
            )
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_model_path):
                os.unlink(temp_model_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating waterfall data: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating waterfall data: {str(e)}"
        )


@router.post("/summary", response_model=ExplanationSummaryResponse)
async def get_explanation_summary(
    request: ExplanationSummaryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get explanation summary across a dataset.
    """
    try:
        # Get model and package
        model_record, model_package = await _get_model_and_package(db, request.model_id)
        
        # Save model to temp file
        temp_model_path = await _save_model_temp(model_package)
        
        try:
            # Get training data
            training_data = await _get_training_data(
                db,
                model_record,
                sample_size=request.sample_size or 100
            )
            
            # Get feature names
            feature_names = model_package.get("feature_names") or model_record.features
            if not feature_names:
                feature_names = [f'feature_{i}' for i in range(training_data.shape[1])]
            
            # Use SHAP for summary (more efficient for global importance)
            shap_explainer = ShapExplainer(
                model_path=temp_model_path,
                feature_names=feature_names
            )
            
            # Generate summary
            summary = shap_explainer.summary(
                X=training_data.values,
                max_features=request.max_features
            )
            
            # Get feature name mapping and original column mapping
            feature_name_mapping = _get_feature_name_mapping(model_record, model_package)
            original_column_mapping = _get_original_column_mapping(model_record, model_package)
            
            # Apply mapping to feature importance
            feature_importance = _apply_feature_name_mapping(
                summary['feature_importance'],
                feature_name_mapping,
                original_column_mapping
            )
            
            # Apply mapping to top features
            top_features = _apply_feature_name_mapping_to_list(
                summary['top_features'],
                feature_name_mapping,
                original_column_mapping
            )
            
            return ExplanationSummaryResponse(
                model_id=request.model_id,
                dataset_id=request.dataset_id,
                feature_importance=feature_importance,
                top_features=top_features,
                num_samples=summary['num_samples'],
                num_features=summary['num_features'],
                base_value=summary['base_value'] if isinstance(summary['base_value'], (int, float)) else summary['base_value'][0],
                method="shap"
            )
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_model_path):
                os.unlink(temp_model_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating explanation summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating explanation summary: {str(e)}"
        )

