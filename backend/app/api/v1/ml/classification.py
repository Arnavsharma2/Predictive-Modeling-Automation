"""
Classification API endpoints.
"""
from typing import List
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.database.ml_models import MLModel, ModelStatus, ModelType
from app.models.schemas.ml import (
    ClassificationRequest,
    ClassificationResponse,
    BatchClassificationRequest,
    BatchClassificationResponse
)
from app.ml.storage.model_storage import model_storage
from app.ml.models.classifier import Classifier

logger = get_logger(__name__)

router = APIRouter(prefix="/classify", tags=["Classification"])


@router.post("", response_model=ClassificationResponse)
async def classify(
    request: ClassificationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Classify a single data point.
    """
    try:
        # Get model
        model_record = await db.get(MLModel, request.model_id)
        
        if not model_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {request.model_id} not found"
            )
        
        if model_record.type != ModelType.CLASSIFICATION:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {request.model_id} is not a classification model"
            )
        
        if model_record.status != ModelStatus.TRAINED and model_record.status != ModelStatus.DEPLOYED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {request.model_id} is not ready for predictions (status: {model_record.status})"
            )
        
        # Load model
        if not model_record.model_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model path not found"
            )
        
        model = await model_storage.load_model(model_record.model_path)
        
        # Prepare features
        features_df = pd.DataFrame([request.features])
        
        # Ensure feature order matches training
        if model_record.features:
            available_features = [f for f in model_record.features if f in features_df.columns]
            missing_features = [f for f in model_record.features if f not in features_df.columns]
            
            if missing_features:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required features: {missing_features}"
                )
            
            features_df = features_df[model_record.features]
        
        # Create classifier and load model
        classifier = Classifier()
        classifier.model = model
        classifier.is_trained = True
        classifier.feature_names = model_record.features
        
        # Get classes from model metadata or model itself
        if hasattr(model, 'classes_'):
            classifier.classes_ = model.classes_
        elif hasattr(model, 'classes'):
            classifier.classes_ = model.classes
        
        # Classify
        result = classifier.predict_with_proba(features_df)
        
        predicted_class = str(result["predictions"][0])
        probabilities = result["probabilities"][0]
        classes = result["classes"]
        
        # Create probability dictionary
        prob_dict = {}
        if classes:
            for i, cls in enumerate(classes):
                prob_dict[str(cls)] = float(probabilities[i])
        else:
            # Fallback if classes not available
            for i in range(len(probabilities)):
                prob_dict[f"class_{i}"] = float(probabilities[i])
        
        return ClassificationResponse(
            model_id=request.model_id,
            predicted_class=predicted_class,
            probabilities=prob_dict
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error classifying: {str(e)}"
        )


@router.post("/batch", response_model=BatchClassificationResponse)
async def classify_batch(
    request: BatchClassificationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Classify multiple data points in batch.
    """
    try:
        # Get model
        model_record = await db.get(MLModel, request.model_id)
        
        if not model_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {request.model_id} not found"
            )
        
        if model_record.type != ModelType.CLASSIFICATION:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {request.model_id} is not a classification model"
            )
        
        if model_record.status != ModelStatus.TRAINED and model_record.status != ModelStatus.DEPLOYED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {request.model_id} is not ready for predictions"
            )
        
        # Load model
        if not model_record.model_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model path not found"
            )
        
        model = await model_storage.load_model(model_record.model_path)
        
        # Prepare features
        features_df = pd.DataFrame(request.features_list)
        
        # Ensure feature order matches training
        if model_record.features:
            missing_features = [f for f in model_record.features if f not in features_df.columns]
            if missing_features:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required features: {missing_features}"
                )
            
            features_df = features_df[model_record.features]
        
        # Create classifier
        classifier = Classifier()
        classifier.model = model
        classifier.is_trained = True
        classifier.feature_names = model_record.features
        
        # Get classes
        if hasattr(model, 'classes_'):
            classifier.classes_ = model.classes_
        elif hasattr(model, 'classes'):
            classifier.classes_ = model.classes
        
        # Classify
        result = classifier.predict_with_proba(features_df)
        
        # Format results
        predictions = []
        classes = result["classes"]
        probabilities = result["probabilities"]
        
        for i in range(len(request.features_list)):
            predicted_class = str(result["predictions"][i])
            
            # Create probability dictionary
            prob_dict = {}
            if classes:
                for j, cls in enumerate(classes):
                    prob_dict[str(cls)] = float(probabilities[i][j])
            else:
                for j in range(len(probabilities[i])):
                    prob_dict[f"class_{j}"] = float(probabilities[i][j])
            
            predictions.append({
                "predicted_class": predicted_class,
                "probabilities": prob_dict
            })
        
        return BatchClassificationResponse(
            model_id=request.model_id,
            predictions=predictions,
            total=len(predictions)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error classifying batch: {str(e)}"
        )

