"""
Anomaly detection API endpoints.
"""
from typing import List
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.database.ml_models import MLModel, ModelStatus, ModelType
from app.models.schemas.ml import (
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    BatchAnomalyDetectionRequest,
    BatchAnomalyDetectionResponse,
    AnomalyStatsResponse
)
from app.ml.storage.model_storage import model_storage
from app.ml.models.anomaly_detector import AnomalyDetector

logger = get_logger(__name__)

router = APIRouter(prefix="/anomaly", tags=["Anomaly Detection"])


@router.post("/detect", response_model=AnomalyDetectionResponse)
async def detect_anomaly(
    request: AnomalyDetectionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Detect anomaly in a single data point.
    """
    try:
        # Get model
        model_record = await db.get(MLModel, request.model_id)
        
        if not model_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {request.model_id} not found"
            )
        
        if model_record.type != ModelType.ANOMALY_DETECTION:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {request.model_id} is not an anomaly detection model"
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
        
        # Create detector and load model
        detector = AnomalyDetector()
        detector.model = model
        detector.is_trained = True
        detector.feature_names = model_record.features
        
        # Load scaler if available (stored in model metadata)
        # For now, we'll assume the model includes the scaler
        
        # Detect anomaly
        result = detector.predict_with_scores(features_df)
        
        is_anomaly = bool(result["is_anomaly"][0])
        score = float(result["scores"][0])
        probability = float(result["anomaly_probability"][0])
        
        return AnomalyDetectionResponse(
            model_id=request.model_id,
            is_anomaly=is_anomaly,
            score=score,
            probability=probability
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting anomaly: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error detecting anomaly: {str(e)}"
        )


@router.post("/detect/batch", response_model=BatchAnomalyDetectionResponse)
async def detect_anomalies_batch(
    request: BatchAnomalyDetectionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Detect anomalies in batch data.
    """
    try:
        # Get model
        model_record = await db.get(MLModel, request.model_id)
        
        if not model_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {request.model_id} not found"
            )
        
        if model_record.type != ModelType.ANOMALY_DETECTION:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {request.model_id} is not an anomaly detection model"
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
        
        # Create detector
        detector = AnomalyDetector()
        detector.model = model
        detector.is_trained = True
        detector.feature_names = model_record.features
        
        # Detect anomalies
        result = detector.predict_with_scores(features_df)
        
        # Format results
        results = []
        anomaly_count = 0
        normal_count = 0
        
        for i in range(len(request.features_list)):
            is_anomaly = bool(result["is_anomaly"][i])
            if is_anomaly:
                anomaly_count += 1
            else:
                normal_count += 1
            
            results.append({
                "is_anomaly": is_anomaly,
                "score": float(result["scores"][i]),
                "probability": float(result["anomaly_probability"][i])
            })
        
        return BatchAnomalyDetectionResponse(
            model_id=request.model_id,
            results=results,
            total=len(results),
            anomaly_count=anomaly_count,
            normal_count=normal_count
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting anomalies in batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error detecting anomalies: {str(e)}"
        )


@router.get("/stats/{model_id}", response_model=AnomalyStatsResponse)
async def get_anomaly_stats(
    model_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get anomaly detection statistics for a model.
    """
    model = await db.get(MLModel, model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    if model.type != ModelType.ANOMALY_DETECTION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {model_id} is not an anomaly detection model"
        )
    
    # For now, return basic stats from model metadata
    # In a real implementation, you might track prediction history
    return AnomalyStatsResponse(
        model_id=model_id,
        total_predictions=0,  # Would come from prediction history
        anomaly_count=0,
        normal_count=0,
        anomaly_rate=0.0,
        score_stats={}
    )

