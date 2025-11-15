"""
ML model prediction API endpoints.
"""
from typing import List
from datetime import datetime
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.database.ml_models import MLModel, ModelStatus
from app.models.schemas.ml import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from app.ml.storage.model_storage import model_storage
from app.ml.models.predictor import RegressionPredictor
from app.ml.preprocessing.preprocessor import DataPreprocessor
from app.core.websocket import manager as ws_manager

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.post("", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Make a single prediction.
    """
    try:
        # Get model
        model_record = await db.get(MLModel, request.model_id)
        
        if not model_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {request.model_id} not found"
            )
        
        if model_record.status != ModelStatus.TRAINED and model_record.status != ModelStatus.DEPLOYED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {request.model_id} is not ready for predictions (status: {model_record.status})"
            )
        
        # Load model package (includes preprocessor)
        if not model_record.model_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model path not found"
            )

        model_package = await model_storage.load_model(model_record.model_path, return_package=True)

        # Extract components
        if isinstance(model_package, dict):
            model = model_package.get("model")
            preprocessor = model_package.get("preprocessor")
        else:
            # Legacy format - no preprocessor
            model = model_package
            preprocessor = None

        # Prepare features from user input (in original column format)
        features_df = pd.DataFrame([request.features])

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
                    # Try to convert to numeric
                    features_df[col] = pd.to_numeric(features_df[col], errors='ignore')
                except:
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
                missing_features = [f for f in model_record.features if f in features_df.columns]
                if missing_features:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Missing required features: {missing_features}"
                    )

                features_df = features_df[model_record.features]
        
        # Make prediction
        predictor = RegressionPredictor()
        predictor.model = model
        predictor.is_trained = True
        predictor.feature_names = model_record.features
        
        prediction = predictor.predict(features_df)
        
        # Get confidence if available
        confidence = None
        try:
            pred_with_conf = predictor.predict_with_confidence(features_df)
            if "std" in pred_with_conf:
                confidence = {
                    "std": float(pred_with_conf["std"][0]),
                    "lower_bound": float(pred_with_conf["lower_bound"][0]),
                    "upper_bound": float(pred_with_conf["upper_bound"][0])
                }
        except Exception:
            pass  # Confidence not available
        
        prediction_value = float(prediction[0])
        response = PredictionResponse(
            model_id=request.model_id,
            prediction=prediction_value,
            confidence=confidence
        )
        
        # Broadcast prediction via WebSocket for real-time streaming
        try:
            await ws_manager.broadcast_prediction(
                request.model_id,
                {
                    "prediction": prediction_value,
                    "features": request.features,
                    "confidence": confidence,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            # Don't fail the prediction if WebSocket broadcast fails
            logger.warning(f"Failed to broadcast prediction via WebSocket: {e}")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Make batch predictions.
    """
    try:
        # Get model
        model_record = await db.get(MLModel, request.model_id)
        
        if not model_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {request.model_id} not found"
            )
        
        if model_record.status != ModelStatus.TRAINED and model_record.status != ModelStatus.DEPLOYED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {request.model_id} is not ready for predictions (status: {model_record.status})"
            )
        
        # Load model package (includes preprocessor)
        if not model_record.model_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model path not found"
            )

        model_package = await model_storage.load_model(model_record.model_path, return_package=True)

        # Extract components
        if isinstance(model_package, dict):
            model = model_package.get("model")
            preprocessor = model_package.get("preprocessor")
        else:
            # Legacy format - no preprocessor
            model = model_package
            preprocessor = None

        # Prepare features from user input (in original column format)
        features_df = pd.DataFrame(request.features_list)

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
                    # Try to convert to numeric
                    features_df[col] = pd.to_numeric(features_df[col], errors='ignore')
                except:
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
        
        # Make predictions
        predictor = RegressionPredictor()
        predictor.model = model
        predictor.is_trained = True
        predictor.feature_names = model_record.features
        
        predictions = predictor.predict(features_df)
        
        return BatchPredictionResponse(
            model_id=request.model_id,
            predictions=[float(p) for p in predictions],
            total=len(predictions)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making batch predictions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making batch predictions: {str(e)}"
        )

