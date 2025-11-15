"""
Pydantic schemas for ML models and training.
"""
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from app.models.database.ml_models import ModelType, ModelStatus
from app.models.database.training_jobs import TrainingJobStatus
from app.models.database.batch_jobs import BatchJobStatus


# Preprocessing Configuration Schemas
class PreprocessingConfig(BaseModel):
    """Schema for preprocessing configuration."""
    model_config = ConfigDict(protected_namespaces=())

    use_auto_preprocessing: bool = Field(
        True,
        description="Use intelligent AutoPreprocessor (recommended)"
    )
    enable_feature_engineering: bool = Field(
        True,
        description="Enable automatic feature engineering (polynomial, log, binning, clustering)"
    )
    enable_outlier_handling: bool = Field(
        True,
        description="Enable outlier detection and handling"
    )
    enable_feature_selection: bool = Field(
        True,
        description="Enable feature selection to reduce dimensionality"
    )
    max_features: Optional[int] = Field(
        None,
        description="Maximum number of features to keep (None = auto-determine)"
    )
    outlier_method: Literal['iqr', 'zscore', 'isolation_forest', 'lof'] = Field(
        'iqr',
        description="Method for outlier detection"
    )
    outlier_strategy: Literal['remove', 'cap', 'flag', 'transform'] = Field(
        'cap',
        description="Strategy for handling outliers"
    )
    feature_selection_method: Literal['univariate', 'model_based', 'rfe', 'variance', 'correlation'] = Field(
        'model_based',
        description="Method for feature selection"
    )
    scale_numeric: bool = Field(
        True,
        description="Scale numeric features (for basic preprocessor)"
    )
    handle_missing: Literal['mean', 'median', 'mode', 'drop'] = Field(
        'mean',
        description="Strategy for handling missing values (for basic preprocessor)"
    )


# Training Request Schemas
class TrainingRequest(BaseModel):
    """Schema for model training request."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str = Field(..., min_length=1, max_length=255, description="Name for the model")
    model_type: str = Field(..., description="Type of model: regression, classification, anomaly_detection")
    data_source_id: int = Field(..., description="ID of data source to use for training")
    target_column: str = Field(..., description="Name of target column")
    algorithm: str = Field("random_forest", description="ML algorithm: random_forest, xgboost")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Model hyperparameters")
    training_config: Optional[Dict[str, Any]] = Field(None, description="Training configuration")
    description: Optional[str] = Field(None, max_length=1000, description="Model description")


class TrainingResponse(BaseModel):
    """Schema for training job response."""
    model_config = ConfigDict(protected_namespaces=())
    
    job_id: int
    model_id: Optional[int] = None
    status: TrainingJobStatus
    message: str


class TrainingJobResponse(BaseModel):
    """Schema for training job details."""
    model_config = ConfigDict(protected_namespaces=(), from_attributes=True)
    
    id: int
    model_id: Optional[int]
    data_source_id: Optional[int]
    status: TrainingJobStatus
    model_type: str
    progress: float
    current_epoch: Optional[int]
    total_epochs: Optional[int]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    metrics: Optional[Dict[str, Any]]
    created_at: datetime


class TrainingJobListResponse(BaseModel):
    """Schema for training job list."""
    jobs: List[TrainingJobResponse]
    total: int


# Prediction Schemas
class PredictionRequest(BaseModel):
    """Schema for single prediction request."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int = Field(..., description="ID of model to use")
    features: Dict[str, Any] = Field(..., description="Feature values for prediction")


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int = Field(..., description="ID of model to use")
    features_list: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int
    prediction: float
    confidence: Optional[Dict[str, float]] = None


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int
    predictions: List[float]
    total: int


# Model Management Schemas
class MLModelResponse(BaseModel):
    """Schema for ML model response."""
    model_config = ConfigDict(
        protected_namespaces=(),
        from_attributes=True
    )
    
    id: int
    name: str
    type: ModelType
    status: ModelStatus
    version: str
    description: Optional[str]
    data_source_id: Optional[int]
    features: Optional[List[str]]
    original_columns: Optional[List[str]]  # Original column names before preprocessing
    feature_name_mapping: Optional[Dict[str, str]]  # Mapping from technical to readable feature names
    hyperparameters: Optional[Dict[str, Any]]
    feature_importance: Optional[Dict[str, float]]
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    rmse: Optional[float]
    mae: Optional[float]
    r2_score: Optional[float]
    model_path: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]


class MLModelListResponse(BaseModel):
    """Schema for ML model list."""
    models: List[MLModelResponse]
    total: int


class ModelDetailResponse(MLModelResponse):
    """Schema for detailed model information."""
    feature_importance: Optional[Dict[str, float]] = None
    training_history: Optional[List[Dict[str, Any]]] = None


# Anomaly Detection Schemas
class AnomalyDetectionRequest(BaseModel):
    """Schema for anomaly detection request."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int = Field(..., description="ID of anomaly detection model")
    features: Dict[str, Any] = Field(..., description="Feature values for anomaly detection")


class BatchAnomalyDetectionRequest(BaseModel):
    """Schema for batch anomaly detection request."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int = Field(..., description="ID of anomaly detection model")
    features_list: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")


class AnomalyDetectionResponse(BaseModel):
    """Schema for anomaly detection response."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int
    is_anomaly: bool
    score: float
    probability: float


class BatchAnomalyDetectionResponse(BaseModel):
    """Schema for batch anomaly detection response."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int
    results: List[Dict[str, Any]]  # List of {is_anomaly, score, probability}
    total: int
    anomaly_count: int
    normal_count: int


class AnomalyStatsResponse(BaseModel):
    """Schema for anomaly statistics response."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int
    total_predictions: int
    anomaly_count: int
    normal_count: int
    anomaly_rate: float
    score_stats: Dict[str, float]


# Classification Schemas
class ClassificationRequest(BaseModel):
    """Schema for classification request."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int = Field(..., description="ID of classification model")
    features: Dict[str, Any] = Field(..., description="Feature values for classification")


class BatchClassificationRequest(BaseModel):
    """Schema for batch classification request."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int = Field(..., description="ID of classification model")
    features_list: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")


class ClassificationResponse(BaseModel):
    """Schema for classification response."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int
    predicted_class: str
    probabilities: Dict[str, float]  # Class -> probability mapping


class BatchClassificationResponse(BaseModel):
    """Schema for batch classification response."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int
    predictions: List[Dict[str, Any]]  # List of {predicted_class, probabilities}
    total: int


# Model Versioning Schemas
class ModelVersionResponse(BaseModel):
    """Schema for model version response."""
    model_config = ConfigDict(
        protected_namespaces=(),
        from_attributes=True
    )
    
    id: int
    model_id: int
    version: str
    performance_metrics: Optional[Dict[str, Any]]
    features: Optional[List[str]]
    hyperparameters: Optional[Dict[str, Any]]
    model_path: Optional[str]
    model_size_bytes: Optional[int]
    training_date: Optional[datetime]
    dataset_size: Optional[int]
    training_duration_seconds: Optional[float]
    feature_importance: Optional[Dict[str, float]]
    is_active: bool
    is_archived: bool
    notes: Optional[str]
    tags: Optional[List[str]]
    created_at: datetime


class ModelVersionListResponse(BaseModel):
    """Schema for model version list."""
    versions: List[ModelVersionResponse]
    total: int


class ModelComparisonResponse(BaseModel):
    """Schema for model version comparison."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int
    versions: List[ModelVersionResponse]
    comparison_metrics: Dict[str, Any]  # Comparison of metrics across versions


class ModelRollbackRequest(BaseModel):
    """Schema for model rollback request."""
    version_id: int = Field(..., description="ID of version to rollback to")


class ModelRollbackResponse(BaseModel):
    """Schema for model rollback response."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int
    previous_version: str
    new_active_version: str
    message: str


# Model Registry Schemas
class ModelRegistryResponse(BaseModel):
    """Schema for model registry entry."""
    model: MLModelResponse
    active_version: Optional[ModelVersionResponse]
    total_versions: int
    tags: Optional[List[str]]


class ModelRegistryListResponse(BaseModel):
    """Schema for model registry list."""
    models: List[ModelRegistryResponse]
    total: int


# Retraining Schemas
class RetrainingScheduleRequest(BaseModel):
    """Schema for scheduling retraining."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int
    new_data_threshold: Optional[int] = Field(None, description="Minimum new data points to trigger retraining")
    performance_threshold: Optional[float] = Field(None, description="Performance degradation threshold")
    training_config: Optional[Dict[str, Any]] = Field(None, description="Training configuration")


class RetrainingScheduleResponse(BaseModel):
    """Schema for retraining schedule response."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int
    scheduled: bool
    message: str


# Batch Prediction Job Schemas
class BatchJobCreateRequest(BaseModel):
    """Schema for creating a batch prediction job."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int = Field(..., description="ID of model to use for predictions")
    input_type: str = Field("data_source", description="Type of input: data_source, file")
    data_source_id: Optional[int] = Field(None, description="ID of data source (if input_type is data_source)")
    input_config: Optional[Dict[str, Any]] = Field(None, description="Input configuration (file path, etc.)")
    job_name: Optional[str] = Field(None, description="Optional name for the job")
    result_format: str = Field("csv", description="Format for results: csv, json, parquet")
    scheduled_at: Optional[datetime] = Field(None, description="When to run the job (None for immediate)")


class BatchJobResponse(BaseModel):
    """Schema for batch prediction job response."""
    model_config = ConfigDict(protected_namespaces=(), from_attributes=True)
    
    id: int
    model_id: int
    data_source_id: Optional[int]
    status: str
    job_name: Optional[str]
    input_type: str
    progress: float
    total_records: Optional[int]
    processed_records: Optional[int]
    failed_records: Optional[int]
    result_path: Optional[str]
    result_format: str
    error_message: Optional[str]
    scheduled_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime


class BatchJobListResponse(BaseModel):
    """Schema for batch job list."""
    jobs: List[BatchJobResponse]
    total: int

