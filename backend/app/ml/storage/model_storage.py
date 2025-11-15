"""
Model persistence and storage.
"""
import joblib
import pickle
from typing import Optional, Any, Dict
from pathlib import Path
import uuid
from datetime import datetime
import io
import asyncio
import time

from app.storage.cloud_storage import cloud_storage
from app.core.logging import get_logger

logger = get_logger(__name__)


class ModelStorage:
    """Handle model persistence to cloud storage."""
    
    def __init__(self):
        """Initialize model storage."""
        self.storage = cloud_storage
        # Simple in-memory cache to avoid reloading the same model multiple times
        # Key: model_path, Value: (model_package, timestamp)
        self._model_cache: Dict[str, tuple] = {}
        self._cache_lock = asyncio.Lock()
        # Cache timeout: 5 minutes (models don't change after being saved)
        self._cache_timeout = 300
    
    def _generate_model_path(self, model_name: str, version: str = "1.0.0") -> str:
        """
        Generate a unique path for model storage.
        
        Args:
            model_name: Name of the model
            version: Model version
            
        Returns:
            Storage path for the model
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{model_name}_v{version}_{timestamp}_{unique_id}.joblib"
        return f"models/{model_name}/{filename}"
    
    async def save_model(
        self,
        model: Any,
        model_name: str,
        version: str = "1.0.0",
        metadata: Optional[dict] = None,
        preprocessor: Optional[Any] = None,
        feature_names: Optional[list] = None
    ) -> str:
        """
        Save model to cloud storage with complete pipeline.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            version: Model version
            metadata: Additional metadata to store with model
            preprocessor: Preprocessing pipeline to save with model
            feature_names: List of feature names expected by the model
            
        Returns:
            Path to saved model in cloud storage
        """
        try:
            # Create a complete model package including preprocessing pipeline
            model_package = {
                "model": model,
                "preprocessor": preprocessor,
                "feature_names": feature_names,
                "metadata": metadata or {},
                "version": version,
                "saved_at": datetime.utcnow().isoformat()
            }
            
            # Serialize complete package (run in thread pool to avoid blocking event loop)
            logger.info("Serializing model package (this may take a moment for large models)...")
            model_bytes = io.BytesIO()
            
            # Run joblib.dump in thread pool to prevent blocking the async event loop
            await asyncio.to_thread(joblib.dump, model_package, model_bytes)
            
            model_bytes.seek(0)
            model_content = model_bytes.read()
            logger.info(f"Model serialized successfully ({len(model_content) / 1024 / 1024:.2f} MB)")
            
            # Generate storage path
            storage_path = self._generate_model_path(model_name, version)
            
            # Upload to cloud storage
            await self.storage.upload_file(
                file_content=model_content,
                file_path=storage_path,
                content_type="application/octet-stream"
            )
            
            logger.info(f"Model with preprocessing pipeline saved to cloud storage: {storage_path}")
            return storage_path
        
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    async def load_model(self, model_path: str, return_package: bool = False) -> Any:
        """
        Load model from cloud storage with caching to prevent concurrent reloads.
        
        Args:
            model_path: Path to model in cloud storage
            return_package: If True, return complete package (model, preprocessor, etc.)
                          If False, return just the model (backward compatible)
            
        Returns:
            Loaded model object or complete package dict
        """
        try:
            # Check cache first (with lock to prevent race conditions)
            async with self._cache_lock:
                cache_key = f"{model_path}:{return_package}"
                if cache_key in self._model_cache:
                    cached_package, cache_time = self._model_cache[cache_key]
                    # Check if cache is still valid
                    if time.time() - cache_time < self._cache_timeout:
                        logger.debug(f"Model loaded from cache: {model_path}")
                        return cached_package
                    else:
                        # Cache expired, remove it
                        del self._model_cache[cache_key]
            
            # Not in cache or expired - load from storage
            # Use a timeout to prevent hanging on slow operations
            logger.info(f"Loading model from storage: {model_path}")
            
            try:
                # Download from cloud storage with timeout
                model_content = await asyncio.wait_for(
                    self.storage.download_file(model_path),
                    timeout=60.0  # 60 second timeout for download
                )
                
                # Deserialize model (run in thread pool to avoid blocking event loop)
                # Add timeout to prevent hanging on very large models
                logger.info("Deserializing model package (this may take a moment for large models)...")
                model_bytes = io.BytesIO(model_content)
                loaded = await asyncio.wait_for(
                    asyncio.to_thread(joblib.load, model_bytes),
                    timeout=120.0  # 120 second timeout for deserialization
                )
                
                # Handle both old format (just model) and new format (package dict)
                if isinstance(loaded, dict) and "model" in loaded:
                    logger.info(f"Model package loaded from cloud storage: {model_path}")
                    result = loaded if return_package else loaded["model"]
                else:
                    # Old format - just return the model
                    logger.info(f"Model (legacy format) loaded from cloud storage: {model_path}")
                    result = loaded
                
                # Cache the result (with lock)
                async with self._cache_lock:
                    cache_key = f"{model_path}:{return_package}"
                    self._model_cache[cache_key] = (result, time.time())
                    # Limit cache size to prevent memory issues (keep last 10 models)
                    if len(self._model_cache) > 10:
                        # Remove oldest entry
                        oldest_key = min(
                            self._model_cache.keys(),
                            key=lambda k: self._model_cache[k][1]
                        )
                        del self._model_cache[oldest_key]
                
                return result
            
            except asyncio.TimeoutError:
                logger.error(f"Timeout loading model {model_path} - operation took too long")
                raise TimeoutError(f"Model loading timeout: {model_path}")
        
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}", exc_info=True)
            raise
    
    async def delete_model(self, model_path: str) -> bool:
        """
        Delete model from cloud storage and clear cache.
        
        Args:
            model_path: Path to model in cloud storage
        
        Returns:
            True if successful, False otherwise
        """
        try:
            result = await self.storage.delete_file(model_path)
            
            # Clear cache entries for this model path
            async with self._cache_lock:
                keys_to_remove = [
                    key for key in self._model_cache.keys()
                    if key.startswith(f"{model_path}:")
                ]
                for key in keys_to_remove:
                    del self._model_cache[key]
            
            return result
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False
    
    def save_model_local(
        self,
        model: Any,
        file_path: str,
        preprocessor: Optional[Any] = None,
        feature_names: Optional[list] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Save model to local filesystem (for development/testing).
        
        Args:
            model: Trained model object
            file_path: Local file path
            preprocessor: Preprocessing pipeline to save with model
            feature_names: List of feature names expected by the model
            metadata: Additional metadata
        """
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create complete package
            model_package = {
                "model": model,
                "preprocessor": preprocessor,
                "feature_names": feature_names,
                "metadata": metadata or {},
                "saved_at": datetime.utcnow().isoformat()
            }
            
            joblib.dump(model_package, file_path)
            logger.info(f"Model with preprocessing pipeline saved locally: {file_path}")
        except Exception as e:
            logger.error(f"Error saving model locally: {e}")
            raise
    
    def load_model_local(self, file_path: str) -> Any:
        """
        Load model from local filesystem.
        
        Args:
            file_path: Local file path
            
        Returns:
            Loaded model object
        """
        try:
            model = joblib.load(file_path)
            logger.info(f"Model loaded locally: {file_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model locally: {e}")
            raise


# Global instance
model_storage = ModelStorage()

