"""
Experiment tracking service for ML training jobs.
"""
from typing import Dict, Any, Optional, List
import numpy as np
import mlflow
from contextlib import contextmanager

from app.ml.tracking.mlflow_client import MLflowClient
from app.core.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """
    High-level experiment tracking service.
    
    Provides convenient methods for tracking training experiments,
    logging metrics, parameters, and artifacts.
    """
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize experiment tracker.
        
        Args:
            tracking_uri: MLflow tracking URI
        """
        self.client = MLflowClient(tracking_uri)
    
    @contextmanager
    def track_experiment(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        log_model: bool = True,
        model: Optional[Any] = None,
        model_artifact_path: str = "model"
    ):
        """
        Context manager for tracking an experiment.
        
        Usage:
            with tracker.track_experiment("my_experiment", run_name="run_1") as run:
                tracker.log_params({"learning_rate": 0.01})
                tracker.log_metrics({"accuracy": 0.95})
                # ... training code ...
        
        Args:
            experiment_name: Name of the experiment
            run_name: Optional name for the run
            tags: Optional tags
            log_model: Whether to log the model at the end
            model: Model to log (if log_model is True)
            model_artifact_path: Path for model artifact
            
        Yields:
            Active run object
        """
        run = self.client.start_run(experiment_name, run_name, tags)
        try:
            yield run
            if log_model and model is not None:
                self.client.log_model(model, artifact_path=model_artifact_path)
            self.client.end_run("FINISHED")
        except Exception as e:
            logger.error(f"Error in experiment tracking: {e}", exc_info=True)
            self.client.end_run("FAILED")
            raise
    
    def log_training_config(
        self,
        config: Dict[str, Any],
        include_hyperparameters: bool = True
    ) -> None:
        """
        Log training configuration.
        
        Args:
            config: Training configuration dictionary
            include_hyperparameters: Whether to log hyperparameters separately
        """
        if include_hyperparameters and "hyperparameters" in config:
            hyperparams = config.pop("hyperparameters", {})
            self.client.log_params(hyperparams)
        
        # Log other config as tags or params
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)):
                self.client.set_tags({key: str(value)})
            elif isinstance(value, dict):
                # Flatten nested dicts
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (str, int, float, bool)):
                        self.client.log_params({f"{key}.{nested_key}": nested_value})
    
    def log_training_metrics(
        self,
        metrics: Dict[str, float],
        epoch: Optional[int] = None
    ) -> None:
        """
        Log training metrics.
        
        Args:
            metrics: Dictionary of metrics
            epoch: Optional epoch/step number
        """
        self.client.log_metrics(metrics, step=epoch)
    
    def log_model_artifacts(
        self,
        model: Any,
        preprocessor: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log model and related artifacts.
        
        Args:
            model: Trained model
            preprocessor: Optional preprocessor
            feature_names: Optional feature names
            metadata: Optional model metadata
        """
        # Log model (errors are handled gracefully in log_model)
        try:
            self.client.log_model(model, artifact_path="model")
        except Exception as e:
            logger.warning(f"Failed to log model artifact (non-critical): {e}")
        
        # Log preprocessor if available
        if preprocessor is not None:
            try:
                self.client.log_model(preprocessor, artifact_path="preprocessor")
            except Exception as e:
                logger.warning(f"Failed to log preprocessor artifact (non-critical): {e}")
        
        # Log feature names and metadata
        artifacts = {}
        if feature_names:
            artifacts["feature_names"] = feature_names
        if metadata:
            artifacts["metadata"] = metadata
        
        if artifacts:
            try:
                self.client.log_dict(artifacts, artifact_path="model_info")
            except Exception as e:
                logger.warning(f"Failed to log model info artifacts (non-critical): {e}")
    
    def log_evaluation_results(
        self,
        results: Dict[str, Any],
        dataset_name: str = "test"
    ) -> None:
        """
        Log evaluation results.
        
        Args:
            results: Dictionary of evaluation metrics
            dataset_name: Name of the dataset (e.g., "test", "validation")
        """
        # Filter out non-scalar values (lists, dicts, etc.) - MLflow only accepts scalars for metrics
        scalar_metrics = {}
        non_scalar_data = {}
        
        for key, value in results.items():
            # Check if value is a scalar (int, float, bool, or numeric string)
            if isinstance(value, (int, float, bool)):
                scalar_metrics[key] = float(value) if isinstance(value, bool) else value
            elif isinstance(value, (list, dict)):
                # Store non-scalar values separately to log as artifacts
                non_scalar_data[key] = value
            elif isinstance(value, (np.integer, np.floating)):
                # Handle numpy scalars
                scalar_metrics[key] = float(value)
            else:
                # Try to convert to float if possible
                try:
                    scalar_metrics[key] = float(value)
                except (ValueError, TypeError):
                    # If conversion fails, store as non-scalar
                    non_scalar_data[key] = value
        
        # Log scalar metrics
        if scalar_metrics:
            prefixed_metrics = {
                f"{dataset_name}_{key}": value
                for key, value in scalar_metrics.items()
            }
            self.client.log_metrics(prefixed_metrics)
        
        # Log non-scalar data as artifacts (e.g., confusion matrices, classification reports)
        if non_scalar_data:
            self.client.log_dict(non_scalar_data, artifact_path=f"{dataset_name}_detailed_metrics")
    
    def compare_runs(
        self,
        experiment_name: str,
        metric_name: str,
        top_n: int = 10,
        ascending: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Compare runs in an experiment.
        
        Args:
            experiment_name: Name of the experiment
            metric_name: Metric to compare
            top_n: Number of top runs to return
            ascending: True to sort ascending (for metrics like RMSE where lower is better),
                      False to sort descending (for metrics like accuracy, R² where higher is better)
            
        Returns:
            List of run summaries
        """
        experiment = self.client.get_experiment(experiment_name)
        if not experiment:
            return []
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=1000
        )
        
        # Filter and sort runs
        valid_runs = [
            run for run in runs
            if metric_name in run.data.metrics
        ]
        
        valid_runs.sort(
            key=lambda r: r.data.metrics[metric_name],
            reverse=not ascending
        )
        
        # Format results
        results = []
        for run in valid_runs[:top_n]:
            results.append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "metric_value": run.data.metrics[metric_name],
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags
            })
        
        return results
    
    def get_best_model_run(
        self,
        experiment_name: str,
        metric_name: str,
        ascending: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.
        
        Args:
            experiment_name: Name of the experiment
            metric_name: Metric to optimize
            ascending: True to minimize, False to maximize
            
        Returns:
            Dictionary with best run information
        """
        best_run = self.client.get_best_run(
            experiment_name,
            metric_name,
            ascending
        )
        
        if not best_run:
            return None
        
        return {
            "run_id": best_run.info.run_id,
            "run_name": best_run.info.run_name,
            "status": best_run.info.status,
            "metric_value": best_run.data.metrics[metric_name],
            "params": best_run.data.params,
            "metrics": best_run.data.metrics,
            "tags": best_run.data.tags
        }

