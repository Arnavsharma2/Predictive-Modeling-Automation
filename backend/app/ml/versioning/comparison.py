"""
Model version comparison utilities.
"""
from typing import Dict, Any, List
from app.models.database.model_versions import ModelVersion


class ModelComparison:
    """Compare model versions."""
    
    @staticmethod
    def compare_versions(versions: List[ModelVersion]) -> Dict[str, Any]:
        """
        Compare multiple model versions.
        
        Args:
            versions: List of model versions to compare
            
        Returns:
            Dictionary with comparison results
        """
        if not versions:
            return {"error": "No versions to compare"}
        
        if len(versions) < 2:
            return {"error": "Need at least 2 versions to compare"}
        
        comparison = {
            "versions": [v.version for v in versions],
            "metrics_comparison": {},
            "feature_comparison": {},
            "hyperparameter_comparison": {}
        }
        
        # Compare performance metrics
        if all(v.performance_metrics for v in versions):
            metrics_comparison = {}
            all_metric_names = set()
            
            for v in versions:
                if v.performance_metrics:
                    all_metric_names.update(v.performance_metrics.keys())
            
            for metric_name in all_metric_names:
                values = []
                for v in versions:
                    if v.performance_metrics and metric_name in v.performance_metrics:
                        values.append({
                            "version": v.version,
                            "value": v.performance_metrics[metric_name]
                        })
                
                if values:
                    metrics_comparison[metric_name] = {
                        "values": values,
                        "best": max(values, key=lambda x: x["value"]) if values else None,
                        "worst": min(values, key=lambda x: x["value"]) if values else None
                    }
            
            comparison["metrics_comparison"] = metrics_comparison
        
        # Compare features
        if all(v.features for v in versions):
            all_features = set()
            for v in versions:
                if v.features:
                    all_features.update(v.features)
            
            feature_comparison = {}
            for feature in all_features:
                present_in = [v.version for v in versions if v.features and feature in v.features]
                feature_comparison[feature] = {
                    "present_in_versions": present_in,
                    "present_in_all": len(present_in) == len(versions)
                }
            
            comparison["feature_comparison"] = feature_comparison
        
        # Compare hyperparameters
        if all(v.hyperparameters for v in versions):
            all_hyperparams = set()
            for v in versions:
                if v.hyperparameters:
                    all_hyperparams.update(v.hyperparameters.keys())
            
            hyperparam_comparison = {}
            for hyperparam in all_hyperparams:
                values = []
                for v in versions:
                    if v.hyperparameters and hyperparam in v.hyperparameters:
                        values.append({
                            "version": v.version,
                            "value": v.hyperparameters[hyperparam]
                        })
                
                if values:
                    hyperparam_comparison[hyperparam] = {
                        "values": values,
                        "all_same": len(set(str(v["value"]) for v in values)) == 1
                    }
            
            comparison["hyperparameter_comparison"] = hyperparam_comparison
        
        return comparison
    
    @staticmethod
    def get_best_version_by_metric(
        versions: List[ModelVersion],
        metric_name: str,
        higher_is_better: bool = True
    ) -> ModelVersion:
        """
        Get the best version by a specific metric.
        
        Args:
            versions: List of versions to compare
            metric_name: Name of metric to compare
            higher_is_better: Whether higher values are better
            
        Returns:
            Best version
        """
        if not versions:
            raise ValueError("No versions provided")
        
        valid_versions = [
            v for v in versions
            if v.performance_metrics and metric_name in v.performance_metrics
        ]
        
        if not valid_versions:
            raise ValueError(f"No versions have metric {metric_name}")
        
        if higher_is_better:
            return max(valid_versions, key=lambda v: v.performance_metrics[metric_name])
        else:
            return min(valid_versions, key=lambda v: v.performance_metrics[metric_name])
    
    @staticmethod
    def summarize_version(version: ModelVersion) -> Dict[str, Any]:
        """
        Create a summary of a version.
        
        Args:
            version: Model version
            
        Returns:
            Summary dictionary
        """
        summary = {
            "id": version.id,
            "version": version.version,
            "is_active": version.is_active,
            "is_archived": version.is_archived,
            "created_at": version.created_at.isoformat() if version.created_at else None,
            "training_date": version.training_date.isoformat() if version.training_date else None
        }
        
        if version.performance_metrics:
            summary["performance_metrics"] = version.performance_metrics
        
        if version.dataset_size:
            summary["dataset_size"] = version.dataset_size
        
        if version.training_duration_seconds:
            summary["training_duration_seconds"] = version.training_duration_seconds
        
        if version.tags:
            summary["tags"] = version.tags
        
        return summary

