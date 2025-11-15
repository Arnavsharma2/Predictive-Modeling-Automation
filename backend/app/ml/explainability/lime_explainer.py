"""
LIME (Local Interpretable Model-agnostic Explanations) explainer.

LIME explains individual predictions by approximating the model locally
with an interpretable model.
"""
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import joblib
from pathlib import Path


class LimeExplainer:
    """
    LIME explainer for machine learning models.

    Provides local explanations for individual predictions by fitting
    an interpretable model in the neighborhood of the prediction.
    """

    def __init__(
        self,
        model_path: str,
        training_data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = 'classification'
    ):
        """
        Initialize LIME explainer.

        Args:
            model_path: Path to the trained model
            training_data: Training data used to understand feature distributions
            feature_names: List of feature names
            class_names: List of class names for classification
            mode: 'classification' or 'regression'
        """
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self.mode = mode

        # Convert training data
        if isinstance(training_data, pd.DataFrame):
            if feature_names is None:
                feature_names = training_data.columns.tolist()
            training_data = training_data.values

        self.training_data = training_data
        self.feature_names = feature_names or [f'feature_{i}' for i in range(training_data.shape[1])]
        self.class_names = class_names

        # Initialize LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=self.mode,
            verbose=False
        )

    def _load_model(self):
        """Load the trained model from disk."""
        return joblib.load(self.model_path)

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        """
        Prediction function for LIME.

        Args:
            X: Input data

        Returns:
            Predictions (probabilities for classification, values for regression)
        """
        if self.mode == 'classification':
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                # For models without predict_proba, return one-hot encoded predictions
                predictions = self.model.predict(X)
                n_classes = len(self.class_names) if self.class_names else 2
                proba = np.zeros((len(predictions), n_classes))
                for i, pred in enumerate(predictions):
                    proba[i, int(pred)] = 1.0
                return proba
        else:
            return self.model.predict(X)

    def explain_instance(
        self,
        instance: Union[pd.DataFrame, np.ndarray, List],
        num_features: int = 10,
        num_samples: int = 5000,
        top_labels: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single instance.

        Args:
            instance: Single instance to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples to generate for local model
            top_labels: Number of top predicted labels to explain (classification only)

        Returns:
            Dictionary containing:
            - prediction: Model prediction
            - explanation: Feature importance list
            - intercept: Local model intercept
            - score: R² score of local model
        """
        # Convert instance to proper format
        if isinstance(instance, list):
            instance_array = np.array(instance)
        elif isinstance(instance, pd.DataFrame):
            instance_array = instance.values[0] if len(instance) == 1 else instance.values
        elif isinstance(instance, np.ndarray):
            instance_array = instance.flatten() if instance.ndim > 1 else instance
        else:
            instance_array = np.array(instance)

        # Get model prediction
        if self.mode == 'classification':
            if hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict_proba(instance_array.reshape(1, -1))[0]
                predicted_class = int(np.argmax(prediction))
            else:
                predicted_class = int(self.model.predict(instance_array.reshape(1, -1))[0])
                prediction = np.zeros(len(self.class_names) if self.class_names else 2)
                prediction[predicted_class] = 1.0
        else:
            prediction = float(self.model.predict(instance_array.reshape(1, -1))[0])
            predicted_class = None

        # Generate LIME explanation
        if self.mode == 'classification':
            labels = top_labels if top_labels is not None else (predicted_class,)
            exp = self.explainer.explain_instance(
                instance_array,
                self._predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                labels=labels
            )

            # Get explanation for predicted class
            explanation_list = exp.as_list(label=predicted_class)
            local_pred = exp.local_pred[predicted_class] if hasattr(exp, 'local_pred') else None

        else:
            exp = self.explainer.explain_instance(
                instance_array,
                self._predict_fn,
                num_features=num_features,
                num_samples=num_samples
            )
            explanation_list = exp.as_list()
            local_pred = exp.local_pred[0] if hasattr(exp, 'local_pred') else None

        # Parse explanation into structured format
        feature_contributions = []
        for feature_desc, weight in explanation_list:
            # Extract feature name from description (e.g., "age <= 30" -> "age")
            feature_name = feature_desc.split()[0] if ' ' in feature_desc else feature_desc
            feature_contributions.append({
                'feature': feature_name,
                'description': feature_desc,
                'weight': float(weight)
            })

        return {
            'prediction': prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
            'predicted_class': predicted_class,
            'local_prediction': float(local_pred) if local_pred is not None else None,
            'feature_contributions': feature_contributions,
            'intercept': float(exp.intercept[predicted_class if self.mode == 'classification' else 0]),
            'score': float(exp.score) if hasattr(exp, 'score') else None,
            'feature_names': self.feature_names,
            'mode': self.mode
        }

    def explain_batch(
        self,
        instances: Union[pd.DataFrame, np.ndarray],
        num_features: int = 10,
        num_samples: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Generate LIME explanations for multiple instances.

        Args:
            instances: Multiple instances to explain
            num_features: Number of features to include in each explanation
            num_samples: Number of samples for local model

        Returns:
            List of explanation dictionaries
        """
        if isinstance(instances, pd.DataFrame):
            instances_array = instances.values
        else:
            instances_array = instances

        explanations = []
        for instance in instances_array:
            exp = self.explain_instance(
                instance,
                num_features=num_features,
                num_samples=num_samples
            )
            explanations.append(exp)

        return explanations

    def get_feature_importance_summary(
        self,
        instances: Union[pd.DataFrame, np.ndarray],
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Dict[str, float]:
        """
        Get aggregated feature importance across multiple instances.

        Args:
            instances: Instances to analyze
            num_features: Number of features per explanation
            num_samples: Number of samples for local model

        Returns:
            Dictionary mapping feature names to average absolute importance
        """
        explanations = self.explain_batch(instances, num_features, num_samples)

        # Aggregate weights
        feature_weights = {}
        feature_counts = {}

        for exp in explanations:
            for contrib in exp['feature_contributions']:
                feature = contrib['feature']
                weight = abs(contrib['weight'])

                if feature not in feature_weights:
                    feature_weights[feature] = 0
                    feature_counts[feature] = 0

                feature_weights[feature] += weight
                feature_counts[feature] += 1

        # Calculate averages
        avg_importance = {
            feature: feature_weights[feature] / feature_counts[feature]
            for feature in feature_weights
        }

        # Sort by importance
        return dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))
