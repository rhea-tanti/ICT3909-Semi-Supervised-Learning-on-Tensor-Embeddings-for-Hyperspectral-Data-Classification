from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

from src.utils.evaluation_metrics import calculate_evaluation_metrics, plot_confusion_matrix


class BaseModel(ABC):
    """Abstract base class for classification models."""

    def __init__(self, **kwargs):
        """Initialize the model with optional parameters."""
        self.model = None
        self.model_name = None
        self.params = kwargs


    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model on the given data."""
        pass


    @staticmethod
    def _preprocess_features(X: np.ndarray) -> np.ndarray:
        """Preprocess features."""
        if len(X.shape) > 2:
            return X.reshape(X.shape[0], -1)
        return X


    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Make predictions."""
        pass


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability estimates."""
        raise NotImplementedError("This model does not support probability predictions")


    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        # Make predictions
        y_pred = self.predict(X_test)
        return calculate_evaluation_metrics(y_test, y_pred)


    def plot_confusion_matrix(self, conf_matrix: np.ndarray, class_names: list, save_path: str,
                              split_index: int = None):
        """Plot and save confusion matrix."""
        title = f'Confusion Matrix - {self.model_name}'
        plot_confusion_matrix(
            conf_matrix=conf_matrix,
            class_names=class_names,
            save_path=save_path,
            title=title,
            split_index=split_index
        )
