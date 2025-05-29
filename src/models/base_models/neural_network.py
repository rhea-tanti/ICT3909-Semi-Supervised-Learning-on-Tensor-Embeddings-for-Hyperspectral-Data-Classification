import numpy as np
from sklearn.neural_network import MLPClassifier

from src.models.base_models.base_model import BaseModel


class NeuralNetworkModel(BaseModel):
    """Neural Network classifier using sklearn's MLPClassifier."""

    def __init__(self, **kwargs):
        """Initialise Neural Network model."""
        super().__init__(**kwargs)
        self.model_name = "NeuralNetwork"

        model_params = {
            'random_state': 42,
            'max_iter': 1000,
        }
        model_params.update(kwargs)
        self.model = MLPClassifier(**model_params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, sample_weight=None) -> None:
        """Train the Neural Network model."""
        X_train = self._preprocess_features(X_train)

        # Ignore sample_weight (not supported by MLPClassifier)
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        X = self._preprocess_features(X)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability estimates for samples."""
        X = self._preprocess_features(X)

        return self.model.predict_proba(X)
