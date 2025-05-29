import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base_models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest classifier using sklearn implementation."""

    def __init__(self, **kwargs):
        """Initialise Random Forest model."""
        super().__init__(**kwargs)
        self.model_name = "RandomForest"

        model_params = {
            'random_state': 42,
        }
        model_params.update(kwargs)
        self.model = RandomForestClassifier(**model_params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, sample_weight=None) -> None:
        """Train the Random Forest model."""
        X_train = self._preprocess_features(X_train)

        # Pass sample_weight to the underlying model if  provided
        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        X = self._preprocess_features(X)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability estimates for samples."""
        X = self._preprocess_features(X)

        return self.model.predict_proba(X)
