from LAMDA_SSL.Algorithm.Classification.Assemble import Assemble as LAMDA_Assemble
import numpy as np


class CustomAssemble(LAMDA_Assemble):
    """Custom Assemble model based on LAMDA_SSL's Assemble class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_mapping = None
        self.reverse_mapping = None
        self.n_classes = None
        self.verbose = True

    def fit(self, X, y, unlabeled_X):
        """Fit the Assemble model to the training data."""
        unique_classes = np.unique(y)
        self.class_mapping = {original: idx for idx, original in enumerate(unique_classes)}
        self.reverse_mapping = {idx: original for original, idx in self.class_mapping.items()}
        self.n_classes = len(unique_classes)

        y_remapped = np.array([self.class_mapping[label] for label in y])
        self.KNN.n_neighbors = min(3, len(y_remapped))

        return super().fit(X, y_remapped, unlabeled_X)

    def predict(self, X):
        """Predict labels."""
        y_pred_remapped = super().predict(X)
        return np.array([self.reverse_mapping[label] for label in y_pred_remapped])

    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.f:
            return np.zeros((X.shape[0], self.n_classes))

        y_proba = np.zeros((X.shape[0], self.n_classes))

        for i in range(len(self.w)):
            proba = self.f[i].predict_proba(X)
            if proba.shape[1] != self.n_classes:
                expanded_proba = np.zeros((X.shape[0], self.n_classes))
                for j in range(proba.shape[1]):
                    if j < self.n_classes:
                        expanded_proba[:, j] = proba[:, j]
                proba = expanded_proba
            y_proba += self.w[i] * proba

        return y_proba
