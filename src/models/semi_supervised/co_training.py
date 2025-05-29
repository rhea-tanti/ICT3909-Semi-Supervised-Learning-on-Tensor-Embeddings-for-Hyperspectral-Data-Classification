from LAMDA_SSL.Algorithm.Classification.Co_Training import Co_Training as LAMDA_Co_Training
import numpy as np


class CustomCoTraining(LAMDA_Co_Training):
    """Custom Co-Training class based on LAMDA-SSL Co-Training implementation."""
    def __init__(self, base_estimator, base_estimator_2, p=5, n=5, k=150, s=80,
                 random_state=42, threshold=0.75, binary=False, evaluation=None, verbose=False, file=None):
        # Pass all parameters to LAMDA Co_Training including binary
        super().__init__(
            base_estimator=base_estimator,
            base_estimator_2=base_estimator_2,
            p=p,
            n=n,
            k=k,
            s=s,
            random_state=random_state,
            threshold=threshold,
            binary=binary,
            evaluation=evaluation,
            verbose=verbose,
            file=file
        )
        self.class_mapping = None
        self.reverse_mapping = None
        self.n_classes = None

    def fit(self, X, y, unlabeled_X, X_2=None, unlabeled_X_2=None):
        """Fit the model using labeled and unlabeled data."""
        unique_classes = np.unique(y)
        self.class_mapping = {original: idx for idx, original in enumerate(unique_classes)}
        self.reverse_mapping = {idx: original for original, idx in self.class_mapping.items()}
        self.n_classes = len(unique_classes)

        y_remapped = np.array([self.class_mapping[label] for label in y])
        return super().fit(X, y_remapped, unlabeled_X, X_2, unlabeled_X_2)

    def predict(self, X, X_2=None):
        """Predict labels."""
        y_pred_remapped = super().predict(X, X_2)
        return np.array([self.reverse_mapping[label] for label in y_pred_remapped])

    def predict_proba(self, X, X_2=None):
        """Predict probability estimates."""
        if X_2 is None:
            if isinstance(X, (list, tuple)):
                X, X_2 = X[0], X[1]
            else:
                X_2 = X

        y1_proba = self.base_estimator.predict_proba(X)
        y2_proba = self.base_estimator_2.predict_proba(X_2)

        if y1_proba.shape[1] != self.n_classes:
            y1_proba_fixed = np.zeros((X.shape[0], self.n_classes))
            for j in range(min(y1_proba.shape[1], self.n_classes)):
                y1_proba_fixed[:, j] = y1_proba[:, j]
            y1_proba = y1_proba_fixed

        if y2_proba.shape[1] != self.n_classes:
            y2_proba_fixed = np.zeros((X.shape[0], self.n_classes))
            for j in range(min(y2_proba.shape[1], self.n_classes)):
                y2_proba_fixed[:, j] = y2_proba[:, j]
            y2_proba = y2_proba_fixed

        return (y1_proba + y2_proba) / 2
