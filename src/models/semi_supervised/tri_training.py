import numpy as np
from LAMDA_SSL.Algorithm.Classification.Tri_Training import Tri_Training as LAMDA_Tri_Training


class CustomTriTraining(LAMDA_Tri_Training):
    """Wrapper around LAMDA-SSL Tri_Training to address compatibility issues."""

    def predict_proba(self, X):
        num_classes = self.estimators[0].predict_proba(X[0:1]).shape[1]
        y_proba = np.full((X.shape[0], num_classes), 0, dtype=np.float64)

        for i in range(3):
            y_proba += self.estimators[i].predict_proba(X)

        return y_proba / 3
