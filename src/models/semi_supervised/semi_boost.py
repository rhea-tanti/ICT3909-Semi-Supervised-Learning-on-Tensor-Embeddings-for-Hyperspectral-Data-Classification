from collections import Counter
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn import neighbors
from sklearn.base import ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np


class CustomSemiBoost(InductiveEstimator, ClassifierMixin):
    """SemiBoost implementation for multilabel classification."""

    def __init__(self, base_estimator=None, n_neighbors=9, n_jobs=None, T=100,
                 sample_percent=0.05, min_confidence=0.75, class_balance=True, max_per_class=None,
                 sigma_percentile=90, similarity_kernel='rbf', gamma=None,
                 ensemble_size=5, diversity_weight=0.3,
                 evaluation=None, verbose=False, file=None):
        self.base_estimator = base_estimator
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.T = T
        self.sample_percent = sample_percent
        self.min_confidence = min_confidence
        self.class_balance = class_balance
        self.max_per_class = max_per_class
        self.sigma_percentile = sigma_percentile
        self.similarity_kernel = similarity_kernel
        self.gamma = gamma
        self.ensemble_size = ensemble_size
        self.diversity_weight = diversity_weight
        self.evaluation = evaluation
        self.verbose = verbose
        self.file = file
        self.models = []
        self.model_weights = []
        self.classes_ = None
        self._estimator_type = "classifier"

    def _build_similarity_matrix(self, X):
        """Build a similarity matrix based on the specified kernel."""
        if self.similarity_kernel == 'knn':
            nn = neighbors.NearestNeighbors(n_neighbors=min(self.n_neighbors, X.shape[0] - 1),
                                            n_jobs=self.n_jobs)
            nn.fit(X)
            graph = nn.kneighbors_graph(X, mode='distance')
            graph.data = np.exp(-graph.data ** 2 / 0.5)
            return graph.toarray()
        elif self.similarity_kernel == 'linear':
            return linear_kernel(X, X)
        elif self.similarity_kernel == 'rbf':
            gamma = self.gamma or 5.0 / X.shape[1]
            return rbf_kernel(X, X, gamma)
        else:
            gamma = self.gamma
            if gamma is None:
                from scipy.spatial.distance import pdist
                dists = pdist(X, 'euclidean')
                gamma = 1.0 / (2 * np.percentile(dists, min(self.sigma_percentile, 50)) ** 2)
            return rbf_kernel(X, X, gamma)

    def fit(self, X, y, unlabeled_X):
        """Fit SemiBoost model."""
        X, y = check_X_y(X, y)
        unlabeled_X = check_array(unlabeled_X)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if self.verbose:
            print(f"Starting SemiBoost with {n_classes} classes")
            print(f"Initial labeled: {X.shape[0]}, Unlabeled: {unlabeled_X.shape[0]}")

        if X.shape[0] > 10:
            from sklearn.model_selection import StratifiedShuffleSplit
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(splitter.split(X, y))
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = X[:1], y[:1]

        model_history = []
        X_train_current = X_train.copy()
        y_train_current = y_train.copy()

        if self.verbose:
            print("Computing similarity matrix...")

        all_X = np.vstack((X, unlabeled_X))
        S = self._build_similarity_matrix(all_X)

        n_labeled = X.shape[0]
        n_unlabeled = unlabeled_X.shape[0]
        labeled_indices = list(range(n_labeled))
        unlabeled_indices = list(range(n_labeled, n_labeled + n_unlabeled))

        for t in range(self.T):
            if self.verbose:
                print(f"\nIteration {t + 1}/{self.T}")
                print(f"Training set size: {len(y_train_current)}")

            model = self._clone_base_estimator()
            try:
                model.fit(X_train_current, y_train_current)
                val_acc = np.mean(model.predict(X_val) == y_val) if len(X_val) > 1 else 0
                model_history.append((model, X_train_current, y_train_current, val_acc))
            except Exception as e:
                if self.verbose:
                    print(f"Error during model training: {str(e)}")
                continue

            remaining_unlabeled = [i for i in unlabeled_indices if i not in labeled_indices]
            if len(remaining_unlabeled) == 0:
                if self.verbose:
                    print("No unlabeled data remaining.")
                break

            X_unlabeled_current = all_X[remaining_unlabeled]

            try:
                proba_unlabeled = model.predict_proba(X_unlabeled_current)
            except (AttributeError, NotImplementedError):
                preds = model.predict(X_unlabeled_current)
                proba_unlabeled = np.zeros((len(preds), len(self.classes_)))
                for i, pred in enumerate(preds):
                    idx = np.where(self.classes_ == pred)[0][0]
                    proba_unlabeled[i, idx] = 1.0

            max_proba = np.max(proba_unlabeled, axis=1)
            pred_classes = np.array([self.classes_[i] for i in np.argmax(proba_unlabeled, axis=1)])

            high_conf_mask = max_proba >= self.min_confidence
            if np.sum(high_conf_mask) == 0:
                if self.verbose:
                    print(f"No samples with confidence >= {self.min_confidence}")
                continue

            high_conf_indices = np.where(high_conf_mask)[0]
            high_conf_proba = max_proba[high_conf_mask]
            high_conf_classes = pred_classes[high_conf_mask]
            high_conf_remaining = [remaining_unlabeled[i] for i in high_conf_indices]

            similarity_weighted_confidence = np.zeros(len(high_conf_indices))

            for i, (idx, conf, pred_class) in enumerate(zip(high_conf_remaining, high_conf_proba, high_conf_classes)):
                sim_to_labeled = S[idx, labeled_indices]
                same_class_indices = [j for j, label in enumerate(y_train_current) if label == pred_class]
                diff_class_indices = [j for j, label in enumerate(y_train_current) if label != pred_class]

                if len(same_class_indices) > 0:
                    avg_sim_same = np.mean(sim_to_labeled[same_class_indices])
                    if len(diff_class_indices) > 0:
                        avg_sim_diff = np.mean(sim_to_labeled[diff_class_indices])
                        similarity_weighted_confidence[i] = conf * (avg_sim_same / (avg_sim_diff + 1e-10))
                    else:
                        similarity_weighted_confidence[i] = conf * avg_sim_same
                else:
                    similarity_weighted_confidence[i] = conf * 0.5

            if self.class_balance:
                class_counts = Counter(y_train_current)
                max_count = max(class_counts.values())
                class_weights = {cls: max_count / (count + 1) for cls, count in class_counts.items()}

                for cls in self.classes_:
                    if cls not in class_weights:
                        class_weights[cls] = max_count * 2

                for i, cls in enumerate(high_conf_classes):
                    similarity_weighted_confidence[i] *= class_weights.get(cls, 1.0)

            n_to_add = min(max(1, int(self.sample_percent * n_unlabeled)), len(high_conf_indices))

            if n_to_add > 0:
                selected_indices = []

                if self.class_balance and self.max_per_class:
                    class_indices = {}
                    for i, cls in enumerate(high_conf_classes):
                        if cls not in class_indices:
                            class_indices[cls] = []
                        class_indices[cls].append(i)

                    for cls, indices in class_indices.items():
                        if not indices:
                            continue
                        sorted_indices = sorted(indices, key=lambda i: similarity_weighted_confidence[i], reverse=True)
                        n_from_class = min(self.max_per_class, len(sorted_indices))
                        selected_indices.extend([indices[i] for i in sorted_indices[:n_from_class]])

                    if len(selected_indices) < n_to_add:
                        remaining_indices = list(set(range(len(high_conf_indices))) - set(selected_indices))
                        remaining_indices = sorted(remaining_indices, key=lambda i: similarity_weighted_confidence[i],
                                                   reverse=True)
                        additional_needed = n_to_add - len(selected_indices)
                        selected_indices.extend(remaining_indices[:additional_needed])
                else:
                    selected_indices = np.argsort(similarity_weighted_confidence)[-n_to_add:]

                selected_dataset_indices = [high_conf_remaining[i] for i in selected_indices]
                X_new = all_X[selected_dataset_indices]
                y_new = high_conf_classes[selected_indices]

                X_train_current = np.vstack((X_train_current, X_new))
                y_train_current = np.append(y_train_current, y_new)
                labeled_indices.extend(selected_dataset_indices)

                if self.verbose:
                    print(f"Added {len(selected_indices)} samples")
            else:
                if self.verbose:
                    print("No samples added")
                if t > 0 and t % 5 == 0:
                    self.min_confidence = max(0.5, self.min_confidence * 0.9)
                    if self.verbose:
                        print(f"Lowering confidence threshold to {self.min_confidence:.2f}")

        self._select_ensemble(model_history, X_val, y_val)
        return self

    def _select_ensemble(self, model_history, X_val, y_val):
        """Select the best models for the ensemble based on validation accuracy and class balance."""
        if len(model_history) == 0:
            raise ValueError("No models trained successfully")

        ensemble_size = min(self.ensemble_size, len(model_history))
        model_scores = []

        for i, (model, _, y_train, val_acc) in enumerate(model_history):
            class_counts = Counter(y_train)
            if len(class_counts) < len(self.classes_):
                balance_score = -np.inf
            else:
                counts = np.array([class_counts.get(cls, 0) for cls in self.classes_])
                cv = np.std(counts) / (np.mean(counts) + 1e-10)
                balance_score = -cv

            score = 0.5 * balance_score + 0.5 * val_acc
            model_scores.append((i, score, model))

        model_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [model_scores[0][0]]
        selected_models = [model_scores[0][2]]
        remaining_candidates = model_scores[1:]

        for _ in range(1, ensemble_size):
            if not remaining_candidates:
                break

            diversity_scores = []
            for idx, score, model in remaining_candidates:
                if len(X_val) > 1:
                    preds = model.predict(X_val)
                    diversity = 0
                    for sel_model in selected_models:
                        sel_preds = sel_model.predict(X_val)
                        disagreement = np.mean(preds != sel_preds)
                        diversity += disagreement
                    diversity /= len(selected_models)
                else:
                    diversity = 0.5

                combined_score = (1 - self.diversity_weight) * score + self.diversity_weight * diversity
                diversity_scores.append((idx, combined_score, model))

            diversity_scores.sort(key=lambda x: x[1], reverse=True)
            best_idx, _, best_model = diversity_scores[0]

            selected_indices.append(best_idx)
            selected_models.append(best_model)
            remaining_candidates = [c for c in remaining_candidates if c[0] != best_idx]

        if len(X_val) > 1:
            weights = []
            for model in selected_models:
                acc = np.mean(model.predict(X_val) == y_val)
                weights.append(max(0.1, acc))
            weights = np.array(weights) / sum(weights)
        else:
            weights = np.ones(len(selected_models)) / len(selected_models)

        self.models = selected_models
        self.model_weights = weights

        if self.verbose:
            print(f"\nSelected {len(self.models)} models for ensemble")

    def _clone_base_estimator(self):
        """Clone the base estimator to create a new instance."""
        if hasattr(self.base_estimator, "get_params"):
            return type(self.base_estimator)(**self.base_estimator.get_params())
        else:
            return type(self.base_estimator)()

    def predict_proba(self, X):
        """Predict class probabilities."""
        check_is_fitted(self, ["models", "model_weights", "classes_"])
        X = check_array(X)

        if len(self.models) == 0:
            return np.ones((X.shape[0], len(self.classes_))) / len(self.classes_)

        y_proba = np.zeros((X.shape[0], len(self.classes_)))

        for model, weight in zip(self.models, self.model_weights):
            try:
                model_proba = model.predict_proba(X)
                if model_proba.shape[1] != len(self.classes_):
                    fixed_proba = np.zeros((X.shape[0], len(self.classes_)))
                    for j in range(min(model_proba.shape[1], len(self.classes_))):
                        fixed_proba[:, j] = model_proba[:, j]
                    model_proba = fixed_proba
                y_proba += weight * model_proba
            except (AttributeError, NotImplementedError):
                preds = model.predict(X)
                model_proba = np.zeros((len(preds), len(self.classes_)))
                for i, pred in enumerate(preds):
                    try:
                        idx = np.where(self.classes_ == pred)[0][0]
                        model_proba[i, idx] = 1.0
                    except IndexError:
                        pass
                y_proba += weight * model_proba

        row_sums = y_proba.sum(axis=1)
        y_proba[row_sums > 0] /= row_sums[row_sums > 0].reshape(-1, 1)
        return y_proba

    def predict(self, X):
        """Predict class labels."""
        check_is_fitted(self, ["models", "model_weights", "classes_"])
        X = check_array(X)
        y_proba = self.predict_proba(X)
        return self.classes_[np.argmax(y_proba, axis=1)]

    def evaluate(self, X, y=None):
        """Evaluate the model on the provided data."""
        check_is_fitted(self, ["models", "model_weights", "classes_"])
        X = check_array(X)

        self.y_pred = self.predict(X)
        self.y_score = self.predict_proba(X)

        if self.evaluation is None:
            return None
        elif isinstance(self.evaluation, (list, tuple)):
            performance = []
            for eval_metric in self.evaluation:
                score = eval_metric.scoring(y, self.y_pred, self.y_score)
                if self.verbose and self.file is not None:
                    print(score, file=self.file)
                performance.append(score)
            self.performance = performance
            return performance
        elif isinstance(self.evaluation, dict):
            performance = {}
            for key, eval_metric in self.evaluation.items():
                performance[key] = eval_metric.scoring(y, self.y_pred, self.y_score)
                if self.verbose and self.file is not None:
                    print(key, ' ', performance[key], file=self.file)
            self.performance = performance
            return performance
        else:
            performance = self.evaluation.scoring(y, self.y_pred, self.y_score)
            if self.verbose and self.file is not None:
                print(performance, file=self.file)
            self.performance = performance
            return performance
