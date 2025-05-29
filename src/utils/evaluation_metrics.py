import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def calculate_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate standardised evaluation metrics."""
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Calculate support (number of samples per class)
    conf_matrix_raw = confusion_matrix(y_true, y_pred)
    support_per_class = conf_matrix_raw.sum(axis=1)

    # Weighted metrics
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Normalised confusion matrix
    conf_matrix_norm = confusion_matrix(y_true, y_pred, normalize='true')

    return {
        'accuracy': accuracy,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': conf_matrix_norm
    }


def plot_confusion_matrix(
        conf_matrix: np.ndarray,
        class_names: List[str],
        save_path: str,
        title: str = 'Confusion Matrix',
        split_index: Optional[int] = None,
        figsize: tuple = (10, 8),
        cmap: str = 'Blues',
        dpi: int = 300
) -> None:
    """Plot and save a confusion matrix as an image."""
    # Create the figure
    plt.figure(figsize=figsize)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)

    full_title = title
    if split_index is not None:
        full_title += f' (Split {split_index})'

    plt.title(full_title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > threshold else "black")

    # save the plot
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close()


def evaluate_model_predictions(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
) -> Dict[str, Any]:
    """Evaluate a model."""
    # Make predictions
    y_pred = model.predict(X_test)
    return calculate_evaluation_metrics(y_test, y_pred)


def print_metrics_summary(
        metrics: Dict[str, Any],
        model_info: str = "",
        split_index: Optional[int] = None
) -> None:
    """Print summary of evaluation metrics."""
    header_parts = []
    if model_info:
        header_parts.append(model_info)
    if split_index is not None:
        header_parts.append(f"Split {split_index}")

    header = f"Results for {' - '.join(header_parts)}:" if header_parts else "Results:"

    print(f"\n{header}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Weighted Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Weighted Recall: {metrics['recall_weighted']:.4f}")
    print(f"  Weighted F1: {metrics['f1_weighted']:.4f}")


def serialize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert NumPy arrays to JSON-serializable format."""
    serialized = {}
    for key, value in metrics.items():
        if hasattr(value, 'tolist'):
            serialized[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32)):
            serialized[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            serialized[key] = float(value)
        else:
            serialized[key] = value
    return serialized
