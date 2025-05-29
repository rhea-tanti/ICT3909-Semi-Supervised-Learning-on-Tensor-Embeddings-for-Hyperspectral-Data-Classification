from os import path
from typing import Tuple, Dict
import numpy as np
import scipy.io as sio

from config.datasets import DATASETS
from config.params import RAW_DATA_DIR


def get_dataset_info(hypercube, labels) -> Dict[str, int]:
    """Get information about the dataset."""
    return {
        'height': hypercube.shape[0],
        'width': hypercube.shape[1],
        'bands': hypercube.shape[2],
        'n_classes': len(np.unique(labels)) - 1,  # Excluding background
        'total_pixels': hypercube.shape[0] * hypercube.shape[1]
    }


def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load hyperspectral data and ground truth labels from .mat files."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(DATASETS.keys())}")

    data_file, gt_file, data_key, gt_key = DATASETS[dataset_name]
    data_file_path = path.join(RAW_DATA_DIR, data_file)
    gt_file_path = path.join(RAW_DATA_DIR, gt_file)

    data = sio.loadmat(data_file_path)
    ground_truth = sio.loadmat(gt_file_path)

    hypercube = data[data_key].astype(np.float32)
    labels = ground_truth[gt_key].astype(np.int64)

    return hypercube, labels


def normalise_data(data: np.ndarray) -> np.ndarray:
    """Normalise hyperspectral data using min-max scaling to range [0, 1]."""
    # Calculate min and max along spatial dimensions (height, width)
    data_min = np.min(data, axis=(0, 1), keepdims=True)
    data_max = np.max(data, axis=(0, 1), keepdims=True)

    return (data - data_min) / (data_max - data_min + 1e-8)
