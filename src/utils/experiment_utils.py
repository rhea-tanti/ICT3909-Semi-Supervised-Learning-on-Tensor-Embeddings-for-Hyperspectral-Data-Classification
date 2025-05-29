import datetime
import json
import os
from typing import Dict, Any, Optional, Tuple
import re
import h5py
import numpy as np

from config.params import DATA_DIR, SHARED_EMBEDDINGS_DIR, SHARED_SPLITS_DIR

EXPERIMENTS_DIR = os.path.join(DATA_DIR, 'experiments')
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)


def create_experiment(
        dataset_name: str,
        embedding_type: str,
        model_name: str,
        experiment_type: str = "supervised",
        algorithm_type: Optional[str] = None,
        training_set: str = "small_train",
        split_index: Optional[int] = None,
        create_folders: bool = True,
        patch_size: int = 5,
        **params
) -> tuple[str, Dict[str, Any]]:
    """Create a new experiment."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"experiment_{timestamp}"

    # Create base experiment directory
    patch_dir = f"patch_{patch_size}"

    # For semi-supervised experiments
    model_name_with_set = model_name

    # For supervised experiments
    if experiment_type == "supervised":
        experiment_base = os.path.join(
            EXPERIMENTS_DIR,
            experiment_type,
            patch_dir,
            dataset_name,
            embedding_type,
            training_set,
            model_name_with_set,
            experiment_id
        )
    elif experiment_type == "semi_supervised":
        if algorithm_type is None:
            raise ValueError("algorithm_type must be specified for semi-supervised experiments")

        if training_set == 'train_5':
            labelled_samples = "5_labelled"
        elif training_set == 'train_10':
            labelled_samples = "10_labelled"
        elif training_set =='train_20':
            labelled_samples = "20_labelled"
        elif training_set == 'train_large':
            labelled_samples = "50_labelled"
        else:
            match = re.search(r'\d+', training_set)
            if match:
                labelled_samples = f"{match.group()}_labelled"
            else:
                labelled_samples = f"{training_set}_labelled"

        experiment_base = os.path.join(
            EXPERIMENTS_DIR,
            experiment_type,
            patch_dir,
            algorithm_type,
            dataset_name,
            embedding_type,
            labelled_samples,
            model_name_with_set,
            experiment_id
        )
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    if split_index is not None:
        experiment_path = os.path.join(experiment_base, f"split_{split_index}")
    else:
        experiment_path = experiment_base

    # Create directories
    if create_folders:
        os.makedirs(experiment_path, exist_ok=True)
        os.makedirs(experiment_base, exist_ok=True)

        if split_index is not None:
            for subdir in ['metrics', 'figures']:
                os.makedirs(os.path.join(experiment_path, subdir), exist_ok=True)

    # Create configuration
    config = {
        "timestamp": timestamp,
        "experiment_id": experiment_id,
        "dataset": dataset_name,
        "embedding_type": embedding_type,
        "experiment_type": experiment_type,
        "training_set": training_set,
        "created_at": datetime.datetime.now().isoformat(),
        "patch_size": patch_size,
        "artifacts": {}
    }

    if experiment_type == "semi_supervised" and algorithm_type:
        config["algorithm_type"] = algorithm_type

    config.update(params)

    # Save configuration
    if create_folders:
        config_path = os.path.join(experiment_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        if split_index is not None:
            base_config = config.copy()
            base_config['splits'] = {f"split_{split_index}": config}
            base_config_path = os.path.join(experiment_base, 'config.json')

            if os.path.exists(base_config_path):
                try:
                    with open(base_config_path, 'r') as f:
                        existing_config = json.load(f)

                    # Update splits information
                    if 'splits' not in existing_config:
                        existing_config['splits'] = {}
                    existing_config['splits'][f"split_{split_index}"] = config
                    base_config = existing_config
                except (json.JSONDecodeError, IOError):
                    pass

            with open(base_config_path, 'w') as f:
                json.dump(base_config, f, indent=2)

    return experiment_path, config


def update_experiment_config(experiment_dir: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update experiment configuration."""
    config_path = os.path.join(experiment_dir, 'config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Update configuration
    for key, value in updates.items():
        if key == 'artifacts' and isinstance(value, dict) and 'artifacts' in config:
            config['artifacts'].update(value)
        else:
            config[key] = value

    # Save updated configuration
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config


def find_latest_split_file(dataset_name: str, embedding_type: str, patch_size: int = 5) -> Optional[str]:
    """Find the latest split file."""
    patch_dir = f"patch_{patch_size}"
    splits_dir = os.path.join(SHARED_SPLITS_DIR, patch_dir, dataset_name)

    if not os.path.exists(splits_dir):
        return None

    # Find all matching split files
    files = [f for f in os.listdir(splits_dir) if
             f.endswith('.json') and f.startswith(f"{dataset_name}_{embedding_type}_splits_")]

    if not files:
        return None

    # Sort by timestamp
    files.sort(key=lambda f: f.split('_')[-1].split('.')[0], reverse=True)

    return os.path.join(splits_dir, files[0])


def load_splits_file(
        dataset_name: str,
        embedding_type: str,
        split_file: str,
        split_index: int = 0,
        patch_size: int = 5
) -> Tuple[Dict, Dict[str, Any], str]:
    """Load splits directly from split file in shared splits directory."""
    if os.path.isabs(split_file) and os.path.exists(split_file):
        split_path = split_file
    else:
        patch_dir = f"patch_{patch_size}"
        splits_dir = os.path.join(SHARED_SPLITS_DIR, patch_dir, dataset_name)
        split_path = os.path.join(splits_dir, split_file)

    if not os.path.exists(split_path):
        raise ValueError(f"Split file {split_file} not found")

    # Load split file
    with open(split_path, 'r') as f:
        split_data = json.load(f)

    metadata = split_data['metadata']
    split_key = f'split_{split_index}'
    split = split_data['splits'][split_key]

    # Get embedding path
    embedding_path = metadata.get('embedding_path')
    if not embedding_path or not os.path.exists(embedding_path):
        embedding_file = metadata.get('embedding_file')
        if embedding_file:
            shared_path = os.path.join(SHARED_EMBEDDINGS_DIR, dataset_name, embedding_file)
            if os.path.exists(shared_path):
                embedding_path = shared_path

    if not embedding_path or not os.path.exists(embedding_path):
        raise ValueError(f"Embedding file referenced in {split_file} could not be found")

    return split, metadata, embedding_path


def load_embeddings_for_split(
        split: Dict,
        embedding_file: str
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    """Load embeddings for a specific split."""
    # Load embeddings
    with h5py.File(embedding_file, 'r') as hf:
        all_embeddings = hf['embeddings'][:]
        all_labels = hf['labels'][:]

        # Get dataset info and metadata
        dataset_info = {}
        for key, value in hf['dataset_info'].attrs.items():
            dataset_info[key] = value

        metadata = {}
        for key, value in hf['metadata'].attrs.items():
            metadata[key] = value

    # Create datasets for this split
    datasets = {
        'train_large': {
            'embeddings': all_embeddings[split['train_large']],
            'labels': all_labels[split['train_large']]
        },
        'train_10': {
            'embeddings': all_embeddings[split['train_10']],
            'labels': all_labels[split['train_10']]
        },
        'train_20': {
            'embeddings': all_embeddings[split['train_20']],
            'labels': all_labels[split['train_20']]
        },
        'train_5': {
            'embeddings': all_embeddings[split['train_5']],
            'labels': all_labels[split['train_5']]
        },
        'test': {
            'embeddings': all_embeddings[split['test']],
            'labels': all_labels[split['test']]
        }
    }

    # Create metadata dict
    full_metadata = {
        'embedding': metadata,
        'dataset_info': dataset_info,
        'split_info': {
            'large_train_size': len(split['train_large']),
            'train_5_size': len(split['train_5']),
            'train_10_size': len(split['train_10']),
            'train_20_size': len(split['train_20']),
            'test_size': len(split['test']),
            'split_seed': split.get('split_seed', None)
        }
    }
    return datasets, full_metadata


def load_datasets(dataset_name: str,
        embedding_type: str,
        split_file: str,
        split_index: int = 0
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    """Prepare datasets from a split file."""
    split, metadata, embedding_path = load_splits_file(
        dataset_name=dataset_name,
        embedding_type=embedding_type,
        split_file=split_file,
        split_index=split_index
    )
    datasets, metadata_from_embedding = load_embeddings_for_split(split, embedding_path)
    full_metadata = {
        **metadata_from_embedding,
        'split_metadata': metadata,
        'splits_file': split_file,
        'embedding_path': embedding_path
    }

    return datasets, full_metadata
