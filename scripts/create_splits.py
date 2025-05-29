import argparse
import datetime
import json
import os
import sys
import h5py
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import (
    SHARED_EMBEDDINGS_DIR, SHARED_SPLITS_DIR, DEFAULT_PATCH_SIZE
)


def load_embeddings(embedding_path):
    """Load embeddings and labels from HDF5 file."""
    with h5py.File(embedding_path, 'r') as hf:
        embeddings = hf['embeddings'][:]
        labels = hf['labels'][:]
        spatial_shape = hf['spatial_shape'][:]

        # Get dataset info
        dataset_info = {}
        for key, value in hf['dataset_info'].attrs.items():
            dataset_info[key] = value

        # Get metadata
        metadata = {}
        for key, value in hf['metadata'].attrs.items():
            metadata[key] = value

    return embeddings, labels, spatial_shape, dataset_info, metadata


def create_dataset_splits(embeddings, labels, n_splits=10, seed=42):
    """
    Create dataset splits for supervised and semi-supervised learning."""
    # Exclude background class (label 0)
    unique_classes = np.unique(labels)
    if 0 in unique_classes:
        unique_classes = unique_classes[unique_classes != 0]

    rng = np.random.RandomState(seed)
    all_splits = {}

    for split_idx in range(n_splits):
        split_seed = rng.randint(0, 10000)  
        split_rng = np.random.RandomState(split_seed)

        train_large_indices = []
        train_20_indices = []
        train_10_indices = []
        train_5_indices = []
        test_indices = []

        # Process each class
        for cls in unique_classes:
            # Get indices
            cls_indices = np.where(labels == cls)[0]
            n_samples = len(cls_indices)

            # Determine number of training samples
            if n_samples < 50:
                # If fewer than 50, use 80% for training
                n_train_large = int(0.8 * n_samples)
                print(
                    f"Warning: Class {cls} has only {n_samples} samples, using {n_train_large} for large training set")
            else:
                n_train_large = 50

            # Shuffle indices
            split_rng.shuffle(cls_indices)

            # Select training and test indices
            cls_train_large = cls_indices[:n_train_large]
            cls_test = cls_indices[n_train_large:]

            # Select train_20 set (subset of large, includes train_10)
            n_train_20 = min(20, n_train_large)
            cls_train_20 = cls_train_large[:n_train_20]

            # Select train_10 set (subset of train_20)
            n_train_10 = min(10, n_train_20)
            cls_train_10 = cls_train_20[:n_train_10]

            # Select small training set (subset of train_10)
            n_train_5 = min(5, n_train_10)
            cls_train_5 = cls_train_10[:n_train_5]

            # Add to overall indices
            train_large_indices.extend(cls_train_large)
            train_20_indices.extend(cls_train_20)
            train_10_indices.extend(cls_train_10)
            train_5_indices.extend(cls_train_5)
            test_indices.extend(cls_test)

        train_large_indices = np.array(train_large_indices)
        train_20_indices = np.array(train_20_indices)
        train_10_indices = np.array(train_10_indices)
        train_5_indices = np.array(train_5_indices)
        test_indices = np.array(test_indices)

        split_rng.shuffle(train_large_indices)
        split_rng.shuffle(train_20_indices)
        split_rng.shuffle(train_10_indices)
        split_rng.shuffle(train_5_indices)
        split_rng.shuffle(test_indices)

        all_splits[f'split_{split_idx}'] = {
            'train_large': train_large_indices.tolist(),
            'train_20': train_20_indices.tolist(),
            'train_10': train_10_indices.tolist(),
            'train_5': train_5_indices.tolist(),
            'test': test_indices.tolist(),
            'split_seed': int(split_seed)
        }

    return all_splits


def print_summary(args, splits_path, timestamp, labels, all_splits):
    """Print summary information about the created splits."""
    print(f"Dataset splits saved to {splits_path}")
    print(f"Timestamp: {timestamp}")
    print(f"\nTo use these splits in training, run:")
    print(f"python src/scripts/train_supervised.py --dataset {args.dataset} --embedding_type {args.embedding_type} "
          f"--split_file {os.path.basename(splits_path)}  --patch_size {args.patch_size} ")

    # Print summary statistics
    print("\nSummary statistics:")
    print(f"Total samples: {len(labels)}")

    # Calculate class distribution
    unique_classes = np.unique(labels)
    if 0 in unique_classes:
        unique_classes = unique_classes[unique_classes != 0]

    print("\nClass distribution:")
    for cls in unique_classes:
        cls_count = np.sum(labels == cls)
        print(f"  Class {cls}: {cls_count} samples")

    # Sample split statistics
    print("\nSample split statistics (split_0):")
    split = all_splits['split_0']
    print(f"  Large training set: {len(split['train_large'])} samples")
    print(f"  20-sample training set: {len(split['train_20'])} samples")
    print(f"  10-sample training set: {len(split['train_10'])} samples")
    print(f"  5-sample training set: {len(split['train_5'])} samples")
    print(f"  Test set: {len(split['test'])} samples")


def save_splits(args, all_splits, embedding_path):
    """Save dataset splits to a JSON file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save splits in the shared splits directory
    patch_dir = f"patch_{args.patch_size}"
    dataset_splits_dir = os.path.join(SHARED_SPLITS_DIR, patch_dir, args.dataset)
    os.makedirs(dataset_splits_dir, exist_ok=True)

    splits_filename = f"{args.dataset}_{args.embedding_type}_splits_{timestamp}.json"
    splits_path = os.path.join(dataset_splits_dir, splits_filename)

    # Prepare metadata
    metadata_dict = {
        'dataset': args.dataset,
        'embedding_type': args.embedding_type,
        'embedding_file': os.path.basename(embedding_path),
        'embedding_path': embedding_path,
        'n_splits': args.n_splits,
        'samples_large': len(all_splits['split_0']['train_large']),
        'samples_20': len(all_splits['split_0']['train_20']),
        'samples_10': len(all_splits['split_0']['train_10']),
        'samples_5': len(all_splits['split_0']['train_5']),
        'seed': args.seed,
        'timestamp': timestamp
    }

    with open(splits_path, 'w') as f:
        json.dump({
            'metadata': metadata_dict,
            'splits': all_splits
        }, f, indent=2)

    return splits_path, timestamp


def locate_embedding_file(args):
    """Locate the embedding file based on provided arguments."""
    if args.embedding_file:
        embedding_path = os.path.join(SHARED_EMBEDDINGS_DIR, args.dataset, args.embedding_file)
        if not os.path.exists(embedding_path):
            raise ValueError(f"Embedding file not found at {embedding_path}")

    else:
        # Find most recent embedding in shared directory
        patch_dir = f"patch_{args.patch_size}"
        embedding_dir = os.path.join(SHARED_EMBEDDINGS_DIR, patch_dir, args.dataset)

        if not os.path.exists(embedding_dir):
            raise ValueError(f"No embeddings directory found for {args.dataset} at {embedding_dir}")

        files = [f for f in os.listdir(embedding_dir) if
                 f.endswith('.h5') and f'{args.dataset}_{args.embedding_type}' in f]

        if not files:
            raise ValueError(f"No {args.embedding_type} embedding files found for {args.dataset}")

        # Sort by timestamp
        files.sort(key=lambda f: f.split('_')[-1].split('.')[0], reverse=True)
        embedding_path = os.path.join(embedding_dir, files[0])

    return embedding_path


def main():
    """Main function to create dataset splits."""
    parser = argparse.ArgumentParser(description='Create dataset splits for supervised and semi-supervised learning')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--embedding_type', type=str, choices=['cae', 'tae'], required=True,
                        help='Type of embedding (cae or tae)')
    parser.add_argument('--embedding_file', type=str, help='Specific embedding file to use (optional)')
    parser.add_argument('--patch_size', type=int, default=DEFAULT_PATCH_SIZE,
                        help='Patch size used for embeddings (default: 5)')
    parser.add_argument('--n_splits', type=int, default=10, help='Number of random splits to create')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    embedding_path = locate_embedding_file(args)

    print(f"Loading embeddings from {embedding_path}")
    embeddings, labels, spatial_shape, dataset_info, metadata = load_embeddings(embedding_path)

    # Create dataset splits
    print(f"Creating {args.n_splits} dataset splits...")
    all_splits = create_dataset_splits(
        embeddings,
        labels,
        n_splits=args.n_splits,
        seed=args.seed
    )

    # Save splits
    splits_path, timestamp = save_splits(args, all_splits, embedding_path)
    print_summary(args, splits_path, timestamp, labels, all_splits)


if __name__ == "__main__":
    main()
