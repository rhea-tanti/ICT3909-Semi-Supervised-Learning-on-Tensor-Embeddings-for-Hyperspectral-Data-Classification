import argparse
import glob
import json
import os
import pandas as pd
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import PROJECT_ROOT, EXPERIMENTS_DIR, DEFAULT_PATCH_SIZE
from config.datasets import DATASETS

# Results directory
RESULTS_DIR = PROJECT_ROOT / 'data' / 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def find_metrics_files(base_dir, dataset, embedding_types, patch_size=None, labelled_samples=None):
    """
    Find all metrics files.
    """
    metrics_files = []
    patch_dir = f"patch_{patch_size}" if patch_size else "*"

    for embedding_type in embedding_types:
        # For supervised experiments
        supervised_pattern = os.path.join(
            base_dir, 'supervised', patch_dir, dataset, embedding_type, '*', '*', 'experiment_*', 'split_*'
        )
        supervised_dirs = glob.glob(supervised_pattern)

        for split_dir in supervised_dirs:
            metrics_dir = os.path.join(split_dir, 'metrics')
            if os.path.exists(metrics_dir):
                # Find all JSON files
                json_files = glob.glob(os.path.join(metrics_dir, '*.json'))
                for file in json_files:
                    metrics_files.append((file, embedding_type))

        # For semi-supervised experiments
        if labelled_samples:
            if isinstance(labelled_samples, str):
                labelled_samples_list = [labelled_samples]
            else:
                labelled_samples_list = labelled_samples

            for samples in labelled_samples_list:
                ssl_pattern = os.path.join(
                    base_dir, 'semi_supervised', patch_dir, '*', dataset, embedding_type, samples, '*', 'experiment_*',
                    'split_*'
                )
                ssl_dirs = glob.glob(ssl_pattern)

                for split_dir in ssl_dirs:
                    metrics_dir = os.path.join(split_dir, 'metrics')
                    if os.path.exists(metrics_dir):
                        # Find all JSON files
                        json_files = glob.glob(os.path.join(metrics_dir, '*.json'))
                        for file in json_files:
                            metrics_files.append((file, embedding_type))
        else:
            ssl_pattern = os.path.join(
                base_dir, 'semi_supervised', patch_dir, '*', dataset, embedding_type, '*', '*', 'experiment_*',
                'split_*'
            )
            ssl_dirs = glob.glob(ssl_pattern)

            for split_dir in ssl_dirs:
                metrics_dir = os.path.join(split_dir, 'metrics')
                if os.path.exists(metrics_dir):
                    # Find all JSON files
                    json_files = glob.glob(os.path.join(metrics_dir, '*.json'))
                    for file in json_files:
                        metrics_files.append((file, embedding_type))

    return metrics_files


def extract_path_components(file_path):
    """Extract components from the file path."""
    parts = Path(file_path).parts

    learning_type = None
    algorithm = None
    training_size = None
    labelled_samples = None
    model = None
    experiment_id = None
    split = None

    # Extract components
    for i, part in enumerate(parts):
        if part == 'supervised':
            learning_type = 'supervised'
        elif part == 'semi_supervised':
            learning_type = 'semi-supervised'
            if i + 2 < len(parts):
                algorithm = parts[i + 2]
        elif part in ['train_5', 'train_10', 'train_20', 'train_large']:
            training_size = part
        elif part.endswith('_labelled'):
            labelled_samples = int(part.split('_')[0])
        elif part.startswith('experiment_'):
            experiment_id = part
        elif part.startswith('split_'):
            split = int(part.split('_')[1])

    # Find model
    for i, part in enumerate(parts):
        if part.startswith('experiment_') and i > 0:
            model = parts[i - 1]

    return {
        'learning_type': learning_type,
        'algorithm': algorithm,
        'training_size': training_size,
        'labelled_samples': labelled_samples,
        'model': model,
        'split': split,
        'experiment_id': experiment_id
    }


def process_metrics_file(file_path, dataset, embedding_type):
    """Process a single metrics file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        path_info = extract_path_components(file_path)

        # Get metrics
        metrics = {}
        if 'metrics' in data:
            metrics = data['metrics']
        elif 'results' in data:
            metrics = data['results']
        else:
            metrics = data

        labelled_samples = path_info['labelled_samples']
        if labelled_samples is None and path_info['training_size']:
            size_mapping = {
                'train_5': 5,
                'train_10': 10,
                'train_20': 20,
                'train_large': 50,
            }
            labelled_samples = size_mapping.get(path_info['training_size'])

        method = 'supervised'
        if path_info['learning_type'] == 'semi-supervised' and path_info['algorithm']:
            method = path_info['algorithm']

        result = {
            'split_index': path_info['split'],
            'dataset': dataset,
            'embedding_type': embedding_type,
            'method': method,
            'labelled_samples': labelled_samples,
            'model': path_info['model'],
            'experiment_id': path_info['experiment_id']
        }

        # Add metrics
        for k, v in metrics.items():
            if not isinstance(v, (list, dict)):  # Skip complex values
                if k == 'precision_weighted':
                    result['precision'] = v
                elif k == 'recall_weighted':
                    result['recall'] = v
                elif k == 'f1_weighted':
                    result['f1'] = v
                elif k not in ['precision', 'recall', 'f1', 'model_name', 'patch_size'] or (
                        k in ['precision', 'recall', 'f1'] and k not in result):
                    result[k] = v

        return result

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def create_combined_results_csv(dataset, embedding_types, patch_size=None, labelled_samples=None, output_dir=None):
    """
    Create CSV file.
    """
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, 'results_csv')

    os.makedirs(output_dir, exist_ok=True)

    # Find all metrics files
    metrics_files = find_metrics_files(EXPERIMENTS_DIR, dataset, embedding_types, patch_size, labelled_samples)

    if not metrics_files:
        print(f"No metrics files found for {dataset} with embedding types {embedding_types}")
        return None


    results = []
    for file_path, emb_type in metrics_files:
        result = process_metrics_file(file_path, dataset, emb_type)
        if result:
            results.append(result)

    if not results:
        print(f"No valid results found for {dataset} with embedding types {embedding_types}")
        return None

    df = pd.DataFrame(results)

    # Define column order
    first_columns = ['split_index', 'dataset', 'embedding_type', 'method', 'labelled_samples', 'model']
    metric_columns = [col for col in df.columns if col not in first_columns and col != 'experiment_id']
    column_order = first_columns + metric_columns + ['experiment_id']

    # Reorder columns
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]

    # Save to CSV
    embedding_names = '_'.join(embedding_types)
    patch_suffix = f"_patch_{patch_size}" if patch_size else ""

    labelled_suffix = ""
    if labelled_samples:
        if isinstance(labelled_samples, str):
            labelled_suffix = f"_{labelled_samples}"
        else:
            labelled_suffix = f"_{'-'.join([s.split('_')[0] for s in labelled_samples])}_labelled"

    csv_filename = f"{dataset}_{embedding_names}{patch_suffix}{labelled_suffix}_results.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False)

    print(f"Results saved to {csv_path} ({len(df)} entries)")
    return df


def main():
    parser = argparse.ArgumentParser(description='Create results CSV files for experiment results')
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASETS.keys()),
                        help='Dataset name')
    parser.add_argument('--embedding_types', type=str, required=True,
                        help='Comma-separated list of embedding types (e.g., "cae,tae")')
    parser.add_argument('--patch_size', type=int, default=DEFAULT_PATCH_SIZE,
                        help='Patch size (default: from config)')
    parser.add_argument('--labelled_samples', type=str,
                        help='Comma-separated list of labelled samples to include (e.g., "5_labelled,10_labelled,20_labelled")')
    parser.add_argument('--output_dir', type=str, help='Output directory for CSV files')

    args = parser.parse_args()
    embedding_types = [et.strip() for et in args.embedding_types.split(',')]

    labelled_samples = None
    if args.labelled_samples:
        labelled_samples = [ls.strip() for ls in args.labelled_samples.split(',')]
        labelled_samples = [ls if ls.endswith('_labelled') else f"{ls}_labelled" for ls in labelled_samples]

    print("Export Results to CSV")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Embedding types: {embedding_types}")
    print(f"Patch size: {args.patch_size}")
    if labelled_samples:
        print(f"Labelled samples filter: {labelled_samples}")

    create_combined_results_csv(args.dataset, embedding_types, args.patch_size, labelled_samples, args.output_dir)


if __name__ == '__main__':
    main()
