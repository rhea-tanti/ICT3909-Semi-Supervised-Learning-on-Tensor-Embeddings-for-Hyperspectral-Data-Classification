import argparse
import json
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

from config.params import DEFAULT_PATCH_SIZE
from src.models.base_models import LogisticRegressionModel, NeuralNetworkModel, RandomForestModel, BASE_MODELS
from src.utils.experiment_utils import create_experiment, update_experiment_config, find_latest_split_file, load_datasets
from src.utils.evaluation_metrics import calculate_evaluation_metrics, serialize_metrics, plot_confusion_matrix


def get_base_model(base_model_type: str) -> Any:
    """Get base model instance based on the specified type."""
    if base_model_type == 'logistic_regression':
        return LogisticRegressionModel()
    elif base_model_type == 'random_forest':
        return RandomForestModel()
    elif base_model_type == 'neural_network':
        return NeuralNetworkModel()
    else:
        raise ValueError(f"Unknown base model type: {base_model_type}")


def prepare_ssl_data(
        dataset_name: str,
        embedding_type: str,
        split_file: str,
        split_index: int,
        training_set: str = 'train_5',
        patch_size: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load and prepare data for SSL training."""
    datasets, metadata = load_datasets(
        dataset_name=dataset_name,
        embedding_type=embedding_type,
        split_file=split_file,
        split_index=split_index
    )

    # Get labeled and unlabeled data
    X_labeled = datasets[training_set]['embeddings']
    y_labeled = datasets[training_set]['labels']
    X_large = datasets['train_large']['embeddings']
    y_large = datasets['train_large']['labels']

    # Create unlabeled data (samples in large set but not in labeled set)
    unlabeled_mask = np.ones(X_large.shape[0], dtype=bool)
    tolerance = 1e-8

    for x_label in X_labeled:
        diff = np.abs(X_large - x_label)
        matches = np.max(diff, axis=tuple(range(1, X_large.ndim))) < tolerance
        unlabeled_mask[matches] = False

    X_unlabeled = X_large[unlabeled_mask]
    X_test = datasets['test']['embeddings']
    y_test = datasets['test']['labels']

    if len(X_labeled.shape) > 2:
        X_labeled = X_labeled.reshape(X_labeled.shape[0], -1)
        X_unlabeled = X_unlabeled.reshape(X_unlabeled.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    return X_labeled, y_labeled, X_unlabeled, X_test, y_test, metadata


def prepare_supervised_data(
        dataset_name: str,
        embedding_type: str,
        split_file: str,
        split_index: int,
        training_set: str = 'train_large',
        patch_size: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load and prepare data for supervised training."""
    datasets, metadata = load_datasets(
        dataset_name=dataset_name,
        embedding_type=embedding_type,
        split_file=split_file,
        split_index=split_index
    )

    # Get training and test data
    X_train = datasets[training_set]['embeddings']
    y_train = datasets[training_set]['labels']
    X_test = datasets['test']['embeddings']
    y_test = datasets['test']['labels']

    if len(X_train.shape) > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test, metadata


def train_and_evaluate_ssl_model(
        model: Any,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        algorithm_name: str,
        model_name: str,
        results_dir: Optional[str] = None,
        figures_dir: Optional[str] = None,
        timestamp: Optional[str] = None,
        dataset_name: Optional[str] = None,
        embedding_type: Optional[str] = None,
        split_index: int = 0,
        training_set: str = 'train_5',
        **additional_params
) -> Dict:
    """Train and evaluate an SSL model."""
    # Train model
    print(f"Training {algorithm_name} with {model_name}...")
    model.fit(X_labeled, y_labeled, X_unlabeled)

    # Evaluate model
    print(f"Evaluating {algorithm_name} on test data...")
    y_pred = model.predict(X_test)
    metrics = calculate_evaluation_metrics(y_test, y_pred)

    # Add metadata
    metrics['model_name'] = model_name
    metrics['algorithm'] = algorithm_name
    metrics = serialize_metrics(metrics)

    if all([results_dir, figures_dir, timestamp, dataset_name, embedding_type]):
        # Save metrics
        results_filename = f"{dataset_name}_{embedding_type}_{algorithm_name}_{model_name}_split{split_index}_{timestamp}.json"
        results_path = os.path.join(results_dir, results_filename)

        os.makedirs(results_dir, exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'metadata': {
                    'dataset': dataset_name,
                    'embedding_type': embedding_type,
                    'split_index': split_index,
                    'training_set': training_set,
                    'algorithm': algorithm_name,
                    'model': model_name,
                    'timestamp': timestamp,
                    **additional_params
                }
            }, f, indent=2)

        # Plot and save confusion matrix
        unique_classes = sorted(list(set(np.unique(y_labeled)) | set(np.unique(y_test))))
        unique_classes = [int(cls) for cls in unique_classes if cls != 0]
        class_names = [str(cls) for cls in unique_classes]

        conf_matrix_filename = f"{dataset_name}_{embedding_type}_{algorithm_name}_{model_name}_split{split_index}_confmat_{timestamp}.png"
        conf_matrix_path = os.path.join(figures_dir, conf_matrix_filename)

        os.makedirs(figures_dir, exist_ok=True)
        conf_matrix = np.array(metrics['confusion_matrix'])
        title = f'Confusion Matrix - {algorithm_name} ({model_name})'
        plot_confusion_matrix(conf_matrix, class_names, conf_matrix_path, title=title, split_index=split_index)

    return metrics


def train_and_evaluate_supervised_model(
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        results_dir: Optional[str] = None,
        figures_dir: Optional[str] = None,
        timestamp: Optional[str] = None,
        dataset_name: Optional[str] = None,
        embedding_type: Optional[str] = None,
        split_index: int = 0,
        training_set: str = 'train_large',
        **model_params
) -> Dict:
    """Train and evaluate a supervised model."""
    # Create model instance
    if model_name not in BASE_MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(BASE_MODELS.keys())}")

    model_class = BASE_MODELS[model_name]
    model = model_class(**model_params)

    # Train model
    print(f"Training {model.model_name} on {training_set} set...")
    model.fit(X_train, y_train)

    # Evaluate model
    print(f"Evaluating {model.model_name} on test set...")
    metrics = model.evaluate(X_test, y_test)
    metrics['model_name'] = model.model_name
    metrics = serialize_metrics(metrics)

    # Save results if directories provided
    if all([results_dir, figures_dir, timestamp, dataset_name, embedding_type]):
        # Save metrics
        results_filename = f"{dataset_name}_{embedding_type}_{training_set}_split{split_index}_{model_name}_{timestamp}.json"
        results_path = os.path.join(results_dir, results_filename)

        os.makedirs(results_dir, exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'metadata': {
                    'dataset': dataset_name,
                    'embedding_type': embedding_type,
                    'split_index': split_index,
                    'training_set': training_set,
                    'model': model_name,
                    'timestamp': timestamp
                }
            }, f, indent=2)

        # Plot and save confusion matrix
        unique_classes = sorted(list(set(np.unique(y_train)) | set(np.unique(y_test))))
        unique_classes = [int(cls) for cls in unique_classes if cls != 0]
        class_names = [str(cls) for cls in unique_classes]

        conf_matrix_filename = f"{dataset_name}_{embedding_type}_{training_set}_split{split_index}_{model_name}_confmat_{timestamp}.png"
        conf_matrix_path = os.path.join(figures_dir, conf_matrix_filename)

        os.makedirs(figures_dir, exist_ok=True)
        conf_matrix = np.array(metrics['confusion_matrix'])
        title = f'Confusion Matrix - {model.model_name}'
        plot_confusion_matrix(conf_matrix, class_names, conf_matrix_path, title=title, split_index=split_index)

    return metrics


def process_training_args(parser: argparse.ArgumentParser) -> Tuple[Any, str]:
    """Process command line arguments."""
    existing_args = {action.dest for action in parser._actions}

    if 'patch_size' not in existing_args:
        parser.add_argument('--patch_size', type=int, default=DEFAULT_PATCH_SIZE,
                            help='Patch size used for embedding generation (default: from config)')

    args = parser.parse_args()

    # Handle split file selection
    if args.split_file is not None:
        split_file = args.split_file
    else:
        split_file = find_latest_split_file(args.dataset, args.embedding_type, args.patch_size)
        if split_file is None:
            raise ValueError(f"No split files found for {args.dataset} with {args.embedding_type} embeddings.")
        print(f"Using latest split file: {split_file}")

    return args, split_file


# Alias for backward compatibility
process_ssl_args = process_training_args


class BaseSSLTrainer(ABC):
    """Base class for SSL training."""

    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.base_models = ['logistic_regression', 'random_forest', 'neural_network']

    @abstractmethod
    def create_model(self, combination: Dict[str, str], model_params: Dict[str, Any]) -> Any:
        """Create SSL model instance."""
        pass

    @abstractmethod
    def get_model_combinations(self) -> List[Dict[str, str]]:
        """Get all possible model combinations for algorithm."""
        pass

    @abstractmethod
    def get_algorithm_specific_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add command line arguments."""
        pass

    @abstractmethod
    def extract_model_params(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Extract parameters from command line args."""
        pass

    def get_model_name(self, combination: Dict[str, str]) -> str:
        """Generate model name."""
        if 'base_model_2' in combination:
            if 'base_model_3' in combination:
                return f"{combination['base_model']}-{combination['base_model_2']}-{combination['base_model_3']}"
            else:
                return f"{combination['base_model']}-{combination['base_model_2']}"
        else:
            return combination['base_model']

    def create_base_parser(self) -> argparse.ArgumentParser:
        """Create base argument parser for SSL."""
        parser = argparse.ArgumentParser(
            description=f'Train and evaluate {self.algorithm_name} models on dataset splits')
        parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
        parser.add_argument('--embedding_type', type=str, choices=['cae', 'tae'], required=True,
                            help='Type of embedding (cae or tae)')
        parser.add_argument('--split_file', type=str, help='Name of the split file to use (defaults to latest)')
        parser.add_argument('--training_set', type=str, default='train_5',
                            choices=['train_5', 'train_10', 'train_20', 'train_large'],
                            help='Which training set to use as labelled data (default: train_5)')
        parser.add_argument('--all', action='store_true',
                            help='Train with all possible base model combinations')
        return parser

    def train_single_combination(
            self,
            dataset_name: str,
            embedding_type: str,
            split_file: str,
            combination: Dict[str, str],
            model_params: Dict[str, Any],
            training_set: str = 'train_5',
            patch_size: int = 5
    ) -> Dict:
        """Train a single model combination on all splits."""

        model_name = self.get_model_name(combination)

        # Create experiment
        experiment_dir, config = create_experiment(
            dataset_name=dataset_name,
            embedding_type=embedding_type,
            model_name=model_name,
            experiment_type="semi_supervised",
            algorithm_type=self.algorithm_name,
            training_set=training_set,
            patch_size=patch_size
        )

        # Update config
        config_update = {**combination, **model_params}
        update_experiment_config(experiment_dir, config_update)

        split_results = {}

        # Process each split
        for split_idx in range(10):
            print(f"\nProcessing split {split_idx}...")
            split_dir = os.path.join(experiment_dir, f"split_{split_idx}")
            os.makedirs(split_dir, exist_ok=True)

            # Create metrics and figures directories
            metrics_dir = os.path.join(split_dir, 'metrics')
            figures_dir = os.path.join(split_dir, 'figures')
            os.makedirs(metrics_dir, exist_ok=True)
            os.makedirs(figures_dir, exist_ok=True)

            # Load and prepare data
            X_labeled, y_labeled, X_unlabeled, X_test, y_test, metadata = prepare_ssl_data(
                dataset_name=dataset_name,
                embedding_type=embedding_type,
                split_file=split_file,
                split_index=split_idx,
                training_set=training_set,
                patch_size=patch_size
            )

            # Create model
            model = self.create_model(combination=combination, model_params=model_params)

            # Co-Training with ViewSplit
            if self.algorithm_name == "co_training":
                from LAMDA_SSL.Split.ViewSplit import ViewSplit
                split_labeled_X = ViewSplit(X_labeled, shuffle=False)
                split_unlabeled_X = ViewSplit(X_unlabeled, shuffle=False)
                split_test_X = ViewSplit(X_test, shuffle=False)

                # Train the model
                model.fit(X=split_labeled_X, y=y_labeled, unlabeled_X=split_unlabeled_X)

                # Evaluate model with viewsplit for Co-Training
                print(f"Training {self.algorithm_name} with {model_name}...")
                print(f"Evaluating {self.algorithm_name} on test data...")
                y_pred = model.predict(split_test_X)
                metrics = calculate_evaluation_metrics(y_test, y_pred)
                metrics['model_name'] = model_name
                metrics['algorithm'] = self.algorithm_name

                # Serialize metrics
                metrics = serialize_metrics(metrics)

                # Save results
                if all([metrics_dir, figures_dir, config['timestamp'], dataset_name, embedding_type]):
                    # Save metrics
                    results_filename = f"{dataset_name}_{embedding_type}_{self.algorithm_name}_{model_name}_split{split_idx}_{config['timestamp']}.json"
                    results_path = os.path.join(metrics_dir, results_filename)

                    os.makedirs(metrics_dir, exist_ok=True)

                    with open(results_path, 'w') as f:
                        json.dump({
                            'metrics': metrics,
                            'metadata': {
                                'dataset': dataset_name,
                                'embedding_type': embedding_type,
                                'split_index': split_idx,
                                'training_set': training_set,
                                'algorithm': self.algorithm_name,
                                'model': model_name,
                                'timestamp': config['timestamp'],
                                **model_params
                            }
                        }, f, indent=2)

                    # Plot and save confusion matrix
                    unique_classes = sorted(list(set(np.unique(y_labeled)) | set(np.unique(y_test))))
                    unique_classes = [int(cls) for cls in unique_classes if cls != 0]
                    class_names = [str(cls) for cls in unique_classes]

                    conf_matrix_filename = f"{dataset_name}_{embedding_type}_{self.algorithm_name}_{model_name}_split{split_idx}_confmat_{config['timestamp']}.png"
                    conf_matrix_path = os.path.join(figures_dir, conf_matrix_filename)

                    os.makedirs(figures_dir, exist_ok=True)
                    conf_matrix = np.array(metrics['confusion_matrix'])
                    title = f'Confusion Matrix - {self.algorithm_name} ({model_name})'
                    plot_confusion_matrix(conf_matrix, class_names, conf_matrix_path, title=title,
                                          split_index=split_idx)
            else:
                # Standard training
                model.fit(X_labeled, y_labeled, X_unlabeled)

                # Evaluate model
                metrics = train_and_evaluate_ssl_model(
                    model=model,
                    X_labeled=X_labeled,
                    y_labeled=y_labeled,
                    X_unlabeled=X_unlabeled,
                    X_test=X_test,
                    y_test=y_test,
                    algorithm_name=self.algorithm_name,
                    model_name=model_name,
                    results_dir=metrics_dir,
                    figures_dir=figures_dir,
                    timestamp=config['timestamp'],
                    dataset_name=dataset_name,
                    embedding_type=embedding_type,
                    split_index=split_idx,
                    training_set=training_set,
                    **model_params
                )

            # Store results and save split config
            split_results[f"split_{split_idx}"] = metrics
            split_config = {
                'split_index': split_idx,
                'dataset': dataset_name,
                'embedding_type': embedding_type,
                'training_set': training_set,
                'algorithm': self.algorithm_name,
                **combination,
                **model_params,
                'metrics': metrics,
                'timestamp': config['timestamp'],
                'parent_experiment': os.path.basename(experiment_dir)
            }

            split_config_path = os.path.join(split_dir, 'config.json')
            with open(split_config_path, 'w') as f:
                json.dump(split_config, f, indent=2)

            # Print summary metrics
            combination_str = " + ".join(combination.values()) if len(combination) > 1 else list(combination.values())[
                0]
            print(f"\nResults for {self.algorithm_name.title()} ({combination_str}) on split {split_idx}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Weighted Precision: {metrics['precision_weighted']:.4f}")
            print(f"  Weighted Recall: {metrics['recall_weighted']:.4f}")
            print(f"  Weighted F1: {metrics['f1_weighted']:.4f}")

        # Save experiment summary
        summary_path = os.path.join(experiment_dir, 'multi_split_summary.json')
        summary_data = {
            'dataset': dataset_name,
            'embedding_type': embedding_type,
            'training_set': training_set,
            'algorithm': self.algorithm_name,
            **combination,
            **model_params,
            'split_file': split_file,
            'split_indices': list(range(10)),
            'results': split_results,
            'timestamp': config['timestamp']
        }

        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        update_experiment_config(experiment_dir, {
            'artifacts': {
                'summary_path': summary_path,
                'split_file': split_file,
                'split_indices': list(range(10)),
            },
            'split_results': split_results
        })

        print(f"\n{'-' * 80}")
        print(f"{self.algorithm_name.title()} complete for all splits!")
        print(f"Experiment ID: {os.path.basename(experiment_dir)}")
        print(f"Summary saved to {summary_path}")
        print(f"{'-' * 80}")

        return {os.path.basename(experiment_dir): split_results}

    def run_training(self):
        """Main training orchestration method."""
        parser = self.create_base_parser()
        parser = self.get_algorithm_specific_args(parser)

        # Process arguments
        args, split_file = process_training_args(parser)
        model_params = self.extract_model_params(args)

        if args.all:
            # Train with all possible combinations
            combinations = self.get_model_combinations()
            print(f"Training {self.algorithm_name.title()} with all {len(combinations)} model combinations...")

            for combination in combinations:
                combination_str = " + ".join(combination.values()) if len(combination) > 1 else \
                list(combination.values())[0]
                print(f"\n\n====== TRAINING WITH {combination_str.upper()} ======\n")

                self.train_single_combination(
                    dataset_name=args.dataset,
                    embedding_type=args.embedding_type,
                    split_file=split_file,
                    combination=combination,
                    model_params=model_params,
                    training_set=args.training_set,
                    patch_size=args.patch_size
                )

            print(f"\nCompleted training for all {self.algorithm_name} combinations!")
        else:
            combination = self.get_single_combination_from_args(args)
            self.train_single_combination(
                dataset_name=args.dataset,
                embedding_type=args.embedding_type,
                split_file=split_file,
                combination=combination,
                model_params=model_params,
                training_set=args.training_set,
                patch_size=args.patch_size
            )

    def get_single_combination_from_args(self, args: argparse.Namespace) -> Dict[str, str]:
        """Extract single model combination from command line args."""
        combination = {'base_model': args.base_model}
        if hasattr(args, 'base_model_2') and args.base_model_2:
            combination['base_model_2'] = args.base_model_2
        if hasattr(args, 'base_model_3') and args.base_model_3:
            combination['base_model_3'] = args.base_model_3
        return combination


class BaseSupervisedTrainer:
    """Base class for supervised training."""

    def __init__(self):
        self.available_models = list(BASE_MODELS.keys())

    def create_base_parser(self) -> argparse.ArgumentParser:
        """Create base argument parser for supervised training."""
        parser = argparse.ArgumentParser(description='Train and evaluate supervised models on dataset splits')
        parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
        parser.add_argument('--embedding_type', type=str, choices=['cae', 'tae'], required=True,
                            help='Type of embedding (cae or tae)')
        parser.add_argument('--split_file', type=str, help='Name of the split file to use (defaults to latest)')
        parser.add_argument('--training_set', type=str, default='train_large',
                            choices=['train_5', 'train_10', 'train_20', 'train_large'],
                            help='Which training set to use (default: train_large)')
        parser.add_argument('--models', type=str, nargs='+',
                            choices=self.available_models + ['all'], default=['all'],
                            help='Models to train (default: all)')
        return parser

    def train_single_model(
            self,
            dataset_name: str,
            embedding_type: str,
            split_file: str,
            model_name: str,
            training_set: str = 'train_large',
            patch_size: int = 5
    ) -> Dict:
        """Train a single model on all splits."""

        # Create experiment
        experiment_dir, config = create_experiment(
            dataset_name=dataset_name,
            embedding_type=embedding_type,
            model_name=model_name,
            experiment_type="supervised",
            training_set=training_set,
            patch_size=patch_size
        )

        split_results = {}

        # Process each split
        for split_idx in range(10):
            print(f"\nProcessing split {split_idx}...")
            split_dir = os.path.join(experiment_dir, f"split_{split_idx}")
            os.makedirs(split_dir, exist_ok=True)

            # Create metrics and figures directories
            metrics_dir = os.path.join(split_dir, 'metrics')
            figures_dir = os.path.join(split_dir, 'figures')
            os.makedirs(metrics_dir, exist_ok=True)
            os.makedirs(figures_dir, exist_ok=True)

            # Load and prepare data
            X_train, y_train, X_test, y_test, metadata = prepare_supervised_data(
                dataset_name=dataset_name,
                embedding_type=embedding_type,
                split_file=split_file,
                split_index=split_idx,
                training_set=training_set,
                patch_size=patch_size
            )

            # Train and evaluate model
            metrics = train_and_evaluate_supervised_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                results_dir=metrics_dir,
                figures_dir=figures_dir,
                timestamp=config['timestamp'],
                dataset_name=dataset_name,
                embedding_type=embedding_type,
                split_index=split_idx,
                training_set=training_set
            )

            # Store results and save split config
            split_results[f"split_{split_idx}"] = metrics
            split_config = {
                'split_index': split_idx,
                'dataset': dataset_name,
                'embedding_type': embedding_type,
                'training_set': training_set,
                'model': model_name,
                'metrics': metrics,
                'timestamp': config['timestamp'],
                'parent_experiment': os.path.basename(experiment_dir)
            }

            split_config_path = os.path.join(split_dir, 'config.json')
            with open(split_config_path, 'w') as f:
                json.dump(split_config, f, indent=2)

            # Print summary metrics
            print(f"\nResults for {model_name} on split {split_idx}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Weighted Precision: {metrics['precision_weighted']:.4f}")
            print(f"  Weighted Recall: {metrics['recall_weighted']:.4f}")
            print(f"  Weighted F1: {metrics['f1_weighted']:.4f}")

        # Save experiment summary
        summary_path = os.path.join(experiment_dir, 'multi_split_summary.json')
        summary_data = {
            'dataset': dataset_name,
            'embedding_type': embedding_type,
            'training_set': training_set,
            'model': model_name,
            'split_file': split_file,
            'split_indices': list(range(10)),
            'results': split_results,
            'timestamp': config['timestamp']
        }

        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        update_experiment_config(experiment_dir, {
            'artifacts': {
                'summary_path': summary_path,
                'split_file': split_file,
                'split_indices': list(range(10)),
            },
            'split_results': split_results
        })

        print(f"\n{'-' * 80}")
        print(f"Supervised training complete for {model_name} on all splits!")
        print(f"Experiment ID: {os.path.basename(experiment_dir)}")
        print(f"Summary saved to {summary_path}")
        print(f"{'-' * 80}")

        return {os.path.basename(experiment_dir): split_results}

    def run_training(self):
        """Main training method."""
        parser = self.create_base_parser()
        args, split_file = process_training_args(parser)

        models_to_train = self.available_models if 'all' in args.models else args.models

        print(f"Training {len(models_to_train)} supervised models on all splits...")

        all_results = {}
        for model_name in models_to_train:
            print(f"\n{'-' * 80}")
            print(f"Training {model_name} on all splits")
            print(f"{'-' * 80}\n")

            model_results = self.train_single_model(
                dataset_name=args.dataset,
                embedding_type=args.embedding_type,
                split_file=split_file,
                model_name=model_name,
                training_set=args.training_set,
                patch_size=args.patch_size
            )
            all_results.update(model_results)

        print(f"\n{'-' * 80}")
        print(f"All supervised models trained successfully!")
        print(f"{'-' * 80}")

        return all_results
