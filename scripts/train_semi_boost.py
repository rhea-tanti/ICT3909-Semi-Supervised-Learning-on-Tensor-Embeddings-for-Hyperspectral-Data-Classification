import argparse
import os
import sys
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.semi_supervised import CustomSemiBoost
from src.utils.training_utils import get_base_model, BaseSSLTrainer


class SemiBoostTrainer(BaseSSLTrainer):
    """SemiBoost trainer implementation."""

    def __init__(self):
        super().__init__("semi_boost")

    def create_model(self, combination: Dict[str, str], model_params: Dict[str, Any]) -> CustomSemiBoost:
        """Create SemiBoost model instance."""
        base_estimator = get_base_model(combination['base_model'])

        return CustomSemiBoost(
            base_estimator=base_estimator,
            n_neighbors=model_params['n_neighbors'],
            T=model_params['max_iterations'],
            sample_percent=model_params['sample_percent'],
            min_confidence=model_params['min_confidence'],
            class_balance=model_params['class_balance'],
            max_per_class=model_params['max_per_class'],
            similarity_kernel=model_params['similarity_kernel'],
            gamma=model_params['gamma'],
            ensemble_size=model_params['ensemble_size'],
            diversity_weight=model_params['diversity_weight']
        )

    def get_model_combinations(self) -> List[Dict[str, str]]:
        """Get all possible SemiBoost model combinations (3 base models)."""
        return [{'base_model': model} for model in self.base_models]

    def get_algorithm_specific_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add SemiBoost command line arguments."""
        parser.add_argument('--base_model', type=str, choices=self.base_models,
                            default='logistic_regression', help='Base model to use for SemiBoost')
        parser.add_argument('--similarity_kernel', type=str, choices=['rbf', 'knn', 'linear'], default='rbf',
                            help='Kernel to use for similarity calculations')
        parser.add_argument('--gamma', type=float,
                            help='Gamma parameter for rbf kernel (only used when similarity_kernel=rbf)')
        parser.add_argument('--n_neighbors', type=int, default=9,
                            help='Number of neighbors for knn similarity (only used when similarity_kernel=knn)')
        parser.add_argument('--iterations', type=int, default=100,
                            help='Maximum number of iterations (default: 100)')
        parser.add_argument('--sample_percent', type=float, default=0.05,
                            help='Percentage of unlabeled samples to add in each iteration (default: 0.05)')
        parser.add_argument('--confidence', type=float, default=0.75,
                            help='Minimum confidence threshold for selecting samples (default: 0.75)')
        parser.add_argument('--class_balance', action='store_true', default=True,
                            help='Whether to use class balancing (default: True)')
        parser.add_argument('--max_per_class', type=int, default=None,
                            help='Maximum number of samples to add per class per iteration (optional)')
        parser.add_argument('--ensemble_size', type=int, default=5,
                            help='Number of models to include in the ensemble (default: 5)')
        parser.add_argument('--diversity_weight', type=float, default=0.3,
                            help='Weight given to diversity vs accuracy in model selection (default: 0.3)')
        return parser

    def extract_model_params(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Extract SemiBoost specific parameters from command line args."""
        return {
            'similarity_kernel': args.similarity_kernel,
            'gamma': args.gamma,
            'n_neighbors': args.n_neighbors,
            'max_iterations': args.iterations,
            'sample_percent': args.sample_percent,
            'min_confidence': args.confidence,
            'class_balance': args.class_balance,
            'max_per_class': args.max_per_class,
            'ensemble_size': args.ensemble_size,
            'diversity_weight': args.diversity_weight
        }

    def get_single_combination_from_args(self, args: argparse.Namespace) -> Dict[str, str]:
        """Extract single SemiBoost model combination from command line args."""
        return {'base_model': args.base_model}


def main():
    """Main function to train and evaluate SemiBoost models."""
    trainer = SemiBoostTrainer()
    trainer.run_training()


if __name__ == "__main__":
    main()
