import argparse
import itertools
import os
import sys
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.semi_supervised import CustomCoTraining
from src.utils.training_utils import get_base_model, BaseSSLTrainer


class CoTrainingTrainer(BaseSSLTrainer):
    """Co-Training trainer implementation."""

    def __init__(self):
        super().__init__("co_training")

    def create_model(self, combination: Dict[str, str], model_params: Dict[str, Any]) -> CustomCoTraining:
        """Create Co-Training model instance."""
        base_estimator = get_base_model(combination['base_model'])
        base_estimator_2 = get_base_model(combination.get('base_model_2', combination['base_model']))

        return CustomCoTraining(
            base_estimator=base_estimator,
            base_estimator_2=base_estimator_2,
            p=5,
            n=5,
            k=model_params['max_iterations'],
            s=80,
            random_state=42,
            binary=False,
            threshold=model_params['confidence_threshold']
        )

    def get_model_combinations(self) -> List[Dict[str, str]]:
        """Get all possible Co-Training model combinations (9 combinations)."""
        combinations = []
        for base_model_1, base_model_2 in itertools.product(self.base_models, self.base_models):
            combinations.append({
                'base_model': base_model_1,
                'base_model_2': base_model_2
            })
        return combinations

    def get_algorithm_specific_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add Co-Training specific command line arguments."""
        parser.add_argument('--base_model', type=str, choices=self.base_models,
                            default='logistic_regression', help='Base model to use for Co-Training')
        parser.add_argument('--base_model_2', type=str, choices=self.base_models,
                            help='Second base model to use for Co-Training (defaults to same as first model)')
        parser.add_argument('--confidence', type=float, default=0.75,
                            help='Confidence threshold for selecting samples (default: 0.75)')
        parser.add_argument('--iterations', type=int, default=150,
                            help='Maximum number of Co-Training iterations (default: 150)')
        return parser

    def extract_model_params(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Extract Co-Training specific parameters from command line args."""
        return {
            'confidence_threshold': args.confidence,
            'max_iterations': args.iterations
        }

    def get_single_combination_from_args(self, args: argparse.Namespace) -> Dict[str, str]:
        """Extract single Co-Training model combination from command line args."""
        return {
            'base_model': args.base_model,
            'base_model_2': args.base_model_2 if args.base_model_2 else args.base_model
        }


def main():
    """Main function to train and evaluate Co-Training models."""
    trainer = CoTrainingTrainer()
    trainer.run_training()


if __name__ == "__main__":
    main()
