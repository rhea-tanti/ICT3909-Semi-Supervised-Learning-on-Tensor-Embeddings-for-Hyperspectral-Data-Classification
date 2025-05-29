import argparse
import itertools
import os
import sys
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.semi_supervised import CustomTriTraining
from src.utils.training_utils import get_base_model, BaseSSLTrainer


class TriTrainingTrainer(BaseSSLTrainer):
    """Tri-Training trainer implementation."""

    def __init__(self):
        super().__init__("tri_training")

    def create_model(self, combination: Dict[str, str], model_params: Dict[str, Any]) -> CustomTriTraining:
        """Create Tri-Training model instance."""
        base_estimator = get_base_model(combination['base_model'])
        base_estimator_2 = get_base_model(combination.get('base_model_2', combination['base_model']))
        base_estimator_3 = get_base_model(combination.get('base_model_3', combination['base_model']))

        return CustomTriTraining(
            base_estimator=base_estimator,
            base_estimator_2=base_estimator_2,
            base_estimator_3=base_estimator_3
        )

    def get_model_combinations(self) -> List[Dict[str, str]]:
        """Get all possible Tri-Training model combinations (27 combinations)."""
        combinations = []
        for base_model_1, base_model_2, base_model_3 in itertools.product(
                self.base_models, self.base_models, self.base_models
        ):
            combinations.append({
                'base_model': base_model_1,
                'base_model_2': base_model_2,
                'base_model_3': base_model_3
            })
        return combinations

    def get_algorithm_specific_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add Tri-Training specific command line arguments."""
        parser.add_argument('--base_model', type=str, choices=self.base_models,
                            default='logistic_regression', help='First base model to use for Tri-Training')
        parser.add_argument('--base_model_2', type=str, choices=self.base_models,
                            help='Second base model to use for Tri-Training (defaults to same as first model)')
        parser.add_argument('--base_model_3', type=str, choices=self.base_models,
                            help='Third base model to use for Tri-Training (defaults to same as first model)')
        return parser

    def extract_model_params(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Extract Tri-Training specific parameters from command line args."""
        return {}  # no additional parameters needed

    def get_single_combination_from_args(self, args: argparse.Namespace) -> Dict[str, str]:
        """Extract single Tri-Training model combination from command line args."""
        return {
            'base_model': args.base_model,
            'base_model_2': args.base_model_2 if args.base_model_2 else args.base_model,
            'base_model_3': args.base_model_3 if args.base_model_3 else args.base_model
        }


def main():
    """Main function to train and evaluate Tri-Training models."""
    trainer = TriTrainingTrainer()
    trainer.run_training()


if __name__ == "__main__":
    main()
