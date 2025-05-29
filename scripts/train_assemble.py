import argparse
import os
import sys
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.semi_supervised import CustomAssemble
from src.utils.training_utils import get_base_model, BaseSSLTrainer


class AssembleTrainer(BaseSSLTrainer):
    """Assemble trainer implementation."""

    def __init__(self):
        super().__init__("assemble")

    def create_model(self, combination: Dict[str, str], model_params: Dict[str, Any]) -> CustomAssemble:
        """Create Assemble model instance."""
        base_estimator = get_base_model(combination['base_model'])

        return CustomAssemble(
            base_estimator=base_estimator,
            T=model_params['iterations'],
            alpha=model_params['alpha'],
            beta=model_params['beta']
        )

    def get_model_combinations(self) -> List[Dict[str, str]]:
        """Get all possible Assemble model combinations (3 base models)."""
        return [{'base_model': model} for model in self.base_models]

    def get_algorithm_specific_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add Assemble specific command line arguments."""
        parser.add_argument('--base_model', type=str, choices=self.base_models,
                            default='logistic_regression', help='Base model to use for Assemble')
        parser.add_argument('--iterations', type=int, default=40, help='Number of iterations (T parameter)')
        parser.add_argument('--alpha', type=float, default=0.5,
                            help='Weight parameter for sample distribution updates (default: 0.5)')
        parser.add_argument('--beta', type=float, default=0.7,
                            help='Weight distribution parameter for labeled vs unlabeled data (default: 0.7)')
        return parser

    def extract_model_params(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Extract Assemble specific parameters from command line args."""
        return {
            'iterations': args.iterations,
            'alpha': args.alpha,
            'beta': args.beta
        }

    def get_single_combination_from_args(self, args: argparse.Namespace) -> Dict[str, str]:
        """Extract single Assemble model combination from command line arguments."""
        return {'base_model': args.base_model}


def main():
    """Main function to train and evaluate Assemble models."""
    trainer = AssembleTrainer()
    trainer.run_training()


if __name__ == "__main__":
    main()
