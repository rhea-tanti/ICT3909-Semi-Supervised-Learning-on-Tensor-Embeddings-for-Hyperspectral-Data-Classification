import os
import sys

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.training_utils import BaseSupervisedTrainer


def main():
    """Main function to train and evaluate supervised models."""
    trainer = BaseSupervisedTrainer()
    trainer.run_training()


if __name__ == "__main__":
    main()
