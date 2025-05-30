import argparse
import subprocess
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.training_utils import process_training_args


def run_ssl_algorithm(algorithm: str, base_args: list, algorithm_args: dict):
    """Run a specific SSL algorithm with the --all flag."""
    command = [
        "python", f"-m", f"scripts.train_{algorithm}",
        *base_args,
        "--all"
    ]

    for key, value in algorithm_args.items():
        if value is not None:
            command.extend([f"--{key}", str(value)])

    print(f"\n\n========== RUNNING {algorithm.upper()} ==========\n")
    subprocess.run(command)


def main():
    """Main function to run all SSL algorithms with all base model combinations."""
    parser = argparse.ArgumentParser(
        description='Run all SSL algorithms with all possible base model combinations')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--embedding_type', type=str, choices=['cae', 'tae'], required=True,
                        help='Type of embedding (cae or tae)')
    parser.add_argument('--split_file', type=str, help='Name of the split file to use (defaults to latest)')
    parser.add_argument('--algorithms', type=str, nargs='+',
                        choices=['co_training', 'tri_training', 'semi_boost', 'assemble', 'all'],
                        default=['all'], help='SSL algorithms to run (default: all)')

    # Algorithm-specific parameters
    parser.add_argument('--confidence', type=float, default=0.75,
                        help='Confidence threshold (Co-Training, SemiBoost)')
    parser.add_argument('--iterations', type=int, default=150,
                        help='Maximum iterations (Co-Training, SemiBoost, Assemble)')
    parser.add_argument('--similarity_kernel', type=str, default='rbf', choices=['rbf', 'knn', 'linear'],
                        help='Similarity kernel for SemiBoost')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Gamma parameter for RBF kernel (SemiBoost)')
    parser.add_argument('--sample_percent', type=float, default=0.05,
                        help='Sample percent for SemiBoost')
    parser.add_argument('--n_neighbors', type=int, default=9,
                        help='Number of neighbors for KNN (SemiBoost)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha parameter for Assemble')
    parser.add_argument('--beta', type=float, default=0.7,
                        help='Beta parameter for Assemble')
    parser.add_argument('--ensemble_size', type=int, default=5,
                        help='Ensemble size for SemiBoost')
    parser.add_argument('--diversity_weight', type=float, default=0.3,
                        help='Diversity weight for SemiBoost')

    args, split_file = process_training_args(parser)

    # Base arguments
    base_args = [
        "--dataset", args.dataset,
        "--embedding_type", args.embedding_type,
        "--training_set", args.training_set,
        "--patch_size", str(args.patch_size)
    ]

    if split_file:
        base_args.extend(["--split_file", split_file])
    if args.description:
        base_args.extend(["--description", args.description])

    # Determine which algorithms to run
    algorithms = ['co_training', 'tri_training', 'semi_boost',
                  'assemble'] if 'all' in args.algorithms else args.algorithms

    # Algorithm-specific parameters
    algorithm_params = {
        'co_training': {'confidence': args.confidence, 'iterations': args.iterations},
        'tri_training': {},
        'semi_boost': {
            'similarity_kernel': args.similarity_kernel,
            'gamma': args.gamma,
            'sample_percent': args.sample_percent,
            'n_neighbors': args.n_neighbors,
            'iterations': args.iterations,
            'confidence': args.confidence,
            'ensemble_size': args.ensemble_size,
            'diversity_weight': args.diversity_weight
        },
        'assemble': {'iterations': args.iterations, 'alpha': args.alpha, 'beta': args.beta}
    }

    # Run each algorithm
    for algorithm in algorithms:
        run_ssl_algorithm(algorithm, base_args, algorithm_params[algorithm])

    print("\nCompleted running all specified SSL algorithms!")


if __name__ == "__main__":
    main()
