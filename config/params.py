import os
import torch
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR: str = os.environ.get('HYPERSPECTRAL_DATA_DIR', str(PROJECT_ROOT / 'data'))
RAW_DATA_DIR: str = os.path.join(DATA_DIR, 'raw')

# Main directories
SHARED_DIR: str = os.path.join(DATA_DIR, 'shared')  # For shared resources across experiments
EXPERIMENTS_DIR: str = os.path.join(DATA_DIR, 'experiments')  # For experiment outputs
RESULTS_DIR: str = os.path.join(DATA_DIR, 'results')  # For results
FIGURES_DIR: str = os.path.join(PROJECT_ROOT, 'figures')  # For all figures and visualisations

# Shared resources subdirectories
SHARED_MODELS_DIR: str = os.path.join(SHARED_DIR, 'models')  # Trained models
SHARED_EMBEDDINGS_DIR: str = os.path.join(SHARED_DIR, 'embeddings')  # Extracted embeddings
SHARED_SPLITS_DIR: str = os.path.join(SHARED_DIR, 'splits')  # Dataset splits

# Experiment subdirectories
SUPERVISED_EXPERIMENTS_DIR: str = os.path.join(EXPERIMENTS_DIR, 'supervised')  # Supervised learning
SEMI_SUPERVISED_EXPERIMENTS_DIR: str = os.path.join(EXPERIMENTS_DIR, 'semi_supervised')  # Semi-supervised learning

# Create directories if they don't exist
all_directories = [
    RAW_DATA_DIR,
    SHARED_DIR,
    SHARED_MODELS_DIR,
    SHARED_EMBEDDINGS_DIR,
    SHARED_SPLITS_DIR,
    EXPERIMENTS_DIR,
    SUPERVISED_EXPERIMENTS_DIR,
    SEMI_SUPERVISED_EXPERIMENTS_DIR,
    RESULTS_DIR,
    FIGURES_DIR
]

for directory in all_directories:
    os.makedirs(directory, exist_ok=True)

# Default model parameters
DEFAULT_PATCH_SIZE = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_EPOCHS = 150
DEFAULT_LEARNING_RATE = 0.001

# Default embedding parameters
DEFAULT_EMBEDDING_BATCH_SIZE = 128

# Training configuration
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_NUM_WORKERS = 2
DEFAULT_PIN_MEMORY = True
DEFAULT_RANDOM_SEED = 42
