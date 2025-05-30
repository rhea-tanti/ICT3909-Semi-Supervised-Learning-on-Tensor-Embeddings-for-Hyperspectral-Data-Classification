# Semi-Supervised Learning on Tensor Embeddings for Hyperspectral Data Classification

## Project Overview

This repository contains the implementation for the FYP titled semi-supervised learning using tensor embeddings for hyperspectral data classification. The project aims to address the challenge of limited labelled data in hyperspectral remote sensing by leveraging both labelled and unlabelled data for classification tasks.

## Setup Instructions

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```
2. Download hyperspectral datasets and place in `data/raw/` directory. The implementation uses [publicly available 
hyperspectral datasets](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes), including Indian Pines, Salinas, Pavia Center, Pavia University, Kennedy Space Center (KSC), and Botswana.

## Key Components

### Data Preprocessing

- `data_loader.py`: Loads hyperspectral data from .mat files and normalizes it
- `patches.py`: Implements patch extraction 

### Embedding Generation

- `convolutional.py`: Implements a Convolutional Autoencoder 
- `tensor.py`: Implements a Tensor Autoencoder using tensor contraction layers
- `embeddings.py`: Provides utilities for extracting and saving embeddings

### Supervised Models

- `logistic_regression.py`: Linear classifier 
- `random_forest.py`: Gradient-free non-linear classifier
- `neural_network.py`: Gradient-based non-linear classifier

### Semi-Supervised Learning

The implemented SSL algorithms build upon the [LAMDA-SSL](https://github.com/YGZWQZD/LAMDA-SSL) framework and include:

- `co_training.py`: Trains two classifiers on different views of the data and uses high-confidence predictions to augment the training set
- `tri_training.py`: Extends co-training to three classifiers with majority voting for unlabelled data
- `assemble.py`: Ensemble learning approach with dynamic weighting based on classifier performance
- `semi_boost.py`: Incorporates boosting principles with similarity-based pseudo-labelling for unlabelled data

## Usage

### 1. Train Autoencoders

Train autoencoders to generate embeddings:

```bash
# Train Convolutional Autoencoder
python -m scripts.train_cae --dataset Pavia --patch_size 5 

# Train Tensor Autoencoder
python -m scripts.train_tae --dataset Pavia --patch_size 5 --visualise
```

### 2. Create Dataset Splits

Create dataset splits from the generated embeddings:

```bash
# Create splits for Pavia dataset with convolutional autoencoder embeddings
python -m scripts.create_splits --dataset Pavia --embedding_type cae

# Create splits for Pavia dataset with tensor autoencoder embeddings
python -m scripts.create_splits --dataset Pavia --embedding_type tae
```

This creates splits with:
- Large training set (`train_large`): 50 samples per class. For classes with fewer than 50 samples, 80% of the samples are used for training and 20% for testing.
- 20-sample training set (`train_20`): 20 samples per class (subset of large set)
- 10-sample training set (`train_10`): 10 samples per class (subset of 20-sample set)
- Small training set (`train_5`): 5 samples per class (subset of 10-sample set)
- Test set: Remaining samples

### 3. Train Supervised Models

```bash
# Train all supervised models on convolutional autoencoder embeddings (defaults to large training set)
python -m scripts.train_supervised --dataset Pavia --embedding_type cae

# Train single model on small training set
python -m scripts.train_supervised --dataset Pavia --embedding_type cae --training_set train_5 --models neural_network 
```

### 4. Train Semi-Supervised Models
The semi-supervised learning models can be trained using different amounts of labeled data:

- **5 labeled samples** (`train_5`): 5 samples per class from the splits. Default for semi-supervised learning.
- **10 labeled samples** (`train_10`): 10 samples per class from the splits
- **20 labeled samples** (`train_20`): 20 samples per class from the splits

In each case, the remaining samples from the large training set (50 samples per class) are used as unlabeled data.
The training can be done individually for each base model or for all combinations of base models in one run.

#### Co-Training

```bash
# Single configuration (default: train_5)
python -m scripts.train_co_training --dataset Pavia --embedding_type cae --base_model logistic_regression --base_model_2 random_forest

# All possible base model combinations (9 total) on train_10
python -m scripts.train_co_training --dataset Pavia --embedding_type cae --training_set train_10 --all
```

#### Tri-Training

```bash
# Single configuration (default: train_5)
python -m scripts.train_tri_training --dataset Pavia --embedding_type cae --base_model logistic_regression --base_model_2 random_forest --base_model_3 neural_network

# All possible base model combinations (27 total) on train_10
python -m scripts.train_tri_training --dataset Pavia --embedding_type cae --training_set train_10 --all
```

#### Assemble

```bash
# Single configuration (default: train_5)
python -m scripts.train_assemble --dataset Pavia --embedding_type cae --base_model logistic_regression

# All possible base models (3 total) on train_10
python -m scripts.train_assemble --dataset Pavia --embedding_type cae --training_set train_10 --all
```

#### SemiBoost

```bash
# Single configuration (default: train_5)
python -m scripts.train_semi_boost --dataset Pavia --embedding_type cae --base_model random_forest 

# All possible base models (3 total) on train_10
python -m scripts.train_semi_boost --dataset Pavia --embedding_type cae --training_set train_10 --all
```

### 5. Export Results to CSV

#### Generate CSV Results

```bash
# Export results for a single dataset and embedding type
python -m scripts.export_results_to_csv --dataset Pavia --embedding-types cae

# Export results for multiple embedding types
python -m scripts.export_results_to_csv --dataset Pavia --embedding-types cae,tae
```
This will create a CSV file with combined results from all experiments for the specified dataset and embedding types.
Results can be found in the `data/results/results_csv` directory.
