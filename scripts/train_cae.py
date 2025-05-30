import argparse
import datetime
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import (
    DATA_DIR, SHARED_MODELS_DIR, SHARED_EMBEDDINGS_DIR, PROJECT_ROOT,
    DEFAULT_BATCH_SIZE, DEFAULT_PATCH_SIZE, DEFAULT_NUM_EPOCHS, DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_WORKERS, DEFAULT_PIN_MEMORY, DEFAULT_DEVICE
)
from src.preprocessing.data_loader import load_dataset, get_dataset_info, normalise_data
from src.preprocessing.patches import create_dataloader
from src.models.autoencoders import ConvolutionalAutoEncoder
from src.preprocessing.embeddings import extract_embeddings
from src.utils.visualisation import plot_training_history, visualise_reconstruction


def train_conv_autoencoder(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        device: torch.device = None
) -> Tuple[nn.Module, List[float]]:
    """Train the convolutional autoencoder."""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0

        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device)

            # Forward pass
            reconstruction, _ = model(data)
            loss = criterion(reconstruction, data)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Calculate average loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        history.append(avg_epoch_loss)

        # Print statistics every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_epoch_loss:.6f}')

    return model, history


def main():
    """Main function to train and evaluate the CAE."""
    parser = argparse.ArgumentParser(description='Train a Convolutional Autoencoder for hyperspectral data')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Data directory')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--patch_size', type=int, default=DEFAULT_PATCH_SIZE, help='Patch size')
    parser.add_argument('--epochs', type=int, default=DEFAULT_NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS, help='Number of dataloader workers')
    parser.add_argument('--visualise', action='store_true', help='Visualise reconstructions')
    args = parser.parse_args()

    if DEFAULT_DEVICE == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device} (config default: {DEFAULT_DEVICE})")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    patch_dir = f"patch_{args.patch_size}"
    model_save_dir = os.path.join(SHARED_MODELS_DIR, patch_dir, args.dataset)
    embeddings_save_dir = os.path.join(SHARED_EMBEDDINGS_DIR, patch_dir, args.dataset)

    training_figures_dir = os.path.join(PROJECT_ROOT, 'figures', 'training')
    run_dir = os.path.join(training_figures_dir, f"{args.dataset}_cae_{timestamp}")
    results_save_dir = os.path.join(run_dir, 'plots')

    # Create directories
    for directory in [model_save_dir, embeddings_save_dir, results_save_dir]:
        os.makedirs(directory, exist_ok=True)

    # Load and preprocess data
    print(f"Loading dataset {args.dataset}...")
    hypercube, labels = load_dataset(args.dataset)
    dataset_info = get_dataset_info(hypercube, labels)
    normalised_data = normalise_data(hypercube)

    # Print dataset information
    print(f"Dataset information:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")

    # Create dataloader
    print(f"Creating patches with size {args.patch_size}x{args.patch_size}...")
    dataloader = create_dataloader(
        normalised_data,
        args.patch_size,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=DEFAULT_PIN_MEMORY
    )

    # Create model
    model = ConvolutionalAutoEncoder(dataset_info=dataset_info, patch_size=args.patch_size)

    # Train model
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}...")
    trained_model, history = train_conv_autoencoder(
        model=model,
        dataloader=dataloader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device
    )

    # Plot training history
    history_filename = f"{args.dataset}_cae_patch{args.patch_size}_history_{timestamp}.png"
    history_path = os.path.join(results_save_dir, history_filename)

    plot_training_history(
        history,
        save_path=history_path,
        dataset_name=args.dataset,
        patch_size=args.patch_size,
        timestamp=timestamp
    )

    # Visualise reconstructions
    if args.visualise:
        print("Visualising reconstructions...")
        visualise_reconstruction(
            trained_model,
            dataloader,
            save_dir=results_save_dir,
            dataset_name=args.dataset,
            patch_size=args.patch_size,
            timestamp=timestamp,
            device=device
        )

    # Save trained model
    model_filename = f"{args.dataset}_cae_patch{args.patch_size}_{timestamp}.pt"
    model_path = os.path.join(model_save_dir, model_filename)
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Extract and save embeddings
    print("Extracting embeddings...")
    embeddings_filename = f"{args.dataset}_cae_patch{args.patch_size}_{timestamp}.h5"
    embeddings_path = os.path.join(embeddings_save_dir, embeddings_filename)

    # Create inference dataloader
    inference_dataloader = create_dataloader(
        normalised_data,
        args.patch_size,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=DEFAULT_PIN_MEMORY
    )

    extract_embeddings(
        model=trained_model,
        dataloader=inference_dataloader,
        hypercube=hypercube,
        labels=labels,
        save_path=embeddings_path,
        dataset_info=dataset_info,
        patch_size=args.patch_size,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        device=device
    )
    print(f"Embeddings saved to {embeddings_path}")

    # Print summary
    print(f"\n{'-' * 80}")
    print("Training completed! Files saved to:")
    print(f"  Model: {model_path}")
    print(f"  Embeddings: {embeddings_path}")
    print(f"  Training figures: {results_save_dir}")
    print(f"{'-' * 80}")


if __name__ == "__main__":
    main()
