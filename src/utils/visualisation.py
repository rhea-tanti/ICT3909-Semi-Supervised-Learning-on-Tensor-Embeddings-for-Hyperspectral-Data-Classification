import os
from typing import List
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def plot_training_history(
        history: List[float],
        save_path: str = None,
        dataset_name: str = None,
        patch_size: int = None,
        timestamp: str = None,
        title: str = "Training Loss Over Time"
) -> None:
    """Plot the training history of the model."""
    plt.figure(figsize=(8, 5))
    plt.plot(history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Add dataset and patch information
    if dataset_name and patch_size:
        full_title = f"{title} - {dataset_name} (Patch Size: {patch_size})"
    else:
        full_title = title

    plt.title(full_title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Training history saved to {save_path}")

    plt.show()
    plt.close()


def visualise_reconstruction(
    model: nn.Module,
    dataloader: DataLoader,
    save_dir: str = None,
    dataset_name: str = None,
    patch_size: int = None,
    timestamp: str = None,
    num_samples: int = 5,
    band_index: int = 10,
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
) -> None:
    """Visualise original and reconstructed patches for a single spectral band."""
    model.eval()

    # Get a batch of data
    data_iter = iter(dataloader)
    batch_data = next(data_iter)[0][:num_samples].to(device)

    # Get reconstructions
    with torch.no_grad():
        reconstructions, _ = model(batch_data)

    batch_data = batch_data.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), 'visualisations')
    os.makedirs(save_dir, exist_ok=True)

    if dataset_name and patch_size and timestamp:
        filename = f"{dataset_name}_patch{patch_size}_band{band_index}_{timestamp}.png"
    else:
        filename = f"reconstruction_band_{band_index}.png"

    save_path = os.path.join(save_dir, filename)

    cmap = 'viridis'

    for i in range(num_samples):
        # Extract single band for visualisation
        band_original = batch_data[i, band_index]
        band_recon = reconstructions[i, band_index]

        # Calculate shared color scale for consistent comparison
        vmin = min(band_original.min(), band_recon.min())
        vmax = max(band_original.max(), band_recon.max())

        # Display original band
        im0 = axes[i, 0].imshow(band_original, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"Original Band {band_index} (Sample {i + 1})")
        axes[i, 0].axis('off')

        # Display reconstructed band
        im1 = axes[i, 1].imshow(band_recon, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"Reconstructed Band {band_index} (Sample {i + 1})")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Reconstruction visualisation saved to {save_path}")
    plt.show()
    plt.close()
