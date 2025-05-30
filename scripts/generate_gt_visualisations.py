"""
Generate ground truth visualisations for hyperspectral datasets.

Usage:
    python -m scripts.generate_ground_truth_visualisations [--dataset DATASET] [--output-dir DIR]
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import PROJECT_ROOT
from config.datasets import DATASETS, CLASS_LABELS_MAP
from src.preprocessing.data_loader import load_dataset

plt.rcParams.update({
    'figure.dpi': 800,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'legend.fontsize': 11,
})


def create_visualisation(dataset_name, save_path):
    """Create ground truth visualisation for a dataset."""
    print(f"Processing {dataset_name}...")

    # Load data
    hypercube, gt_labels = load_dataset(dataset_name)
    class_labels = CLASS_LABELS_MAP.get(dataset_name, {})
    num_classes = len(np.unique(gt_labels)) - 1

    # Create false-colour image
    h, w, bands = hypercube.shape
    band_indices = [int(bands * 0.8), int(bands * 0.5), int(bands * 0.2)]  # RGB bands (example indices)
    normalised_data = (hypercube - np.min(hypercube, axis=(0, 1), keepdims=True)) / \
                     (np.max(hypercube, axis=(0, 1), keepdims=True) -
                      np.min(hypercube, axis=(0, 1), keepdims=True) + 1e-8)

    # Create false-color image
    rgb_img = np.stack([normalised_data[:, :, i] for i in band_indices], axis=-1)
    rgb_img = np.clip(rgb_img, 0, 1)

    # Create colormap
    colors = cm.tab20(np.linspace(0, 1, num_classes + 1))
    colors[0] = [0, 0, 0, 1]
    cmap = ListedColormap(colors)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # False-colour composite
    ax1.imshow(rgb_img)
    ax1.set_title('False-Colour Composite')
    ax1.axis('off')

    # Ground truth
    ax2.imshow(gt_labels, cmap=cmap, vmin=0, vmax=num_classes)
    ax2.set_title('Ground Truth Classes')
    ax2.axis('off')

    # Legend
    legend_elements = []
    legend_labels = []

    for i in range(1, num_classes + 1):
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor='black', linewidth=0.5))
        legend_labels.append(f'{i}: {class_labels.get(i, "Unknown")}')

    fig.legend(legend_elements, legend_labels, loc='lower center', ncol=min(5, num_classes),
               bbox_to_anchor=(0.5, -0.05), frameon=True)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate ground truth visualisations')
    parser.add_argument('--dataset', choices=list(DATASETS.keys()),
                       help='Specific dataset (default: all datasets)')
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'figures' / 'ground_truth'),
                       help='Output directory')

    args = parser.parse_args()

    print("Ground Truth Visualisation Generator")
    print("=" * 50)
    print(f"Output directory: {args.output_dir}")

    # Dataset filename mapping
    dataset_files = {
        'Pavia': 'pavia_visualization.png',
        'Indian_Pines': 'indian_pines_visualization.png',
        'PaviaU': 'paviau_visualization.png',
        'Salinas': 'salinas_visualization.png',
        'KSC': 'ksc_visualization.png',
        'Botswana': 'botswana_visualization.png'
    }

    if args.dataset:
        # Single dataset
        filename = dataset_files[args.dataset]
        save_path = os.path.join(args.output_dir, filename)
        create_visualisation(args.dataset, save_path)
    else:
        # All datasets
        print(f"\nProcessing {len(dataset_files)} datasets...")
        for dataset, filename in dataset_files.items():
            save_path = os.path.join(args.output_dir, filename)
            try:
                create_visualisation(dataset, save_path)
            except Exception as e:
                print(f"Error with {dataset}: {e}")

    print(f"\nFiles saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
