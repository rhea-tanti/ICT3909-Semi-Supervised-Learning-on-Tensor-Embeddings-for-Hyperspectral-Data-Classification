#!/usr/bin/env python3
"""
Analyse and compare model parameters for CAE and TAE across all datasets.

Generates a comprehensive report comparing parameter counts, efficiency,
and model architectures for both Convolutional and Tensor Autoencoders.

Usage:
    python -m scripts.analyse_ae_parameters [--output-file FILE] [--patch-size SIZE]
"""

import argparse
import io
import os
import sys
from datetime import datetime
from pathlib import Path
from torchinfo import summary

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import DEFAULT_PATCH_SIZE, PROJECT_ROOT
from config.datasets import DATASETS
from src.models.autoencoders import ConvolutionalAutoEncoder, TensorAutoEncoder
from src.preprocessing.data_loader import load_dataset, get_dataset_info, normalise_data
from src.preprocessing.patches import create_dataloader


def analyse_dataset_models(dataset_name, patch_size=5):
    """Analyse CAE and TAE models for a specific dataset."""
    output_lines = []

    output_lines.append(f"\nDATASET: {dataset_name}")
    output_lines.append("-" * 80)

    try:
        # Load dataset
        hypercube, labels = load_dataset(dataset_name)
        dataset_info = get_dataset_info(hypercube, labels)
        normalised_data = normalise_data(hypercube)

        dataloader = create_dataloader(
            normalised_data,
            patch_size=patch_size,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )

        # Get sample batch
        sample_batch = next(iter(dataloader))[0]

        # Dataset information
        output_lines.append(f"Dataset dimensions: {hypercube.shape}")
        output_lines.append(f"Number of classes: {dataset_info['n_classes']}")
        output_lines.append(f"Number of spectral bands: {dataset_info['bands']}")
        output_lines.append(f"Sample batch shape: {sample_batch.shape}")
        output_lines.append(f"Patch size: {patch_size}x{patch_size}")
        output_lines.append("")

        # Initialise models
        cae_model = ConvolutionalAutoEncoder(dataset_info=dataset_info, patch_size=patch_size)
        tae_model = TensorAutoEncoder(dataset_info=dataset_info, patch_size=patch_size)

        # Get parameter counts
        cae_params = sum(p.numel() for p in cae_model.parameters())
        tae_params = sum(p.numel() for p in tae_model.parameters())

        # CAE Summary
        summary_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = summary_output

        summary(
            cae_model,
            input_data=sample_batch,
            col_width=20,
            depth=3,
            verbose=2,
            row_settings=["var_names"]
        )

        sys.stdout = original_stdout
        output_lines.append("CONVOLUTIONAL AUTOENCODER SUMMARY")
        output_lines.append("-" * 80)
        output_lines.append(summary_output.getvalue())

        # TAE Summary
        summary_output = io.StringIO()
        sys.stdout = summary_output

        summary(
            tae_model,
            input_data=sample_batch,
            col_width=20,
            depth=3,
            verbose=2,
            row_settings=["var_names"]
        )

        sys.stdout = original_stdout
        output_lines.append("\nTENSOR AUTOENCODER SUMMARY")
        output_lines.append("-" * 80)
        output_lines.append(summary_output.getvalue())

        # Parameter comparison
        output_lines.append("\nPARAMETER COMPARISON")
        output_lines.append("-" * 80)
        output_lines.append(f"CAE Total Parameters: {cae_params:,}")
        output_lines.append(f"TAE Total Parameters: {tae_params:,}")
        output_lines.append(f"Difference: {cae_params - tae_params:,} parameters")

        if cae_params > 0:
            reduction = (1 - tae_params / cae_params) * 100
            output_lines.append(
                f"TAE uses {tae_params / cae_params:.2%} of CAE parameters ({reduction:.1f}% reduction)")

        # Parameter efficiency
        bands = dataset_info['bands']
        output_lines.append(f"\nParameter efficiency (parameters per spectral band):")
        output_lines.append(f"CAE: {cae_params / bands:,.2f} parameters/band")
        output_lines.append(f"TAE: {tae_params / bands:,.2f} parameters/band")

        output_lines.append("\n" + "=" * 80)

        return output_lines, {
            'dataset': dataset_name,
            'cae_params': cae_params,
            'tae_params': tae_params,
            'bands': bands,
            'reduction_percent': (1 - tae_params / cae_params) * 100 if cae_params > 0 else 0
        }
    except Exception as e:
        output_lines.append(f"Error processing dataset {dataset_name}: {e}")
        output_lines.append("\n" + "=" * 80)
        return output_lines, None


def generate_summary_table(results):
    """Generate a summary table of all results."""
    if not results:
        return []

    output_lines = []
    output_lines.append("\nSUMMARY TABLE")
    output_lines.append("=" * 100)
    output_lines.append(
        f"{'Dataset':<15} {'CAE Params':<12} {'TAE Params':<12} {'Bands':<8} {'Reduction':<10} {'CAE/Band':<12} {'TAE/Band':<12}")
    output_lines.append("-" * 100)

    total_cae = 0
    total_tae = 0

    for result in results:
        if result:
            cae_per_band = result['cae_params'] / result['bands']
            tae_per_band = result['tae_params'] / result['bands']

            line = f"{result['dataset']:<15} {result['cae_params']:<12,} {result['tae_params']:<12,} "
            line += f"{result['bands']:<8} {result['reduction_percent']:<9.1f}% "
            line += f"{cae_per_band:<12,.0f} {tae_per_band:<12,.0f}"
            output_lines.append(line)

            total_cae += result['cae_params']
            total_tae += result['tae_params']

    output_lines.append("-" * 100)
    overall_reduction = (1 - total_tae / total_cae) * 100 if total_cae > 0 else 0
    avg_line = f"{'AVERAGE':<15} {total_cae / len(results):<12,.0f} {total_tae / len(results):<12,.0f} "
    avg_line += f"{'N/A':<8} {overall_reduction:<9.1f}% {'N/A':<12} {'N/A':<12}"
    output_lines.append(avg_line)

    return output_lines


def main():
    """Main function to analyse model parameters."""
    parser = argparse.ArgumentParser(description='Analyse model parameters for CAE and TAE')
    parser.add_argument('--output-file', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'results' / 'model_parameters.txt'),
                        help='Output file path')
    parser.add_argument('--patch-size', type=int, default=DEFAULT_PATCH_SIZE,
                        help='Patch size for analysis')

    args = parser.parse_args()

    print("Model Parameters Analysis")
    print("=" * 50)
    print(f"Patch size: {args.patch_size}")
    print(f"Output file: {args.output_file}")

    all_output_lines = []
    all_output_lines.append(f"Model Parameters Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    all_output_lines.append("=" * 80)
    all_output_lines.append(f"Patch size: {args.patch_size}")

    results = []
    for dataset_name in DATASETS.keys():
        print(f"Processing {dataset_name}...")
        dataset_output, result = analyse_dataset_models(dataset_name, args.patch_size)
        all_output_lines.extend(dataset_output)
        if result:
            results.append(result)

    # Generate summary table
    summary_lines = generate_summary_table(results)
    all_output_lines.extend(summary_lines)

    # Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for line in all_output_lines:
            f.write(line + '\n')

    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
