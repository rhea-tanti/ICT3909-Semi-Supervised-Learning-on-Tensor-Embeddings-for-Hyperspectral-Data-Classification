import datetime
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from tqdm import tqdm


def extract_embeddings(
        model: nn.Module,
        dataloader: DataLoader,
        hypercube: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        dataset_info: Dict,
        patch_size: int,
        dataset_name: str,
        model_type: str = 'autoencoder',
        batch_size: int = 128,
        device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
) -> None:
    """Extract embeddings from any trained encoder and save."""
    model.eval()

    height, width = hypercube.shape[0], hypercube.shape[1]
    spatial_shape = (height, width)
    total_pixels = height * width

    # Get embedding shape
    with torch.no_grad():
        sample_batch = next(iter(dataloader))[0][:1].to(device)
        sample_embedding = model.get_embedding(sample_batch)
        embedding_shape = sample_embedding.shape[1:]

    flattened_labels = labels.reshape(-1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with h5py.File(save_path, 'w') as hf:
        emb_dataset = hf.create_dataset(
            'embeddings',
            shape=(total_pixels, *embedding_shape),
            chunks=(min(batch_size, total_pixels), *embedding_shape),
            compression='gzip',
            dtype=np.float32
        )

        label_dataset = hf.create_dataset(
            'labels',
            data=flattened_labels,
            compression='gzip'
        )

        hf.create_dataset('spatial_shape', data=np.array(spatial_shape))

        # Store dataset information
        info_group = hf.create_group('dataset_info')
        for key, value in dataset_info.items():
            info_group.attrs[key] = value
        info_group.attrs['patch_size'] = patch_size

        # Process batches and save
        start_idx = 0
        progress_bar = tqdm(dataloader, desc="Extracting embeddings", unit="batch")

        with torch.no_grad():
            for data, in progress_bar:
                data = data.to(device)

                batch_embeddings = model.get_embedding(data)
                batch_embeddings_cpu = batch_embeddings.cpu().numpy()
                batch_size_actual = batch_embeddings_cpu.shape[0]
                end_idx = start_idx + batch_size_actual

                # Store in HDF5 file
                emb_dataset[start_idx:end_idx] = batch_embeddings_cpu
                start_idx = end_idx

                # Update progress bar
                progress_bar.set_postfix({
                    "progress": f"{end_idx}/{total_pixels}",
                    "percent": f"{end_idx / total_pixels * 100:.1f}%"
                })

        # Store additional metadata
        metadata = hf.create_group('metadata')
        metadata.attrs['dataset_name'] = dataset_name
        metadata.attrs['patch_size'] = patch_size
        metadata.attrs['embedding_dim'] = embedding_shape
        metadata.attrs['creation_date'] = datetime.datetime.now().isoformat()

        if model_type:
            metadata.attrs['model_type'] = model_type

    print(f"Embeddings successfully saved to {save_path}")


def load_embeddings(file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load embeddings from an HDF5 file."""
    with h5py.File(file_path, 'r') as hf:
        embeddings = hf['embeddings'][()]
        labels = hf['labels'][()]

        # Extract metadata
        metadata = {}
        if 'dataset_info' in hf:
            for key, value in hf['dataset_info'].attrs.items():
                metadata[key] = value

        if 'metadata' in hf:
            for key, value in hf['metadata'].attrs.items():
                metadata[key] = value

        if 'spatial_shape' in hf:
            metadata['spatial_shape'] = tuple(hf['spatial_shape'][()])

    return embeddings, labels, metadata
