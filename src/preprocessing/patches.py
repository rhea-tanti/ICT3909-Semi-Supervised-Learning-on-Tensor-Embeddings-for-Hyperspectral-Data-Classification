import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HyperspectralPatchDataset(Dataset):
    """Dataset for hyperspectral patches."""

    def __init__(self, data: np.ndarray, patch_size: int):
        """Initialize dataset."""
        self.data = data
        self.patch_size = patch_size
        self.height, self.width, self.bands = data.shape
        self.pad_size = patch_size // 2

        # Pad data
        self.padded_data = np.pad(
            self.data,
            pad_width=((self.pad_size, self.pad_size),
                       (self.pad_size, self.pad_size),
                       (0, 0)),
            mode='constant'
        )

    def __len__(self):
        return self.height * self.width

    def __getitem__(self, idx):
        i, j = idx // self.width, idx % self.width
        # Extract patch coordinates in padded data
        i_pad = i + self.pad_size
        j_pad = j + self.pad_size

        patch = self.padded_data[i_pad - self.pad_size:i_pad + self.pad_size + 1,
                j_pad - self.pad_size:j_pad + self.pad_size + 1, :]

        patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1).copy()).float()

        return (patch_tensor,)


def create_dataloader(
        data: np.ndarray,
        patch_size: int,
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 2,
        pin_memory: bool = True
) -> DataLoader:
    """Create a DataLoader for hyperspectral patches."""
    dataset = HyperspectralPatchDataset(data, patch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dataloader
