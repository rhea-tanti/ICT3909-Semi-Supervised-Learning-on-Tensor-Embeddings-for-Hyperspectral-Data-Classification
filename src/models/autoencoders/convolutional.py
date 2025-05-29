from typing import Dict, Tuple
import torch
import torch.nn as nn


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self,
                 dataset_info: Dict,
                 patch_size: int = 5):
        super().__init__()

        n_bands = dataset_info['bands']
        self.patch_size = patch_size

        self.channel_1 = min(128, n_bands * 2)
        self.channel_2 = min(64, n_bands)
        self.channel_3 = 32

        # Encoder
        self.enc_block1 = nn.Sequential(
            nn.Conv2d(n_bands, self.channel_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_1),
            nn.LeakyReLU(0.2)
        )

        self.enc_block2 = nn.Sequential(
            nn.Conv2d(self.channel_1, self.channel_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_2),
            nn.LeakyReLU(0.2)
        )

        self.enc_block3 = nn.Sequential(
            nn.Conv2d(self.channel_2, self.channel_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_3),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        )

        # Decoder
        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose2d(self.channel_3, self.channel_2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(self.channel_2),
            nn.LeakyReLU(0.2)
        )

        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose2d(self.channel_2, self.channel_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_1),
            nn.LeakyReLU(0.2)
        )

        self.dec_block3 = nn.Sequential(
            nn.ConvTranspose2d(self.channel_1, n_bands, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input to get the latent representation."""
        x = self.enc_block1(x)
        x = self.enc_block2(x)
        x = self.enc_block3(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the latent representation back to the original space."""
        x = self.dec_block1(x)
        x = self.dec_block2(x)
        x = self.dec_block3(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder."""
        enc = self.encode(x)
        dec = self.decode(enc)
        return dec, enc

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get the embedding representation."""
        return self.encode(x)
