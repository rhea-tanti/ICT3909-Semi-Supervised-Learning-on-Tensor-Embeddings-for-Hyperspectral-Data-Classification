from typing import Dict, Tuple
import tensorly as tl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tltorch.factorized_layers import TCL

tl.set_backend('pytorch')


class TensorAutoEncoder(nn.Module):
    """Tensor Autoencoder for hyperspectral data using Tensor Contraction Layers."""

    def __init__(self, dataset_info: Dict, patch_size: int = 5):
        """Initialize the model."""
        super(TensorAutoEncoder, self).__init__()

        n_bands = dataset_info['bands']

        self.rank_1 = (min(128, n_bands * 2), patch_size, patch_size)
        self.rank_2 = (min(64, n_bands), patch_size, patch_size)
        self.rank_3 = (32, patch_size - 2, patch_size - 2)
        input_shape = (n_bands, patch_size, patch_size)

        # Encoder TCL layers
        self.encoder_tcl1 = TCL(
            input_shape=input_shape,
            rank=self.rank_1
        )
        self.encoder_tcl2 = TCL(
            input_shape=self.rank_1,
            rank=self.rank_2
        )
        self.encoder_tcl3 = TCL(
            input_shape=self.rank_2,
            rank=self.rank_3
        )

        # Decoder TCL layers
        self.decoder_tcl1 = TCL(
            input_shape=self.rank_3,
            rank=self.rank_2
        )
        self.decoder_tcl2 = TCL(
            input_shape=self.rank_2,
            rank=self.rank_1
        )
        self.decoder_tcl3 = TCL(
            input_shape=self.rank_1,
            rank=input_shape
        )

        # Encoder normalization layers
        self.enc_norm1 = nn.LayerNorm(self.rank_1)
        self.enc_norm2 = nn.LayerNorm(self.rank_2)
        self.enc_norm3 = nn.LayerNorm(self.rank_3)

        # Decoder normalization layers
        self.dec_norm1 = nn.LayerNorm(self.rank_2)
        self.dec_norm2 = nn.LayerNorm(self.rank_1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the encoding operations with all activations."""
        enc = F.leaky_relu(self.enc_norm1(self.encoder_tcl1(x)))
        enc = F.leaky_relu(self.enc_norm2(self.encoder_tcl2(enc)))
        enc = F.leaky_relu(self.enc_norm3(self.encoder_tcl3(enc)))
        return enc

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the decoding operations with all activations."""
        dec = F.leaky_relu(self.dec_norm1(self.decoder_tcl1(x)))
        dec = F.leaky_relu(self.dec_norm2(self.decoder_tcl2(dec)))
        dec = F.sigmoid(self.decoder_tcl3(dec))
        return dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        enc = self._encode(x)
        dec = self._decode(enc)
        return dec, enc

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get the embedding representation."""
        return self._encode(x)
