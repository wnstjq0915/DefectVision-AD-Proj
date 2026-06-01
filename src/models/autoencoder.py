from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DeconvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ConvAutoEncoder(nn.Module):
    """Convolutional autoencoder baseline for reconstruction-based anomaly detection."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, latent_channels: int = 256) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 4),
            ConvBlock(base_channels * 4, latent_channels),
        )
        self.decoder = nn.Sequential(
            DeconvBlock(latent_channels, base_channels * 4),
            DeconvBlock(base_channels * 4, base_channels * 2),
            DeconvBlock(base_channels * 2, base_channels),
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(image))


def reconstruction_error_map(image: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    """Return per-pixel mean squared reconstruction error as Bx1xHxW."""
    if image.shape != reconstruction.shape:
        raise ValueError(f"Input and reconstruction shapes differ: {image.shape} != {reconstruction.shape}")
    return torch.mean((image - reconstruction) ** 2, dim=1, keepdim=True)
