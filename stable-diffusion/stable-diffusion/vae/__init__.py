"""
Self-contained Variational AutoEncoder (VAE) implementation.
Compatible with the latent-diffusion / Stable Diffusion first-stage autoencoder design.
"""

from .model import Encoder, Decoder, DiagonalGaussianDistribution, VAE
from .loss import VAELoss

__all__ = [
    "Encoder",
    "Decoder",
    "DiagonalGaussianDistribution",
    "VAE",
    "VAELoss",
]
