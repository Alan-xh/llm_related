"""Tiny Wan2.1-inspired video diffusion teaching model."""

from .model import Wan21Config, Wan21Model, build_model

__all__ = ["Wan21Config", "Wan21Model", "build_model"]
