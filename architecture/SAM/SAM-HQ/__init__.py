"""Compact SAM-HQ teaching model.

The decoder adds a projected high-resolution feature path before dynamic mask
generation, returning masks ``[B,K,H,W]`` and scores ``[B,K]``.
"""
