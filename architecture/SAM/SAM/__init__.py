"""Compact SAM v1 teaching model.

Pipeline: image ``[B,3,H,W]`` -> image/prompt encoders -> two-way decoder ->
masks ``[B,K,H,W]`` and IoU scores ``[B,K]``.
"""
