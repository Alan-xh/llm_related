"""Compact FastSAM teaching model.

The encoder creates candidate masks ``[B,N,H,W]`` once; prompts rank them into
selected masks ``[B,K,H,W]``.
"""
