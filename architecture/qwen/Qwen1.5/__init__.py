"""Qwen1.5 teaching implementation.

The model package can select a dense SwiGLU or top-k MoE feed-forward path;
both paths preserve the block shape ``[B, T, H]``.
"""
