"""Compact, executable teaching implementations of the SAM model family.

All subpackages share ``architecture.SAM.common``. Model inputs are normally
images ``[B,3,H,W]`` with optional prompts; outputs are mask tensors and
quality scores, with SAM 2/3 variants adding memory or concept dimensions.
These modules are educational approximations, not official checkpoints.
"""
