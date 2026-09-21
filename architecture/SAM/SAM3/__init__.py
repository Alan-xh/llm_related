"""Compact SAM 3 concept-segmentation teaching model.

Text prompts become concept tokens ``[B,1,C]`` and the output adds presence
logits ``[B,1]`` to the usual masks and quality scores.
"""
