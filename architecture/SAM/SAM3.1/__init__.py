"""Compact SAM 3.1 multi-object teaching model.

One image encoder pass serves multiple object prompts; masks restore the
object axis as ``[1,O,K,H,W]``.
"""
