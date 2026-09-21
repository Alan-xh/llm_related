"""Compact MobileSAM teaching model.

The depthwise-separable student emits ``[B,C,H/8,W/8]`` features and reuses
the SAM prompt and dynamic-mask decoder interfaces.
"""
