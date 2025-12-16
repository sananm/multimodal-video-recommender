"""
Model architectures for multimodal video recommendation.
"""

from .fusion import ConcatFusion, GatedFusion, AttentionFusion

__all__ = [
    "ConcatFusion",
    "GatedFusion",
    "AttentionFusion"
]
