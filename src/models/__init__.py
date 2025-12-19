"""
Model architectures for multimodal video recommendation.
"""

from .fusion import ConcatFusion, GatedFusion, AttentionFusion
from .two_tower import VideoTower, TwoTowerModel

__all__ = [
    "ConcatFusion",
    "GatedFusion",
    "AttentionFusion",
    "VideoTower",
    "TwoTowerModel"
]
