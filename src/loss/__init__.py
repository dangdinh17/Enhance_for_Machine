"""Loss functions used to train enhancement models."""

from .feature_loss import FeatureLoss, ResNet18FeatureExtractor

__all__ = ["FeatureLoss", "ResNet18FeatureExtractor"]
