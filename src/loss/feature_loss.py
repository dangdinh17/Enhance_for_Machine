"""ResNet-18 FPN feature loss for images and videos."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


PYRAMID_NAMES = ("p2", "p3", "p4", "p5", "p6")
BACKBONE_CHANNELS = (64, 128, 256, 512)


class ResNet18FPNFeatureExtractor(nn.Module):
    """Build hierarchical P2-P6 features with ResNet-18 and Torchvision FPN.

    ResNet stages ``layer1`` through ``layer4`` provide C2-C5 features. The
    :class:`~torchvision.ops.FeaturePyramidNetwork` adds lateral and top-down
    paths to produce P2-P5, while ``LastLevelMaxPool`` produces P6. All
    extractor parameters are frozen because this module is used as a fixed
    perceptual metric.

    Args:
        feature_names: Pyramid levels returned by :meth:`forward`.
        weights: Torchvision weights to load. Pass ``None`` for randomly
            initialized weights, which is mainly useful for offline tests.
        out_channels: Number of channels in every FPN output level.
        normalize: Apply ImageNet input normalization before extraction.
    """

    def __init__(
        self,
        feature_names: Sequence[str] = PYRAMID_NAMES,
        weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
        out_channels: int = 256,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        invalid_names = set(feature_names) - set(PYRAMID_NAMES)
        if invalid_names:
            raise ValueError(f"Unknown ResNet-18 FPN levels: {sorted(invalid_names)}")
        if not feature_names:
            raise ValueError("At least one FPN level must be selected")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")

        backbone = resnet18(weights=weights)
        self.backbone = create_feature_extractor(
            backbone,
            return_nodes={
                "layer1": "p2",
                "layer2": "p3",
                "layer3": "p4",
                "layer4": "p5",
            },
        )
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(BACKBONE_CHANNELS),
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.feature_names = tuple(feature_names)
        self.normalize = normalize

        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> ResNet18FPNFeatureExtractor:
        """Keep the frozen ResNet-FPN extractor in evaluation mode."""
        del mode
        return super().train(False)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        """Return selected P2-P6 maps for a 4-D RGB image batch."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "Feature extractor input must have shape [N, 3, H, W], "
                f"got {tuple(image.shape)}"
            )
        if not image.is_floating_point():
            raise TypeError("Feature extractor input must be floating point")

        value = image
        if self.normalize:
            value = (value - self.image_mean) / self.image_std

        pyramid = self.fpn(self.backbone(value))
        pyramid["p6"] = pyramid.pop("pool")
        return {name: pyramid[name] for name in self.feature_names}


# Backward-compatible name for callers that imported the previous extractor.
ResNet18FeatureExtractor = ResNet18FPNFeatureExtractor


class FeatureLoss(nn.Module):
    """Compute weighted perceptual distance over ResNet-18 FPN levels.

    Inputs may be images shaped ``[B, C, H, W]`` or videos shaped
    ``[B, T, C, H, W]``. One-channel luma data is repeated to RGB. Pixel
    values are expected in ``input_range`` and are mapped to ``[0, 1]``.

    Args:
        layer_weights: Weight of each selected P2-P6 pyramid level.
        weights: Torchvision ResNet-18 weights.
        fpn_channels: Number of channels in every FPN output level.
        distance: Feature distance, either ``"l1"`` or ``"mse"``.
        input_range: Minimum and maximum input pixel values.
    """

    def __init__(
        self,
        layer_weights: Mapping[str, float] | None = None,
        weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
        fpn_channels: int = 256,
        distance: str = "l1",
        input_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        selected_weights = dict(
            layer_weights
            if layer_weights is not None
            else {
                "p2": 1.0,
                "p3": 1.0,
                "p4": 1.0,
                "p5": 1.0,
                "p6": 1.0,
            }
        )
        if any(value < 0.0 for value in selected_weights.values()):
            raise ValueError("Feature layer weights must be non-negative")
        if not selected_weights or sum(selected_weights.values()) <= 0.0:
            raise ValueError("At least one feature layer weight must be positive")
        if distance not in {"l1", "mse"}:
            raise ValueError("distance must be either 'l1' or 'mse'")
        input_min, input_max = input_range
        if input_max <= input_min:
            raise ValueError("input_range maximum must be greater than minimum")

        self.layer_weights = selected_weights
        self.extractor = ResNet18FPNFeatureExtractor(
            feature_names=tuple(selected_weights),
            weights=weights,
            out_channels=fpn_channels,
        )
        self.criterion: nn.Module = nn.L1Loss() if distance == "l1" else nn.MSELoss()
        self.input_min = float(input_min)
        self.input_scale = float(input_max - input_min)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Return the weighted mean feature distance between two inputs."""
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have identical shapes, got "
                f"{tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        prediction_images = self._prepare_input(prediction)
        target_images = self._prepare_input(target)

        prediction_features = self.extractor(prediction_images)
        with torch.no_grad():
            target_features = self.extractor(target_images)

        total = prediction_images.new_zeros(())
        total_weight = sum(self.layer_weights.values())
        for name, layer_weight in self.layer_weights.items():
            total = total + layer_weight * self.criterion(
                prediction_features[name], target_features[name]
            )
        return total / total_weight

    def _prepare_input(self, value: Tensor) -> Tensor:
        if value.ndim == 5:
            batch, frames, channels, height, width = value.shape
            value = value.reshape(batch * frames, channels, height, width)
        elif value.ndim != 4:
            raise ValueError(
                "Input must be an image [B, C, H, W] or video "
                f"[B, T, C, H, W], got {tuple(value.shape)}"
            )
        if value.shape[1] == 1:
            value = value.repeat(1, 3, 1, 1)
        elif value.shape[1] != 3:
            raise ValueError("Input must contain either 1 or 3 channels")
        if not value.is_floating_point():
            raise TypeError("Feature loss inputs must be floating point tensors")
        return (value - self.input_min) / self.input_scale
