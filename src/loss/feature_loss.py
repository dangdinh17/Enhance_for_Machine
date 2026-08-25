"""Multi-level ResNet-18 feature loss for images and videos."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


FEATURE_NAMES = ("relu", "layer1", "layer2", "layer3", "layer4")


class ResNet18FeatureExtractor(nn.Module):
    """Extract intermediate features from an ImageNet ResNet-18.

    Args:
        feature_names: ResNet stages returned by :meth:`forward`.
        weights: Torchvision weights to load. Pass ``None`` for randomly
            initialized weights, which is mainly useful for offline tests.
        normalize: Apply ImageNet input normalization before extraction.
    """

    def __init__(
        self,
        feature_names: Sequence[str] = FEATURE_NAMES,
        weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        invalid_names = set(feature_names) - set(FEATURE_NAMES)
        if invalid_names:
            raise ValueError(
                f"Unknown ResNet-18 feature stages: {sorted(invalid_names)}"
            )
        if not feature_names:
            raise ValueError("At least one feature stage must be selected")

        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
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

    def train(self, mode: bool = True) -> ResNet18FeatureExtractor:
        """Keep the frozen backbone in evaluation mode."""
        del mode
        return super().train(False)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        """Return selected feature maps for a 4-D RGB image batch."""
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

        features: dict[str, Tensor] = {}
        value = self.stem(value)
        if "relu" in self.feature_names:
            features["relu"] = value
        value = self.layer1(self.maxpool(value))
        if "layer1" in self.feature_names:
            features["layer1"] = value
        value = self.layer2(value)
        if "layer2" in self.feature_names:
            features["layer2"] = value
        value = self.layer3(value)
        if "layer3" in self.feature_names:
            features["layer3"] = value
        value = self.layer4(value)
        if "layer4" in self.feature_names:
            features["layer4"] = value
        return features


class FeatureLoss(nn.Module):
    """Compute weighted perceptual distance at multiple ResNet-18 stages.

    Inputs may be images shaped ``[B, C, H, W]`` or videos shaped
    ``[B, T, C, H, W]``. One-channel luma data is repeated to RGB. Pixel
    values are expected in ``input_range`` and are mapped to ``[0, 1]``.

    Args:
        layer_weights: Weight of each selected feature stage.
        weights: Torchvision ResNet-18 weights.
        distance: Feature distance, either ``"l1"`` or ``"mse"``.
        input_range: Minimum and maximum input pixel values.
    """

    def __init__(
        self,
        layer_weights: Mapping[str, float] | None = None,
        weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
        distance: str = "l1",
        input_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        selected_weights = dict(
            layer_weights
            if layer_weights is not None
            else {
                "relu": 1.0,
                "layer1": 1.0,
                "layer2": 1.0,
                "layer3": 1.0,
                "layer4": 1.0,
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
        self.extractor = ResNet18FeatureExtractor(
            feature_names=tuple(selected_weights),
            weights=weights,
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
