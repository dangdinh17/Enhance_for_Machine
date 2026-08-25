"""PyTorch implementation of the official TensorFlow MFQEv2 inference graph."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, activation: str = "prelu") -> None:
        super().__init__()
        self.tensorflow_same = stride > 1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=0 if self.tensorflow_same else kernel_size // 2,
        )
        if activation == "prelu":
            self.activation: nn.Module = nn.PReLU(out_channels)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, value: Tensor) -> Tensor:
        if self.tensorflow_same:
            height, width = value.shape[-2:]
            stride_h, stride_w = self.conv.stride
            kernel_h, kernel_w = self.conv.kernel_size
            output_h = (height + stride_h - 1) // stride_h
            output_w = (width + stride_w - 1) // stride_w
            pad_h = max((output_h - 1) * stride_h + kernel_h - height, 0)
            pad_w = max((output_w - 1) * stride_w + kernel_w - width, 0)
            value = nn.functional.pad(
                value,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
        return self.activation(self.conv(value))


def tensorflow_bilinear_warp(image: Tensor, flow: Tensor) -> Tensor:
    """Reproduce MFQEv2's TensorFlow sampler, including its border behavior.

    ``flow`` is ordered as (dx, dy) and is expressed in units of 64 pixels,
    exactly as in the original ``transformer`` implementation.
    """
    batch, channels, height, width = image.shape
    if flow.shape != (batch, 2, height, width):
        raise ValueError(
            f"flow must have shape {(batch, 2, height, width)}, got {tuple(flow.shape)}"
        )

    dtype = flow.dtype
    device = flow.device
    base_y, base_x = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    x = base_x.unsqueeze(0) + flow[:, 0] * 64.0
    y = base_y.unsqueeze(0) + flow[:, 1] * 64.0

    x0 = torch.floor(x).to(torch.long)
    x1 = x0 + 1
    y0 = torch.floor(y).to(torch.long)
    y1 = y0 + 1
    # The reference clips integer sample locations before calculating weights.
    x0 = x0.clamp(0, width - 1)
    x1 = x1.clamp(0, width - 1)
    y0 = y0.clamp(0, height - 1)
    y1 = y1.clamp(0, height - 1)

    flat = image.permute(0, 2, 3, 1).reshape(batch * height * width, channels)
    offsets = (
        torch.arange(batch, device=device).view(batch, 1, 1) * height * width
    )

    def gather(sample_x: Tensor, sample_y: Tensor) -> Tensor:
        indices = (offsets + sample_y * width + sample_x).reshape(-1)
        return flat[indices].reshape(batch, height, width, channels)

    ia = gather(x0, y0)
    ib = gather(x0, y1)
    ic = gather(x1, y0)
    id_ = gather(x1, y1)
    x0f, x1f = x0.to(dtype), x1.to(dtype)
    y0f, y1f = y0.to(dtype), y1.to(dtype)
    wa = ((x1f - x) * (y1f - y)).unsqueeze(-1)
    wb = ((x1f - x) * (y - y0f)).unsqueeze(-1)
    wc = ((x - x0f) * (y1f - y)).unsqueeze(-1)
    wd = ((x - x0f) * (y - y0f)).unsqueeze(-1)
    output = wa * ia + wb * ib + wc * ic + wd * id_
    return output.permute(0, 3, 1, 2).contiguous()


class EasyFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = ConvAct(2, 24, 5, stride=2)
        self.c2 = ConvAct(24, 24, 3)
        self.c3 = ConvAct(24, 24, 5, stride=2)
        self.c4 = ConvAct(24, 24, 3)
        self.c5 = ConvAct(24, 32, 3, activation="tanh")
        self.s1 = ConvAct(5, 24, 5, stride=2)
        self.s2 = ConvAct(24, 24, 3)
        self.s3 = ConvAct(24, 24, 3)
        self.s4 = ConvAct(24, 24, 3)
        self.s5 = ConvAct(24, 8, 3, activation="tanh")
        self.a1 = ConvAct(5, 24, 3)
        self.a2 = ConvAct(24, 24, 3)
        self.a3 = ConvAct(24, 24, 3)
        self.a4 = ConvAct(24, 24, 3)
        self.a5 = ConvAct(24, 2, 3, activation="tanh")

    def forward(self, current: Tensor, neighbor: Tensor) -> Tensor:
        height, width = current.shape[-2:]
        if height % 4 or width % 4:
            raise ValueError("MFQEv2 EasyFlow input height and width must be divisible by 4")
        inputs = torch.cat((current, neighbor), dim=1)
        c5 = self.c5(self.c4(self.c3(self.c2(self.c1(inputs)))))
        coarse_flow = nn.functional.pixel_shuffle(c5, 4)
        warped1 = tensorflow_bilinear_warp(neighbor, coarse_flow)

        packed = torch.cat((inputs, coarse_flow, warped1), dim=1)
        s5 = self.s5(self.s4(self.s3(self.s2(self.s1(packed)))))
        medium_flow = coarse_flow + nn.functional.pixel_shuffle(s5, 2)
        warped2 = tensorflow_bilinear_warp(neighbor, medium_flow)

        packed2 = torch.cat((inputs, medium_flow, warped2), dim=1)
        fine_flow = medium_flow + self.a5(
            self.a4(self.a3(self.a2(self.a1(packed2))))
        )
        return tensorflow_bilinear_warp(neighbor, fine_flow)


class ReconstructionLowQP(nn.Module):
    """Network used by official QP 22/27/32 checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        for frame in range(1, 4):
            setattr(self, f"conv3_{frame}", ConvAct(1, 32, 3))
            setattr(self, f"conv5_{frame}", ConvAct(1, 32, 5))
            setattr(self, f"conv7_{frame}", ConvAct(1, 32, 7))
        channels = [288, 32, 32, 32, 32, 32, 32, 32]
        for index, in_channels in enumerate(channels, start=1):
            out_channels = 16 if index == 8 else 32
            setattr(self, f"cconv{index}", ConvAct(in_channels, out_channels, 3))
        self.cout = ConvAct(16, 1, 3, activation="none")

    def _features(self, value: Tensor, frame: int) -> Tensor:
        return torch.cat(
            tuple(getattr(self, f"conv{size}_{frame}")(value) for size in (3, 5, 7)),
            dim=1,
        )

    def forward(self, previous: Tensor, current: Tensor, following: Tensor) -> Tensor:
        value = torch.cat(
            (self._features(previous, 1), self._features(current, 2),
             self._features(following, 3)),
            dim=1,
        )
        for index in range(1, 9):
            value = getattr(self, f"cconv{index}")(value)
        return current + self.cout(value)


class ConvBnPrelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.activation = nn.PReLU(out_channels)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(self.bn(self.conv(value)))


class ReconstructionHighQP(nn.Module):
    """Dense BatchNorm network used by official QP 37/42 checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        for frame in range(1, 4):
            for size in (3, 5, 7):
                setattr(self, f"c{size}_{frame}", ConvAct(1, 32, size))
        self.c1 = ConvBnPrelu(288, 32)
        self.c2 = ConvBnPrelu(32, 32)
        self.c3 = ConvBnPrelu(64, 32)
        self.c4 = ConvBnPrelu(96, 32)
        self.c5 = ConvBnPrelu(128, 32)
        self.c6 = ConvBnPrelu(32, 1)

    def _features(self, value: Tensor, frame: int) -> Tensor:
        return torch.cat(
            tuple(getattr(self, f"c{size}_{frame}")(value) for size in (3, 5, 7)),
            dim=1,
        )

    def forward(self, previous: Tensor, current: Tensor, following: Tensor) -> Tensor:
        merged = torch.cat(
            (self._features(previous, 1), self._features(current, 2),
             self._features(following, 3)),
            dim=1,
        )
        c1 = self.c1(merged)
        c2 = self.c2(c1)
        c3 = self.c3(torch.cat((c1, c2), dim=1))
        c4 = self.c4(torch.cat((c1, c2, c3), dim=1))
        c5 = self.c5(torch.cat((c1, c2, c3, c4), dim=1))
        return current + self.c6(c5)


class MFQEv2(nn.Module):
    def __init__(self, model_qp: int) -> None:
        super().__init__()
        if model_qp not in (22, 27, 32, 37, 42):
            raise ValueError(f"Unsupported official MFQEv2 model QP: {model_qp}")
        self.model_qp = model_qp
        self.easyflow = EasyFlow()
        self.reconstruction: nn.Module
        if model_qp in (37, 42):
            self.reconstruction = ReconstructionHighQP()
        else:
            self.reconstruction = ReconstructionLowQP()

    def forward(self, previous: Tensor, current: Tensor, following: Tensor) -> Tensor:
        previous_warped = self.easyflow(current, previous)
        following_warped = self.easyflow(current, following)
        return self.reconstruction(previous_warped, current, following_warped)


def load_mfqev2(checkpoint_path: str, device: torch.device | str = "cpu") -> MFQEv2:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_qp = int(checkpoint["model_qp"])
    model = MFQEv2(model_qp)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()
