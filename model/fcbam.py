"""Frequency/channel attention blocks for YOLO feature maps."""
import torch
from torch import nn


class FreqChannelAttention(nn.Module):
    """Blend pooled low/high spatial-frequency descriptors into channel weights.

    alpha starts at 0.5 and is sigmoid-transformed, so the initial mixture is
    sigmoid(0.5)=0.622 low-frequency and 0.378 high-frequency.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.low_reduce = nn.Conv2d(channels, hidden, 1, bias=False)
        self.low_expand = nn.Conv2d(hidden, channels, 1, bias=False)
        self.high_reduce = nn.Conv2d(channels, hidden, 1, bias=False)
        self.high_expand = nn.Conv2d(hidden, channels, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.pool(x)
        local = nn.functional.avg_pool2d(x, 2, stride=1, padding=1)[:, :, :x.shape[2], :x.shape[3]]
        high = self.pool(x - local)
        alpha = self.alpha.sigmoid()
        low_path = self.low_expand(self.relu(self.low_reduce(low)))
        high_path = self.high_expand(self.relu(self.high_reduce(high)))
        return x * self.sigmoid(alpha * low_path + (1.0 - alpha) * high_path)


class FreqSpatialAttention(nn.Module):
    """Attend to spatial locations using channel mean and maximum maps."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        descriptor = torch.cat((x.mean(1, keepdim=True), x.max(1, keepdim=True).values), dim=1)
        return x * self.sigmoid(self.conv(descriptor))


class FCBAM(nn.Module):
    """Sequential channel then spatial attention; preserves input shape."""
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = FreqChannelAttention(channels, reduction)
        self.spatial_attention = FreqSpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_attention(self.channel_attention(x))
