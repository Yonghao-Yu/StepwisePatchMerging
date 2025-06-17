import math

import torch
import torch.nn as nn


__all__ = ["StepwisePatchMerging"]


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x


class MultiScaleAggregation(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, expand_ratio: int = 2):
        super().__init__()
        self.kernel_sizes = [3, 5, 7, 9]

        self.num_kernels = len(self.kernel_sizes)

        self.local_dim = in_dim // self.num_kernels

        for kernel_size in self.kernel_sizes:
            local_conv = nn.Conv2d(
                in_channels=self.local_dim,
                out_channels=self.local_dim,
                kernel_size=kernel_size,
                stride=1, padding=kernel_size // 2,
                groups=self.local_dim
            )
            setattr(self, f"local_conv_{kernel_size}", local_conv)

        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * expand_ratio, kernel_size=1, groups=self.local_dim),
            nn.BatchNorm2d(in_dim * expand_ratio),
            nn.GELU(),
            nn.Conv2d(in_dim * expand_ratio, out_dim, kernel_size=1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        x_ = x.view(B, self.num_kernels, self.local_dim, H, W).permute(1, 0, 2, 3, 4)
        for i, kernel_size in enumerate(self.kernel_sizes):
            local_conv = getattr(self, f"local_conv_{kernel_size}")
            x_i = local_conv(x_[i]).unsqueeze(2)
            if i == 0:
                x_out = x_i
            else:
                x_out = torch.cat([x_out, x_i], dim=2)
        x_out = x_out.reshape(B, C, H, W)
        x_out = self.proj(x_out)
        return x_out


class GuidedLocalEnhancement(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        # Guide Token Generator
        self.gtg = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, stride=2, padding=3, groups=dim)
        )

        self.attn = Attention(dim=dim)

        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # (B * N // 4, 4, C)
        windows = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 3, 5, 1).contiguous().view(B * N // 4, 4, C)

        # (B, C, H // 2, W // 2) -> (B, N // 4, C) -> (B * N // 4, C) -> (B * N // 4, 1, C)
        guide_tokens = self.gtg(x).flatten(2).transpose(-2, -1).reshape(B * N // 4, C).unsqueeze(1)

        # (B * N // 4, 5, C)
        x = torch.cat([guide_tokens, windows], dim=1)

        # (B * N // 4, C) -> (B, H // 2, W // 2, C)
        x = self.attn(self.norm0(x))[:, 0].view(B, H // 2, W // 2, C)

        return self.norm1(x)


class StepwisePatchMerging(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.msa = MultiScaleAggregation(in_dim=in_dim, out_dim=out_dim)

        self.gle = GuidedLocalEnhancement(dim=out_dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        return self.gle(self.msa(x))  # (B, H, W, C)


if __name__ == "__main__":
    spm = StepwisePatchMerging(in_dim=64, out_dim=128)
    x = torch.randn(2, 56, 56, 64)
    y = spm(x)
    print(y.shape)