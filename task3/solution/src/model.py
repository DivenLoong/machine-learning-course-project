"""Fovea heatmaps from a timm encoder and a U-Net decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, padding_mode: str = "zeros"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class HeatmapNet(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True,
                 drop_path_rate: float = 0.0, drop_rate: float = 0.0,
                 padding_mode: str = "zeros", use_reg: bool = False,
                 aux_head: bool = False, in_chans: int = 3):
        super().__init__()
        # Select strides 4, 8, 16 and 32. EfficientNet also exposes a stride-2 stem.
        out_indices = (0, 1, 2, 3) if "convnext" in backbone.lower() else (1, 2, 3, 4)
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, out_indices=out_indices,
            drop_path_rate=drop_path_rate, drop_rate=drop_rate, in_chans=in_chans,
        )
        chs = self.encoder.feature_info.channels()  # strides 4, 8, 16, 32

        self.dec3 = ConvBlock(chs[3] + chs[2], 128, padding_mode)
        self.dec2 = ConvBlock(128 + chs[1], 64, padding_mode)
        self.dec1 = ConvBlock(64 + chs[0], 32, padding_mode)
        self.head = nn.Conv2d(32, 1, 1)
        self.aux_head = nn.Conv2d(32, 1, 1) if aux_head else None
        self.use_reg = bool(use_reg)
        if self.use_reg:
            self.reg_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(chs[3], 256),
                nn.SiLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 2),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor, return_reg: bool = False,
                return_aux: bool = False):
        c1, c2, c3, c4 = self.encoder(x)
        d = F.interpolate(c4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec3(torch.cat([d, c3], dim=1))
        d = F.interpolate(d, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec2(torch.cat([d, c2], dim=1))
        d = F.interpolate(d, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec1(torch.cat([d, c1], dim=1))
        hm = torch.sigmoid(self.head(d))
        if return_aux:
            if self.aux_head is None:
                raise RuntimeError("Model was built without an auxiliary head.")
            aux = torch.sigmoid(self.aux_head(d))
            if return_reg:
                return hm, self.reg_head(c4), aux
            return hm, aux
        if return_reg:
            if not self.use_reg:
                raise RuntimeError("Model was built without a regression head.")
            return hm, self.reg_head(c4)
        return hm


def soft_argmax(hm: torch.Tensor, beta: float = 25.0) -> torch.Tensor:
    """Differentiable global soft-argmax, returns normalized (u, v)."""
    b, _, h, w = hm.shape
    flat = hm.reshape(b, -1)
    p = torch.softmax(flat * beta, dim=1).reshape(b, 1, h, w)
    xs = (torch.arange(w, device=hm.device, dtype=hm.dtype) + 0.5) / w
    ys = (torch.arange(h, device=hm.device, dtype=hm.dtype) + 0.5) / h
    u = (p.sum(dim=2) * xs).sum(dim=2)
    v = (p.sum(dim=3) * ys).sum(dim=2)
    return torch.cat([u, v], dim=1)


def mask_centroid(hm: torch.Tensor, power: float = 2.0) -> torch.Tensor:
    """Differentiable global centroid of a (soft) segmentation mask."""
    b, _, h, w = hm.shape
    weight = hm.clamp_min(0).pow(power)
    denom = weight.sum(dim=(2, 3)).clamp_min(1e-6)
    xs = (torch.arange(w, device=hm.device, dtype=hm.dtype) + 0.5) / w
    ys = (torch.arange(h, device=hm.device, dtype=hm.dtype) + 0.5) / h
    u = (weight.sum(dim=2) * xs).sum(dim=2) / denom
    v = (weight.sum(dim=3) * ys).sum(dim=2) / denom
    return torch.cat([u, v], dim=1)


def heatmap_to_coord(hm: torch.Tensor, window: int = 17) -> torch.Tensor:
    """Decode a square heatmap near its peak using normalized pixel-centre coordinates (u, v)."""
    b, _, r, _ = hm.shape
    flat = hm.reshape(b, -1)
    idx = flat.argmax(dim=1)
    out = torch.empty(b, 2, device=hm.device, dtype=hm.dtype)
    half = window // 2
    for k in range(b):
        ay, ax = int(idx[k] // r), int(idx[k] % r)
        y0, y1 = max(0, ay - half), min(r, ay + half + 1)
        x0, x1 = max(0, ax - half), min(r, ax + half + 1)
        w = hm[k, 0, y0:y1, x0:x1].clamp_min(1e-8)
        w = w / w.sum()
        xs = torch.arange(x0, x1, device=hm.device, dtype=hm.dtype) + 0.5
        ys = torch.arange(y0, y1, device=hm.device, dtype=hm.dtype) + 0.5
        out[k, 0] = (w.sum(dim=0) * xs).sum() / r
        out[k, 1] = (w.sum(dim=1) * ys).sum() / r
    return out
