"""ImageNet backbones and a grayscale ResEmoteNet adaptation (arXiv:2409.10545)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLS = 6


class SE(nn.Module):
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        hidden = max(1, ch // r)
        self.fc1 = nn.Linear(ch, hidden)
        self.fc2 = nn.Linear(hidden, ch)

    def forward(self, x):
        s = x.mean(dim=(2, 3))
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s[:, :, None, None]


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, pool: bool = True, drop: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.pool(self.body(x)))


class ResSEBlock(nn.Module):
    """Residual block with squeeze-and-excitation and a shortcut projection when channels differ."""

    def __init__(self, cin: int, cout: int, drop: float = 0.2, se_ratio: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.se = SE(cout, se_ratio)
        self.short = (
            nn.Identity()
            if cin == cout
            else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        )
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + self.short(x))
        return self.drop(out)


class ResEmoteNet(nn.Module):
    def __init__(self, num_classes: int = N_CLS, width: float = 1.0, drop: float = 0.2):
        super().__init__()
        w = lambda c: max(8, int(round(c * width)))
        self.stem = nn.Sequential(
            ConvBlock(1, w(64), pool=True, drop=drop * 0.25),
            ConvBlock(w(64), w(128), pool=True, drop=drop * 0.25),
            ConvBlock(w(128), w(256), pool=True, drop=drop * 0.5),
        )
        self.res = nn.Sequential(
            ResSEBlock(w(256), w(256), drop=drop),
            ResSEBlock(w(256), w(512), drop=drop),
            ResSEBlock(w(512), w(512), drop=drop),
        )
        self.head_drop = nn.Dropout(drop)
        self.fc = nn.Linear(w(512), num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.res(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(self.head_drop(x))


CUSTOM = {"resemotenet": (ResEmoteNet, 1, 48)}


def is_custom(name: str) -> bool:
    return name.split(":")[0] in CUSTOM


def build_model(name: str, num_classes: int = N_CLS, drop_path: float = 0.1, pretrained: bool = True):
    """Return the classifier, its input channel count and the default input size."""
    key = name.split(":")[0]
    if key in CUSTOM:
        cls, in_ch, res = CUSTOM[key]
        kw = {}
        for part in name.split(":")[1:]:
            k, _, v = part.partition("=")
            kw[k] = float(v) if "." in v or v.isdigit() else v
        if "width" in kw:
            kw["width"] = float(kw["width"])
        return cls(num_classes=num_classes, **kw), in_ch, res

    import timm

    try:
        model = timm.create_model(
            name, pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path
        )
    except TypeError:
        # Architectures without stochastic depth (e.g. DenseNet) reject drop_path_rate.
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    return model, 3, 224


def head_and_backbone_params(model):
    """Split parameters into (head, backbone) for a differential learning rate."""
    head = []
    for attr in ("head", "fc", "classifier"):
        mod = getattr(model, attr, None)
        if isinstance(mod, nn.Module):
            head.extend(mod.parameters())
    seen, uniq = set(), []
    for p in head:
        if id(p) not in seen:
            seen.add(id(p))
            uniq.append(p)
    head = uniq
    head_ids = {id(p) for p in head}
    backbone = []
    for p in model.parameters():
        if id(p) not in head_ids:
            backbone.append(p)
    return head, backbone
