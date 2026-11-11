from __future__ import annotations

import sys
import types
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from segmentation_models_pytorch import Unet
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    """Compact four-level U-Net for 20-image retinal vessel segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        self.encoders = nn.ModuleList()
        current = in_channels
        for channel in channels:
            self.encoders.append(ConvBlock(current, channel, dropout=dropout))
            current = channel
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, dropout=dropout)
        self.ups = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[-1] * 2, channels[-1], 2, stride=2),
                nn.ConvTranspose2d(channels[-1], channels[-2], 2, stride=2),
                nn.ConvTranspose2d(channels[-2], channels[-3], 2, stride=2),
                nn.ConvTranspose2d(channels[-3], channels[-4], 2, stride=2),
            ]
        )
        self.decoders = nn.ModuleList(
            [
                ConvBlock(channels[-1] * 2, channels[-1], dropout=dropout),
                ConvBlock(channels[-2] * 2, channels[-2], dropout=dropout),
                ConvBlock(channels[-3] * 2, channels[-3], dropout=dropout),
                ConvBlock(channels[-4] * 2, channels[-4], dropout=dropout),
            ]
        )
        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.head(x)


class TimmUNet(nn.Module):
    """High-resolution U-Net decoder for a hierarchical timm encoder."""

    def __init__(
        self,
        model_name: str = "convnext_tiny.fb_in22k_ft_in1k",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        channels = list(self.encoder.feature_info.channels())
        if len(channels) != 4:
            raise ValueError(f"Expected four encoder stages, got {channels}")
        decoder = [512, 256, 128, 64, 32]
        self.center = ConvBlock(channels[-1], decoder[0], dropout=0.2)
        self.up3 = nn.ConvTranspose2d(decoder[0], decoder[1], 2, stride=2)
        self.dec3 = ConvBlock(channels[-2] + decoder[1], decoder[1], dropout=0.15)
        self.up2 = nn.ConvTranspose2d(decoder[1], decoder[2], 2, stride=2)
        self.dec2 = ConvBlock(channels[-3] + decoder[2], decoder[2], dropout=0.1)
        self.up1 = nn.ConvTranspose2d(decoder[2], decoder[3], 2, stride=2)
        self.dec1 = ConvBlock(channels[-4] + decoder[3], decoder[3], dropout=0.05)
        self.up0 = nn.ConvTranspose2d(decoder[3], decoder[4], 2, stride=2)
        self.final = ConvBlock(decoder[4], decoder[4])
        self.up_full = nn.ConvTranspose2d(decoder[4], decoder[4], 2, stride=2)
        self.head = nn.Conv2d(decoder[4], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1, f2, f3, f4 = self.encoder(x)
        x = self.center(f4)
        x = self.dec3(torch.cat([self.up3(x), f3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), f2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), f1], dim=1))
        x = self.final(self.up0(x))
        return self.head(self.up_full(x))


class _FRConv(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _FRFuse(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.paths = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            ]
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(sum(path(x) for path in self.paths))


class _FRBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        up: bool = False,
        down: bool = False,
        fuse: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fuse = _FRFuse(in_channels, out_channels) if fuse else nn.Identity()
        self.conv = _FRConv(out_channels, dropout)
        self.up = (
            nn.Sequential(
                nn.ConvTranspose2d(
                    out_channels, out_channels // 2, 2, stride=2, bias=False
                ),
                nn.BatchNorm2d(out_channels // 2),
                nn.LeakyReLU(0.1, inplace=True),
            )
            if up
            else None
        )
        self.down = (
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels * 2, 2, stride=2, bias=False),
                nn.BatchNorm2d(out_channels * 2),
                nn.LeakyReLU(0.1, inplace=True),
            )
            if down
            else None
        )

    def forward(self, x: torch.Tensor):
        if self.in_channels != self.out_channels:
            x = self.fuse(x)
        x = self.conv(x)
        outputs = [x]
        if self.up is not None:
            outputs.append(self.up(x))
        if self.down is not None:
            outputs.append(self.down(x))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


class FRUNet(nn.Module):
    """Self-contained adaptation of FR-UNet (MIT), modified to return logits."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, feature_scale: int = 2):
        super().__init__()
        f = [value // feature_scale for value in (64, 128, 256, 512, 1024)]
        dropout = 0.2
        self.b13 = _FRBlock(in_channels, f[0], dropout, down=True)
        self.b12 = _FRBlock(f[0], f[0], dropout, down=True)
        self.b22 = _FRBlock(f[1], f[1], dropout, up=True, down=True)
        self.b11 = _FRBlock(f[0] * 2, f[0], dropout, down=True)
        self.b21 = _FRBlock(f[1] * 2, f[1], dropout, up=True, down=True)
        self.b31 = _FRBlock(f[2], f[2], dropout, up=True, down=True)
        self.b10 = _FRBlock(f[0] * 2, f[0], dropout, down=True)
        self.b20 = _FRBlock(f[1] * 3, f[1], dropout, up=True, down=True)
        self.b30 = _FRBlock(f[2] * 2, f[2], dropout, up=True)
        self.b40 = _FRBlock(f[3], f[3], dropout, up=True)
        self.b11b = _FRBlock(f[0] * 2, f[0], dropout, down=True)
        self.b21b = _FRBlock(f[1] * 3, f[1], dropout, up=True)
        self.b31b = _FRBlock(f[2] * 3, f[2], dropout, up=True)
        self.b12b = _FRBlock(f[0] * 2, f[0], dropout)
        self.b22b = _FRBlock(f[1] * 3, f[1], dropout, up=True)
        self.b13b = _FRBlock(f[0] * 2, f[0], dropout)
        self.output_layers = nn.ModuleList(
            [nn.Conv2d(f[0], num_classes, 1) for _ in range(5)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x13, d13 = self.b13(x)
        x12, d12 = self.b12(x13)
        x22, u22, d22 = self.b22(d13)
        x11, d11 = self.b11(torch.cat([x12, u22], dim=1))
        x21, u21, d21 = self.b21(torch.cat([d12, x22], dim=1))
        x31, u31, d31 = self.b31(d22)
        x10, d10 = self.b10(torch.cat([x11, u21], dim=1))
        x20, u20, d20 = self.b20(torch.cat([d11, x21, u31], dim=1))
        x30, u30 = self.b30(torch.cat([d21, x31], dim=1))
        _, u40 = self.b40(d31)
        x11b, d11b = self.b11b(torch.cat([x10, u20], dim=1))
        x21b, u21b = self.b21b(torch.cat([d10, x20, u30], dim=1))
        _, u31b = self.b31b(torch.cat([d20, x30, u40], dim=1))
        x12b = self.b12b(torch.cat([x11b, u21b], dim=1))
        _, u22b = self.b22b(torch.cat([d11b, x21b, u31b], dim=1))
        x13b = self.b13b(torch.cat([x12b, u22b], dim=1))
        outputs = [x11, x10, x11b, x12b, x13b]
        return torch.stack(
            [layer(feature) for layer, feature in zip(self.output_layers, outputs)], dim=0
        ).mean(dim=0)


class _LadderBasic(nn.Module):
    def __init__(self, in_channels: int, channels: int):
        super().__init__()
        self.pre = nn.Conv2d(in_channels, channels, 3, padding=1) if in_channels != channels else nn.Identity()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.activation(self.pre(x)) if not isinstance(self.pre, nn.Identity) else x
        out = self.conv(residual)
        out = self.activation(out)
        out = self.drop(out)
        out = self.conv(out)
        return self.activation(out + residual)


class _InitialLadder(nn.Module):
    def __init__(self, planes: int, layers: int, in_channels: int):
        super().__init__()
        self.layers = layers
        self.inconv = nn.Sequential(
            nn.Conv2d(in_channels, planes, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for level in range(layers):
            channels = planes * 2**level
            self.down_blocks.append(_LadderBasic(channels, channels))
            self.down_convs.append(nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1))
        self.bottom = _LadderBasic(planes * 2**layers, planes * 2**layers)
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for index in range(layers):
            upper = planes * 2 ** (layers - index - 1)
            self.up_convs.append(
                nn.ConvTranspose2d(upper * 2, upper, 3, stride=2, padding=1, output_padding=1)
            )
            self.up_blocks.append(_LadderBasic(upper, upper))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.inconv(x)
        down_outputs = []
        for block, downsample in zip(self.down_blocks, self.down_convs):
            x = block(x)
            down_outputs.append(x)
            x = F.relu(downsample(x))
        x = self.bottom(x)
        outputs = [x]
        for index, (upsample, block) in enumerate(zip(self.up_convs, self.up_blocks)):
            x = upsample(x) + down_outputs[-index - 1]
            x = block(x)
            outputs.append(x)
        return outputs


class _LadderBlock(nn.Module):
    def __init__(self, planes: int, layers: int):
        super().__init__()
        self.layers = layers
        self.inconv = _LadderBasic(planes, planes)
        self.down_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for level in range(layers):
            channels = planes * 2**level
            self.down_blocks.append(_LadderBasic(channels, channels))
            self.down_convs.append(nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1))
        self.bottom = _LadderBasic(planes * 2**layers, planes * 2**layers)
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for index in range(layers):
            upper = planes * 2 ** (layers - index - 1)
            self.up_convs.append(
                nn.ConvTranspose2d(upper * 2, upper, 3, stride=2, padding=1, output_padding=1)
            )
            self.up_blocks.append(_LadderBasic(upper, upper))

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        x = self.inconv(inputs[-1])
        down_outputs = []
        for index, (block, downsample) in enumerate(zip(self.down_blocks, self.down_convs)):
            x = block(x + inputs[-index - 1])
            down_outputs.append(x)
            x = F.relu(downsample(x))
        x = self.bottom(x)
        outputs = [x]
        for index, (upsample, block) in enumerate(zip(self.up_convs, self.up_blocks)):
            x = upsample(x) + down_outputs[-index - 1]
            x = block(x)
            outputs.append(x)
        return outputs


class LadderNet(nn.Module):
    """LadderNet adaptation from the Apache-2.0 VesselSeg-Pytorch toolkit."""

    def __init__(self, in_channels: int = 3, planes: int = 16, layers: int = 4):
        super().__init__()
        self.initial = _InitialLadder(planes, layers, in_channels)
        self.middle = _LadderBlock(planes, layers)
        self.head = nn.Conv2d(planes, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.initial(x)
        features = self.middle(features)
        return self.head(features[-1])


class _SGLConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        if in_channels == 3:
            layers = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.PReLU(),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
            ]
        else:
            layers = [
                nn.PReLU(),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.PReLU(),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
            ]
        super().__init__(*layers)


class _SGLEncoderDecoder(nn.Module):
    def __init__(self, output_channels: int):
        super().__init__()
        self.conv1 = _SGLConvBlock(3, 32)
        self.conv2 = _SGLConvBlock(32, 64)
        self.conv3 = _SGLConvBlock(64, 128)
        self.conv4 = _SGLConvBlock(128, 256)
        self.conv5 = _SGLConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = _SGLConvBlock(512, 256)
        self.conv7 = _SGLConvBlock(256, 128)
        self.conv8 = _SGLConvBlock(128, 64)
        self.conv9 = _SGLConvBlock(64, 32)
        self.conv11 = _SGLConvBlock(35, output_channels)
        self.last_act = nn.PReLU()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))
        u4 = self.conv6(torch.cat([self.upconv4(x5), x4], dim=1))
        u3 = self.conv7(torch.cat([self.upconv3(u4), x3], dim=1))
        u2 = self.conv8(torch.cat([self.upconv2(u3), x2], dim=1))
        u1 = self.conv9(torch.cat([self.upconv1(u2), x1], dim=1))
        return torch.cat([u1, x], dim=1)


class _SGLEnhanceNet(_SGLEncoderDecoder):
    def __init__(self):
        super().__init__(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv11(self.features(x / 255.0)))


class _SGLSegNet(_SGLEncoderDecoder):
    def __init__(self):
        super().__init__(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv11(self.features(x)))


class SGLVesselNet(nn.Module):
    """Study Group Learning network, adapted from the MIT reference code."""

    def __init__(self):
        super().__init__()
        self.s1 = _SGLEnhanceNet()
        self.s2 = _SGLSegNet()

    def forward_probability(self, x: torch.Tensor) -> torch.Tensor:
        return self.s2(self.s1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probability = self.forward_probability(x).float()
        return torch.logit(probability.clamp(1e-4, 1.0 - 1e-4))


class _LWConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.pool = nn.MaxPool2d(2) if pool else None
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            x = self.pool(x)
        return self.block(x) + self.shortcut(x)


class _LWUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_layer = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv_bridge_layer = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        self.conv_layer = _LWConvBlock(out_channels * 2, out_channels, pool=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv_layer(
            torch.cat([self.up_layer(x), self.conv_bridge_layer(skip)], dim=1)
        )


class _LWUNet(nn.Module):
    def __init__(self, in_channels: int, layers: tuple[int, int, int]):
        super().__init__()
        self.first = _LWConvBlock(in_channels, layers[0], pool=False)
        self.down_path = nn.ModuleList(
            [
                _LWConvBlock(layers[index], layers[index + 1], pool=True)
                for index in range(len(layers) - 1)
            ]
        )
        reversed_layers = tuple(reversed(layers))
        self.up_path = nn.ModuleList(
            [
                _LWUpBlock(reversed_layers[index], reversed_layers[index + 1])
                for index in range(len(layers) - 1)
            ]
        )
        self.final = nn.Conv2d(layers[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        skips = []
        for layer in self.down_path:
            skips.append(x)
            x = layer(x)
        for layer, skip in zip(self.up_path, reversed(skips)):
            x = layer(x, skip)
        return self.final(x)


class LittleWNet(nn.Module):
    """W-Net implementation adapted from the MIT-licensed lwnet repository."""

    def __init__(self):
        super().__init__()
        self.unet1 = _LWUNet(3, (8, 16, 32))
        self.unet2 = _LWUNet(4, (8, 16, 32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.unet1(x)
        return self.unet2(torch.cat([x, first], dim=1))


def build_model(config: dict) -> nn.Module:
    name = str(config.get("name", "resnet34"))
    if name == "small_unet":
        return SmallUNet(
            in_channels=int(config.get("in_channels", 1)),
            base_channels=int(config.get("base_channels", 32)),
            dropout=float(config.get("dropout", 0.05)),
        )
    if name in {"resnet34", "efficientnet_b4"}:
        return Unet(
            encoder_name=name,
            encoder_weights="imagenet" if config.get("pretrained", True) else None,
            in_channels=3,
            classes=1,
            decoder_attention_type=config.get("decoder_attention_type"),
        )
    if name == "convnext_tiny":
        return TimmUNet(
            model_name=config.get("timm_name", "convnext_tiny.fb_in22k_ft_in1k"),
            pretrained=bool(config.get("pretrained", True)),
        )
    if name == "fr_unet":
        return FRUNet(
            in_channels=int(config.get("in_channels", 3)),
            feature_scale=int(config.get("feature_scale", 2)),
        )
    if name == "laddernet":
        return LadderNet(planes=int(config.get("planes", 16)))
    if name == "sgl":
        return SGLVesselNet()
    if name == "lwnet":
        return LittleWNet()
    raise ValueError(f"Unknown model: {name}")


def _remap_official_fr_state(state: dict) -> dict:
    block_names = {
        "block1_3": "b13",
        "block1_2": "b12",
        "block1_1": "b11",
        "block10": "b10",
        "block11": "b11b",
        "block12": "b12b",
        "block13": "b13b",
        "block2_2": "b22",
        "block2_1": "b21",
        "block20": "b20",
        "block21": "b21b",
        "block22": "b22b",
        "block3_1": "b31",
        "block30": "b30",
        "block31": "b31b",
        "block40": "b40",
    }
    mapped = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.startswith("fuse."):
            continue
        for source_name, target_name in block_names.items():
            prefix = f"{source_name}."
            if key.startswith(prefix):
                key = target_name + "." + key[len(prefix) :]
                break
        for index in range(1, 6):
            prefix = f"final{index}."
            if key.startswith(prefix):
                key = f"output_layers.{index - 1}." + key[len(prefix) :]
                break
        key = key.replace(".fuse.conv11.", ".fuse.paths.0.")
        key = key.replace(".fuse.conv33_di.", ".fuse.paths.2.")
        key = key.replace(".fuse.conv33.", ".fuse.paths.1.")
        key = key.replace(".conv.conv.", ".conv.block.")
        key = key.replace(".down.down.", ".down.")
        key = key.replace(".up.up.", ".up.")
        mapped[key] = value
    return mapped


def load_initial_weights(model: nn.Module, path: str | Path) -> dict:
    """Load a trusted checkpoint and adapt upstream parameter names to the local model."""
    # Legacy checkpoints may pickle bunch.Bunch metadata.
    module = types.ModuleType("bunch")

    class Bunch(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    module.Bunch = Bunch
    previous = sys.modules.get("bunch")
    sys.modules["bunch"] = module
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        if previous is None:
            sys.modules.pop("bunch", None)
        else:
            sys.modules["bunch"] = previous
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state = checkpoint["model"]
    else:
        state = checkpoint.get(
            "state_dict", checkpoint.get("model_state_dict", checkpoint)
        )
    if isinstance(model, FRUNet):
        state = _remap_official_fr_state(state)
    if isinstance(model, LittleWNet):
        mapped = {}
        for key, value in state.items():
            key = key.replace(".up_layer.block.0.", ".up_layer.")
            key = key.replace(".conv_bridge_layer.block.", ".conv_bridge_layer.")
            mapped[key] = value
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        bad_missing = [
            key for key in missing if not key.endswith("num_batches_tracked")
        ]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"W-Net checkpoint mismatch: missing={bad_missing}, "
                f"unexpected={unexpected}"
            )
    else:
        model.load_state_dict(state, strict=True)
    return {
        "initial_checkpoint": str(Path(path).resolve()),
        "initial_epoch": checkpoint.get("epoch"),
        "initial_monitor_best": checkpoint.get("monitor_best"),
    }
