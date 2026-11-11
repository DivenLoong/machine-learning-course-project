"""List candidate feature extractors and their input sizes / parameter counts."""

from __future__ import annotations

import sys

import timm

sys.stdout.reconfigure(encoding="utf-8")

CANDIDATES = [
    "vit_base_patch14_dinov2.lvd142m",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch16_clip_224.openai",
    "vit_large_patch14_clip_224.openai",
    "vit_base_patch16_224.mae",
    "convnext_base.fb_in22k_ft_in1k",
]


def main() -> None:
    dinov2 = [n for n in timm.list_models(pretrained=True) if "dinov2" in n]
    print(f"dinov2 entries in timm ({len(dinov2)}):")
    for n in dinov2[:12]:
        print("   ", n)
    print()
    for name in CANDIDATES:
        try:
            model = timm.create_model(name, pretrained=False)
            cfg = model.pretrained_cfg or {}
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"{name:<42s} input={cfg.get('input_size')} params={params:.1f}M")
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<42s} ERROR {exc}")


if __name__ == "__main__":
    main()
