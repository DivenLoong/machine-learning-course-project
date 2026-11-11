"""Infer the native-resolution CLAHE+Frangi U-Net and emit 512 masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "reti_ves_seg"))

from drive_vessel.data import (  # noqa: E402
    estimate_fov,
    list_drive_samples,
    preprocess_image,
    read_binary_mask,
    read_rgb_native,
)
from drive_vessel.inference import postprocess_mask  # noqa: E402
from drive_vessel.metrics import binary_metrics, summarize_metrics  # noqa: E402
from drive_vessel.rle import masks_to_dataframe  # noqa: E402
from unet import UNet  # noqa: E402


def pad_to(image: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    if height > size or width > size:
        raise ValueError(f"Image {height}x{width} exceeds {size}")
    top = (size - height) // 2
    left = (size - width) // 2
    padding = [(top, size - height - top), (left, size - width - left)]
    if image.ndim == 3:
        padding.append((0, 0))
    return np.pad(image, padding, mode="constant"), (top, left)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--native-pad-size", type=int, default=608)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--data-root", default="2026task-4/segmentation"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=2, base_channels=64, return_logits=True).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output = ROOT / args.output
    mask_dir = output / "masks"
    output.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    samples = list_drive_samples(ROOT / args.data_root, args.split)
    probabilities: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    rows = []

    for index, sample in enumerate(samples, 1):
        rgb = read_rgb_native(sample.image_path)
        input_image = preprocess_image(rgb, "clahe_frangi")
        padded, (top, left) = pad_to(input_image, args.native_pad_size)
        tensor = torch.from_numpy(
            np.ascontiguousarray(padded.transpose(2, 0, 1).astype(np.float32))
        )[None].to(device)
        probability = torch.sigmoid(model(tensor).float())[0, 0].cpu().numpy()
        height, width = rgb.shape[:2]
        probability = probability[
            top : top + height, left : left + width
        ]
        probability = cv2.resize(
            probability, (512, 512), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        probabilities[sample.image_id] = probability
        mask = postprocess_mask(
            probability,
            sample.image_path,
            512,
            args.threshold,
            min_area=0,
            fov_mask=True,
        )
        masks[sample.image_id] = mask
        cv2.imwrite(str(mask_dir / f"{sample.image_id}.png"), mask * 255)

        if sample.mask_path is not None:
            target = read_binary_mask(sample.mask_path, 512) > 0.5
            rows.append(
                {"id": sample.image_id, **binary_metrics(target, mask > 0)}
            )
        print(f"[{index}/{len(samples)}] {sample.image_path.name}")

    np.savez_compressed(
        output / "probabilities.npz",
        **{key: value.astype(np.float16) for key, value in probabilities.items()},
    )
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "threshold": args.threshold,
        "native_pad_size": args.native_pad_size,
        "preprocessing": "clahe_frangi",
        "evaluation": summarize_metrics(rows) if rows else None,
    }
    (output / "prediction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if args.split == "test":
        masks_to_dataframe(masks).to_csv(output / "submission.csv", index=False)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
