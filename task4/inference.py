from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from drive_vessel.data import list_drive_samples
from drive_vessel.inference import (
    load_probability_archive,
    postprocess_mask,
    predict_probabilities,
    save_probability_archive,
)
from drive_vessel.rle import masks_to_dataframe
from drive_vessel.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the frozen FR-UNet submission without test labels."
    )
    parser.add_argument("--config", default="configs/fr_unet_official_patch_drive.yaml")
    parser.add_argument(
        "--checkpoint",
        default="weights/fr_unet_drive_epoch40_packaged.pt",
    )
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--mask-dir", default="runs/final_inference/masks")
    parser.add_argument(
        "--probability-archive",
        default=(
            "snapshots/best_public_0.81766_fr_official/probabilities.npz"
        ),
        help="Frozen probability archive; use an empty value to regenerate.",
    )
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--threshold-low", type=float, default=0.40)
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    samples = {
        sample.image_id: sample
        for sample in list_drive_samples(config["data"]["root"], "test")
    }
    if args.probability_archive and Path(args.probability_archive).is_file():
        probabilities = load_probability_archive(args.probability_archive)
        metadata = {
            "probability_archive": str(Path(args.probability_archive).resolve()),
            "tta": args.tta,
            "split": "test",
        }
    else:
        probabilities, metadata = predict_probabilities(
            config,
            [args.checkpoint],
            split="test",
            tta=args.tta,
        )
        if args.probability_archive:
            save_probability_archive(Path(args.probability_archive), probabilities, metadata)
    mask_dir = Path(args.mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    masks = {}
    for image_id in sorted(probabilities, key=int):
        mask = postprocess_mask(
            probabilities[image_id],
            samples[image_id].image_path,
            int(config["data"]["size"]),
            threshold=args.threshold,
            min_area=args.min_area,
            fov_mask=True,
            threshold_low=args.threshold_low,
            dti=args.threshold_low < args.threshold,
        )
        if mask.shape != (512, 512):
            raise ValueError(f"Unexpected mask shape for {image_id}: {mask.shape}")
        masks[image_id] = mask
        cv2.imwrite(str(mask_dir / f"{image_id}.png"), mask * 255)

    frame = masks_to_dataframe(masks)
    if len(frame) != 20 or list(frame.columns) != ["Id", "Predicted"]:
        raise ValueError("Submission schema or image count is invalid")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} validated rows to {args.output}")
    print(metadata)


if __name__ == "__main__":
    main()
