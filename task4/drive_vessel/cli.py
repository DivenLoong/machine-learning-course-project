from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .audit import audit_dataset
from .data import list_drive_samples
from .engine import train_fold
from .inference import (
    evaluate_checkpoint,
    load_probability_archive,
    postprocess_mask,
    predict_probabilities,
    save_probability_archive,
)
from .models import build_model, load_initial_weights
from .optimize import combine_oof_archives, optimize_threshold_and_area
from .rle import masks_to_dataframe
from .utils import load_yaml, parse_overrides, write_json


def _load_config(path: str, overrides: list[str]) -> dict:
    config = load_yaml(path)
    override_config = parse_overrides(overrides)
    from .utils import deep_update

    return deep_update(config, override_config)


def command_audit(args) -> None:
    config = _load_config(args.config, args.set)
    audit_dataset(
        config["data"]["root"],
        args.output,
        int(config["data"]["size"]),
        int(config["data"]["num_folds"]),
    )


def command_train(args) -> None:
    config = _load_config(args.config, args.set)
    run_dir = args.run_dir or str(Path(config["output"]["root"]) / f"{config['model']['name']}_fold{args.fold}")
    result = train_fold(config, args.fold, run_dir)
    print(result)


def command_evaluate(args) -> None:
    config = _load_config(args.config, args.set)
    result = evaluate_checkpoint(
        config,
        args.checkpoint,
        split=args.split,
        threshold=float(config["eval"].get("threshold", 0.5)),
        tta=args.tta,
    )
    output = Path(args.output or Path(args.checkpoint).parent / "evaluation.json")
    write_json(output, result)
    print(result["summary"])


def command_predict(args) -> None:
    config = _load_config(args.config, args.set)
    probabilities, meta = predict_probabilities(
        config, args.checkpoint, split=args.split, tta=not args.no_tta
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_probability_archive(output / "probabilities.npz", probabilities, meta)
    write_json(output / "prediction_metadata.json", meta)
    print(f"Saved {len(probabilities)} probability maps to {output}")


def command_ensemble(args) -> None:
    archives = [load_probability_archive(path) for path in args.probabilities]
    weights = args.weights or [1.0] * len(archives)
    if len(weights) != len(archives):
        raise ValueError("--weights must match --probabilities")
    if sum(weights) <= 0:
        raise ValueError("Ensemble weights must have a positive sum")
    image_ids = sorted(archives[0], key=int)
    if any(set(archive) != set(image_ids) for archive in archives):
        raise ValueError("All probability archives must contain identical image IDs")
    combined = {
        image_id: sum(
            weight * archive[image_id] for weight, archive in zip(weights, archives)
        )
        / sum(weights)
        for image_id in image_ids
    }
    config = _load_config(args.config, args.set)
    output = Path(args.output)
    mask_dir = output / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    samples = {sample.image_id: sample for sample in list_drive_samples(config["data"]["root"], "test")}
    masks = {}
    for image_id in image_ids:
        mask = postprocess_mask(
            combined[image_id],
            samples[image_id].image_path,
            int(config["data"]["size"]),
            float(args.threshold),
            min_area=int(args.min_area),
            fov_mask=not args.no_fov_mask,
            threshold_low=float(config["eval"].get("threshold_low", args.threshold)),
            dti=bool(config["eval"].get("dti", False)),
        )
        masks[image_id] = mask
        import cv2

        cv2.imwrite(str(mask_dir / f"{image_id}.png"), mask * 255)
    np.savez_compressed(
        output / "probabilities.npz",
        **{key: value.astype(np.float16) for key, value in combined.items()},
        _metadata=np.asarray([repr({"sources": args.probabilities, "weights": weights})]),
    )
    write_json(
        output / "ensemble_metadata.json",
        {
            "probabilities": args.probabilities,
            "weights": weights,
            "threshold": args.threshold,
            "min_area": args.min_area,
            "fov_mask": not args.no_fov_mask,
        },
    )
    print(f"Saved ensemble masks to {mask_dir}")


def command_submit(args) -> None:
    import cv2

    mask_dir = Path(args.masks)
    files = sorted(mask_dir.glob("*.png"), key=lambda path: int(path.stem))
    if not files:
        raise ValueError(f"No PNG masks found in {mask_dir}")
    masks = {
        path.stem: (cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        for path in files
    }
    if any(mask.shape != (512, 512) for mask in masks.values()):
        raise ValueError("All submission masks must be exactly 512x512")
    frame = masks_to_dataframe(masks)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote validated RLE submission to {output} ({len(frame)} rows)")


def command_tune(args) -> None:
    config = _load_config(args.config, args.set)
    probabilities = combine_oof_archives(args.probabilities)
    result = optimize_threshold_and_area(config, probabilities)
    output = Path(args.output)
    write_json(output, result)
    print(result["best"])


def command_pack_initial(args) -> None:
    config = _load_config(args.config, args.set)
    init_path = config["model"].get("init_checkpoint")
    if not init_path:
        raise ValueError("Config model.init_checkpoint is required")
    model = build_model(config["model"])
    metadata = load_initial_weights(model, init_path)
    checkpoint = {
        "model": model.state_dict(),
        "config": config,
        "fold": -1,
        "epoch": metadata.get("initial_epoch"),
        "validation_dice": None,
        "initialization": metadata,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Packed initial checkpoint to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DRIVE retinal vessel segmentation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", required=True)
    audit.add_argument("--output", default="runs/data_audit.json")
    audit.add_argument("--set", nargs="*", default=[])
    audit.set_defaults(func=command_audit)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--fold", type=int, required=True)
    train.add_argument("--run-dir")
    train.add_argument("--set", nargs="*", default=[])
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", default="train")
    evaluate.add_argument("--output")
    evaluate.add_argument("--tta", action="store_true")
    evaluate.add_argument("--set", nargs="*", default=[])
    evaluate.set_defaults(func=command_evaluate)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--config", required=True)
    predict.add_argument("--checkpoint", nargs="+", required=True)
    predict.add_argument("--split", default="test")
    predict.add_argument("--output", required=True)
    predict.add_argument("--no-tta", action="store_true")
    predict.add_argument("--set", nargs="*", default=[])
    predict.set_defaults(func=command_predict)

    ensemble = subparsers.add_parser("ensemble")
    ensemble.add_argument("--config", required=True)
    ensemble.add_argument("--probabilities", nargs="+", required=True)
    ensemble.add_argument("--weights", nargs="+", type=float)
    ensemble.add_argument("--threshold", type=float, default=0.5)
    ensemble.add_argument("--min-area", type=int, default=0)
    ensemble.add_argument("--no-fov-mask", action="store_true")
    ensemble.add_argument("--output", required=True)
    ensemble.add_argument("--set", nargs="*", default=[])
    ensemble.set_defaults(func=command_ensemble)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--masks", required=True)
    submit.add_argument("--output", default="submission.csv")
    submit.set_defaults(func=command_submit)

    tune = subparsers.add_parser("tune")
    tune.add_argument("--config", required=True)
    tune.add_argument("--probabilities", nargs="+", required=True)
    tune.add_argument("--output", default="runs/threshold_search.json")
    tune.add_argument("--set", nargs="*", default=[])
    tune.set_defaults(func=command_tune)

    pack = subparsers.add_parser("pack-initial")
    pack.add_argument("--config", required=True)
    pack.add_argument("--output", required=True)
    pack.add_argument("--set", nargs="*", default=[])
    pack.set_defaults(func=command_pack_initial)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
