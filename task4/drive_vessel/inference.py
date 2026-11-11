from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .data import (
    FOVCropInferenceDataset,
    NativePadInferenceDataset,
    RetinaDataset,
    estimate_fov,
    list_drive_samples,
    read_binary_mask,
)
from .metrics import binary_metrics, summarize_metrics
from .models import build_model
from .utils import get_device


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


def _tta_probabilities(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Average calibrated probabilities, as in the SGL reference test code.

    Averaging logits is not equivalent to averaging probabilities and changes
    the selected foreground area, especially for the highly imbalanced vessel
    task.  The original SGL implementation averages the eight probability
    maps from its flip/transpose ensemble.
    """
    variants = [
        (images, lambda value: value),
        (torch.flip(images, dims=[3]), lambda value: torch.flip(value, dims=[3])),
        (torch.flip(images, dims=[2]), lambda value: torch.flip(value, dims=[2])),
        (torch.flip(images, dims=[2, 3]), lambda value: torch.flip(value, dims=[2, 3])),
        (images.transpose(-1, -2), lambda value: value.transpose(-1, -2)),
        (
            torch.flip(images, dims=[3]).transpose(-1, -2),
            lambda value: torch.flip(value.transpose(-1, -2), dims=[3]),
        ),
        (
            torch.flip(images, dims=[2]).transpose(-1, -2),
            lambda value: torch.flip(value.transpose(-1, -2), dims=[2]),
        ),
        (
            torch.flip(images, dims=[2, 3]).transpose(-1, -2),
            lambda value: torch.flip(value.transpose(-1, -2), dims=[2, 3]),
        ),
    ]
    probabilities = []
    for tensor, undo in variants:
        logits = undo(model(tensor)).float()
        probabilities.append(torch.sigmoid(logits))
    return torch.stack(probabilities, dim=0).mean(dim=0)


@torch.no_grad()
def predict_probabilities(
    config: dict,
    checkpoints: list[str | Path],
    split: str,
    tta: bool = True,
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> tuple[dict[str, np.ndarray], dict]:
    device = get_device(config)
    samples = list_drive_samples(config["data"]["root"], split)
    inference_mode = str(config["data"].get("inference_mode", "resize"))
    if inference_mode == "native_pad":
        dataset = NativePadInferenceDataset(
            samples,
            pad_size=int(config["data"].get("native_pad_size", 592)),
            preprocessing=str(config["data"]["preprocessing"]),
            input_channels=int(config["data"].get("input_channels", 3)),
            normalization=str(config["data"].get("normalization", "imagenet")),
        )
    elif inference_mode == "fov_crop":
        dataset = FOVCropInferenceDataset(
            samples,
            crop_size=int(config["data"].get("crop_size", 512)),
            preprocessing=str(config["data"]["preprocessing"]),
            input_channels=int(config["data"].get("input_channels", 3)),
            normalization=str(config["data"].get("normalization", "imagenet")),
        )
    elif inference_mode == "resize":
        dataset = RetinaDataset(
            samples,
            size=int(config["data"]["size"]),
            training=False,
            preprocessing=str(config["data"]["preprocessing"]),
            augment=False,
            input_channels=int(config["data"].get("input_channels", 3)),
            normalization=str(config["data"].get("normalization", "imagenet")),
        )
    else:
        raise ValueError(f"Unknown inference mode: {inference_mode}")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(config["train"].get("predict_batch_size", 1)), shuffle=False, num_workers=0
    )

    per_checkpoint: list[dict[str, np.ndarray]] = []
    metadata = []
    for checkpoint_path in checkpoints:
        model, checkpoint = load_checkpoint(checkpoint_path, device)
        current: dict[str, np.ndarray] = {}
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None and device.type == "cuda",
            ):
                if tta:
                    probability_tensor = _tta_probabilities(model, images)
                else:
                    probability_tensor = torch.sigmoid(model(images).float())
            probabilities = probability_tensor.cpu().numpy()[:, 0]
            for index, image_id in enumerate(batch["image_id"]):
                probability = probabilities[index]
                if inference_mode == "native_pad":
                    height = int(batch["original_height"][index])
                    width = int(batch["original_width"][index])
                    probability = cv2.resize(
                        probability[:height, :width],
                        (int(config["data"]["size"]), int(config["data"]["size"])),
                        interpolation=cv2.INTER_LINEAR,
                    )
                elif inference_mode == "fov_crop":
                    height = int(batch["original_height"][index])
                    width = int(batch["original_width"][index])
                    crop_height = int(batch["crop_height"][index])
                    crop_width = int(batch["crop_width"][index])
                    crop_x = int(batch["crop_x"][index])
                    crop_y = int(batch["crop_y"][index])
                    full_image = np.zeros((height, width), dtype=np.float32)
                    full_image[
                        crop_y : crop_y + crop_height,
                        crop_x : crop_x + crop_width,
                    ] = cv2.resize(
                        probability,
                        (crop_width, crop_height),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    probability = cv2.resize(
                        full_image,
                        (int(config["data"]["size"]), int(config["data"]["size"])),
                        interpolation=cv2.INTER_LINEAR,
                    )
                current[image_id] = probability
        per_checkpoint.append(current)
        metadata.append(
            {
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "fold": checkpoint.get("fold"),
                "epoch": checkpoint.get("epoch"),
                "validation_dice": checkpoint.get("validation_dice"),
                "model": checkpoint["config"]["model"]["name"],
            }
        )
    if not per_checkpoint:
        raise ValueError("At least one checkpoint is required")
    image_ids = sorted(per_checkpoint[0], key=int)
    averaged = {
        image_id: np.mean([result[image_id] for result in per_checkpoint], axis=0)
        for image_id in image_ids
    }
    return averaged, {"checkpoints": metadata, "tta": tta, "split": split}


def postprocess_mask(
    probability: np.ndarray,
    image_path: Path,
    size: int,
    threshold: float,
    min_area: int = 0,
    fov_mask: bool = True,
    threshold_low: float | None = None,
    dti: bool = False,
) -> np.ndarray:
    binary = (probability >= threshold).astype(np.uint8)
    if dti:
        low = float(threshold if threshold_low is None else threshold_low)
        if low > threshold:
            raise ValueError("DTI low threshold cannot exceed the high threshold")
        expandable = (probability >= low).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        while True:
            expanded = cv2.dilate(binary, kernel, iterations=1) * expandable
            if np.array_equal(expanded, binary):
                break
            binary = expanded
    if fov_mask:
        from .data import read_rgb

        binary = binary * estimate_fov(read_rgb(image_path, size))
    if min_area > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        filtered = np.zeros_like(binary)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == label] = 1
        binary = filtered
    return binary


def evaluate_checkpoint(
    config: dict,
    checkpoint_path: str | Path,
    split: str,
    threshold: float,
    tta: bool,
) -> dict:
    samples = list_drive_samples(config["data"]["root"], split)
    probabilities, prediction_metadata = predict_probabilities(
        config, [checkpoint_path], split, tta=tta
    )
    rows = []
    for sample in samples:
        target = read_binary_mask(sample.mask_path, int(config["data"]["size"])) > 0.5
        prediction = postprocess_mask(
            probabilities[sample.image_id],
            sample.image_path,
            int(config["data"]["size"]),
            threshold,
            min_area=int(config["eval"].get("min_area", 0)),
            fov_mask=bool(config["eval"].get("fov_mask", True)),
            threshold_low=float(config["eval"].get("threshold_low", threshold)),
            dti=bool(config["eval"].get("dti", False)),
        )
        rows.append({"id": sample.image_id, **binary_metrics(target, prediction)})
    return {
        "summary": summarize_metrics(rows),
        "rows": rows,
        "checkpoint": str(checkpoint_path),
        "tta": tta,
        "prediction_metadata": prediction_metadata,
    }


def save_probability_archive(path: Path, probabilities: dict[str, np.ndarray], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{key: value.astype(np.float16) for key, value in probabilities.items()},
        _metadata=np.asarray([repr(meta)]),
    )


def load_probability_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: archive[key].astype(np.float32)
            for key in archive.files
            if key != "_metadata"
        }
