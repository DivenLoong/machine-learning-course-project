from __future__ import annotations

import gzip
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from skimage.filters import frangi
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: Path
    mask_path: Path | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def list_drive_samples(root: str | Path, split: str = "train") -> list[Sample]:
    root = Path(root)
    image_dir = root / split / "image"
    label_dir = root / split / "label"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    samples = []
    for image_path in sorted(image_dir.glob("*.jpg"), key=lambda p: int(p.stem)):
        mask_path = label_dir / image_path.name
        if split == "train" and not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask for {image_path.name}")
        samples.append(
            Sample(
                image_id=image_path.stem,
                image_path=image_path,
                mask_path=mask_path if mask_path.is_file() else None,
            )
        )
    if not samples:
        raise ValueError(f"No JPEG images found under {image_dir}")
    return samples


def read_rgb(path: Path, size: int) -> np.ndarray:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle, Image.open(io.BytesIO(handle.read())) as image:
            array = np.asarray(image.convert("RGB"))
    else:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
    return cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA)


def read_rgb_native(path: Path) -> np.ndarray:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle, Image.open(io.BytesIO(handle.read())) as image:
            return np.asarray(image.convert("RGB"))
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def read_binary_mask(path: Path, size: int) -> np.ndarray:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle, Image.open(io.BytesIO(handle.read())) as image:
            array = np.asarray(image.convert("L"))
    else:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"))
    resized = cv2.resize(array, (size, size), interpolation=cv2.INTER_NEAREST)
    binary = (resized > 127).astype(np.float32)
    if binary.min() == binary.max():
        raise ValueError(f"Mask is not binary at threshold 127: {path}")
    return binary


def read_training_mask(path: Path, size: int) -> np.ndarray:
    """Load binary image masks or continuous `.npy` distillation targets."""
    if path.suffix == ".npy":
        mask = np.load(path).astype(np.float32)
        if mask.shape != (size, size):
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask, 0.0, 1.0)
    try:
        return read_binary_mask(path, size)
    except ValueError as error:
        if "Mask is not binary" not in str(error):
            raise
        return np.zeros((size, size), dtype=np.float32)


def list_external_samples(
    chase_root: str | Path | None = None,
    stare_root: str | Path | None = None,
    fives_root: str | Path | None = None,
) -> list[Sample]:
    """Load the first-observer annotations from CHASE_DB1 and STARE."""
    samples: list[Sample] = []
    if chase_root:
        root = Path(chase_root)
        masks = {
            path.name.removesuffix("_1stHO.png"): path
            for path in root.glob("Image_*_1stHO.png")
        }
        for image_path in sorted(root.glob("Image_*.jpg")):
            if image_path.stem not in masks:
                raise FileNotFoundError(f"Missing CHASE mask for {image_path.name}")
            samples.append(
                Sample(
                    image_id=f"chase_{image_path.stem}",
                    image_path=image_path,
                    mask_path=masks[image_path.stem],
                )
            )
    if stare_root:
        root = Path(stare_root)
        for image_path in sorted(root.glob("im*.ppm.gz")):
            if ".ah." in image_path.name:
                continue
            mask_path = image_path.with_name(
                image_path.name.replace(".ppm.gz", ".ah.ppm.gz")
            )
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing STARE mask for {image_path.name}")
            samples.append(
                Sample(
                    image_id=f"stare_{image_path.name.split('.')[0]}",
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )
    if fives_root:
        root = Path(fives_root)
        image_dir = root / "train" / "Original"
        mask_dir = root / "train" / "Ground truth"
        for image_path in sorted(image_dir.glob("*.png")):
            mask_path = mask_dir / image_path.name
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            samples.append(
                Sample(
                    image_id=f"fives_{image_path.stem}",
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )
    return samples


def list_extra_samples(root: str | Path, prefix: str = "pseudo") -> list[Sample]:
    """Load an unlabelled-target set accompanied by model-generated masks."""
    root = Path(root)
    image_dir = root / "image"
    label_dir = root / "label"
    samples = []
    for image_path in sorted(image_dir.glob("*.jpg"), key=lambda path: int(path.stem)):
        candidates = [
            label_dir / f"{image_path.stem}{suffix}"
            for suffix in (".png", ".jpg", ".npy")
        ]
        mask_path = next((path for path in candidates if path.is_file()), None)
        if mask_path is None:
            raise FileNotFoundError(f"Missing pseudo mask for {image_path.name}")
        samples.append(
            Sample(f"{prefix}_{image_path.stem}", image_path, mask_path)
        )
    return samples


def estimate_fov(image: np.ndarray) -> np.ndarray:
    """Estimate the circular retinal field of view from a resized RGB image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    candidate = (gray > 7).astype(np.uint8)
    kernel = np.ones((9, 9), dtype=np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(gray, dtype=np.uint8)
    filled = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return filled


def preprocess_image(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return image
    if mode == "clahe_frangi":
        green = image[..., 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(green)
        vesselness = frangi(
            clahe.astype(np.float32) / 255.0,
            sigmas=range(1, 5),
            black_ridges=False,
        )
        vesselness = np.nan_to_num(vesselness, nan=0.0)
        vesselness = vesselness / max(float(vesselness.max()), 1e-8)
        return np.stack(
            [clahe.astype(np.float32) / 255.0, vesselness.astype(np.float32)],
            axis=-1,
        )
    if mode == "green_clahe":
        green = image[..., 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(green)
        output = image.copy()
        output[..., 1] = clahe
        return output
    raise ValueError(f"Unknown preprocessing mode: {mode}")


def image_to_tensor(
    image: np.ndarray,
    input_channels: int,
    normalization: str,
) -> torch.Tensor:
    if input_channels == 1:
        image = image.astype(np.float32) / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        minimum, maximum = float(image.min()), float(image.max())
        image = (image - minimum) / max(maximum - minimum, 1e-6)
        return torch.from_numpy(np.ascontiguousarray(image[None]))
    if input_channels == 2:
        if image.ndim != 3 or image.shape[2] != 2:
            raise ValueError("Two-channel input requires an HxWx2 image")
        return torch.from_numpy(
            np.ascontiguousarray(image.astype(np.float32).transpose(2, 0, 1))
        )
    if normalization == "raw":
        image = image.astype(np.float32)
    elif normalization == "unit":
        image = image.astype(np.float32) / 255.0
    elif normalization == "imagenet":
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def normalize_grayscale_image(image: np.ndarray) -> np.ndarray:
    """Match FR-UNet preprocessing before any random crop."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    minimum, maximum = float(gray.min()), float(gray.max())
    return (gray - minimum) / max(maximum - minimum, 1e-6)


def build_train_transform(
    size: int,
    patch_size: int | None = None,
    profile: str = "strong",
) -> A.Compose:
    if profile == "mild":
        transforms: list[Any] = []
        if patch_size is not None and patch_size < size:
            transforms.append(
                A.RandomCrop(
                    height=patch_size,
                    width=patch_size,
                    pad_if_needed=True,
                    pad_position="center",
                    fill=0,
                    fill_mask=0,
                    p=1.0,
                )
            )
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(
                    scale=(0.95, 1.05),
                    translate_percent=(-0.02, 0.02),
                    rotate=(-10, 10),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.2
                ),
            ]
        )
        return A.Compose(transforms)
    if profile != "strong":
        raise ValueError(f"Unknown augmentation profile: {profile}")

    transforms: list[Any] = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.12),
            translate_percent=(-0.04, 0.04),
            rotate=(-30, 30),
            shear=(-8, 8),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.7,
        ),
        A.OneOf(
            [
                A.ElasticTransform(alpha=1.0, sigma=20, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.15, p=1.0),
            ],
            p=0.25,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.2, p=0.6
        ),
        A.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.12,
            hue=0.03,
            p=0.35,
        ),
        A.OneOf(
            [
                A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ],
            p=0.15,
        ),
    ]
    if patch_size is not None and patch_size < size:
        transforms.append(
            A.RandomCrop(
                height=patch_size,
                width=patch_size,
                pad_if_needed=True,
                pad_position="center",
                fill=0,
                fill_mask=0,
                p=1.0,
            )
        )
    return A.Compose(
        transforms
    )


def build_eval_transform(size: int) -> A.Compose:
    return A.Compose([])


class RetinaDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: list[Sample],
        size: int = 512,
        training: bool = False,
        preprocessing: str = "raw",
        augment: bool = True,
        patches_per_image: int = 1,
        train_patch_size: int | None = None,
        cache: bool = False,
        input_channels: int = 3,
        normalization: str = "imagenet",
        augment_profile: str = "strong",
        fov_center_crop: bool = False,
    ) -> None:
        if not samples:
            raise ValueError("Dataset requires at least one sample")
        if patches_per_image < 1:
            raise ValueError("patches_per_image must be positive")
        self.samples = samples
        self.size = size
        self.training = training
        self.preprocessing = preprocessing
        self.patches_per_image = patches_per_image if training else 1
        self.patch_size = train_patch_size if training else None
        self.cache = cache
        if input_channels not in {1, 2, 3}:
            raise ValueError("input_channels must be 1, 2, or 3")
        self.input_channels = input_channels
        if normalization not in {"imagenet", "unit", "raw"}:
            raise ValueError(f"Unknown normalization: {normalization}")
        self.normalization = normalization
        self.augment_profile = augment_profile
        self.fov_center_crop = fov_center_crop
        self._cached: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        transform_patch_size = (
            None
            if self.patch_size is not None and self.fov_center_crop
            else self.patch_size
        )
        self.transform = (
            build_train_transform(size, transform_patch_size, augment_profile)
            if training and augment
            else build_eval_transform(size)
        )

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index % len(self.samples)]
        if self.cache and sample.image_id in self._cached:
            image, mask = self._cached[sample.image_id]
            image = image.copy()
        else:
            image = preprocess_image(read_rgb(sample.image_path, self.size), self.preprocessing)
            if self.input_channels == 1:
                # Preserve the official FR-UNet normalization order: normalize
                # the complete image, then crop. Normalizing after a crop changes
                # the local contrast and is inconsistent with the checkpoint.
                image = normalize_grayscale_image(image)
            mask = (
                read_training_mask(sample.mask_path, self.size)
                if sample.mask_path is not None
                else np.zeros((self.size, self.size), dtype=np.float32)
            )
            if self.cache:
                self._cached[sample.image_id] = (image.copy(), mask.copy())
        if (
            sample.mask_path is not None
            and self.patch_size is not None
            and self.fov_center_crop
        ):
            height, width = image.shape[:2]
            patch = int(self.patch_size)
            foreground = image if image.ndim == 2 else image[..., 0]
            top = left = 0
            for _ in range(20):
                top = int(np.random.randint(0, height - patch + 1))
                left = int(np.random.randint(0, width - patch + 1))
                if foreground[top + patch // 2, left + patch // 2] > 1e-4:
                    break
            image = image[top : top + patch, left : left + patch]
            mask = mask[top : top + patch, left : left + patch]

        if sample.mask_path is not None:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        else:
            image = self.transform(image=image)["image"]
            mask = np.zeros(image.shape[:2], dtype=np.float32)

        if self.input_channels == 1:
            image_tensor = torch.from_numpy(
                np.ascontiguousarray(image[None].astype(np.float32))
            )
        else:
            image_tensor = image_to_tensor(
                image, self.input_channels, self.normalization
            )
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask[None].astype(np.float32)))
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_id": sample.image_id,
        }


class NativePadInferenceDataset(Dataset[dict[str, Any]]):
    """Preserve source aspect ratio, pad bottom/right, and crop after inference."""

    def __init__(
        self,
        samples: list[Sample],
        pad_size: int,
        preprocessing: str,
        input_channels: int,
        normalization: str,
    ) -> None:
        self.samples = samples
        self.pad_size = pad_size
        self.preprocessing = preprocessing
        self.input_channels = input_channels
        self.normalization = normalization
        if normalization == "fr_official":
            if input_channels != 1:
                raise ValueError("fr_official normalization requires input_channels=1")
            values = []
            for sample in samples:
                image = preprocess_image(
                    read_rgb_native(sample.image_path), preprocessing
                )
                gray = (
                    cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
                    / 255.0
                )
                values.append(gray.reshape(-1))
            combined = np.concatenate(values)
            self.global_mean = float(combined.mean())
            self.global_std = float(combined.std())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = preprocess_image(read_rgb_native(sample.image_path), self.preprocessing)
        original_height, original_width = image.shape[:2]
        if original_height > self.pad_size or original_width > self.pad_size:
            raise ValueError(
                f"{sample.image_path.name} exceeds native pad {self.pad_size}: "
                f"{original_height}x{original_width}"
            )
        if self.normalization == "fr_official":
            gray = (
                cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
                / 255.0
            )
            gray = (gray - self.global_mean) / max(self.global_std, 1e-6)
            minimum, maximum = float(gray.min()), float(gray.max())
            gray = (gray - minimum) / max(maximum - minimum, 1e-6)
            padded = np.pad(
                gray,
                (
                    (0, self.pad_size - original_height),
                    (0, self.pad_size - original_width),
                ),
                mode="constant",
            )
            return {
                "image": torch.from_numpy(
                    np.ascontiguousarray(padded[None].astype(np.float32))
                ),
                "image_id": sample.image_id,
                "original_height": original_height,
                "original_width": original_width,
            }
        image = np.pad(
            image,
            (
                (0, self.pad_size - original_height),
                (0, self.pad_size - original_width),
                (0, 0),
            ),
            mode="constant",
        )
        return {
            "image": image_to_tensor(
                image, self.input_channels, self.normalization
            ),
            "image_id": sample.image_id,
            "original_height": original_height,
            "original_width": original_width,
        }


class FOVCropInferenceDataset(Dataset[dict[str, Any]]):
    """Crop to the retinal field of view and resize the crop to a square."""

    def __init__(
        self,
        samples: list[Sample],
        crop_size: int,
        preprocessing: str,
        input_channels: int,
        normalization: str,
    ) -> None:
        self.samples = samples
        self.crop_size = crop_size
        self.preprocessing = preprocessing
        self.input_channels = input_channels
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = preprocess_image(read_rgb_native(sample.image_path), self.preprocessing)
        original_height, original_width = image.shape[:2]
        fov = estimate_fov(cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA))
        points = cv2.findNonZero((fov > 0).astype(np.uint8))
        if points is None:
            x, y, width, height = 0, 0, original_width, original_height
        else:
            x, y, width, height = cv2.boundingRect(points)
            x = round(x * original_width / 512)
            y = round(y * original_height / 512)
            width = round(width * original_width / 512)
            height = round(height * original_height / 512)
            width = max(1, min(width, original_width - x))
            height = max(1, min(height, original_height - y))
        crop = image[y : y + height, x : x + width]
        crop = cv2.resize(
            crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC
        )
        return {
            "image": image_to_tensor(
                crop, self.input_channels, self.normalization
            ),
            "image_id": sample.image_id,
            "original_height": original_height,
            "original_width": original_width,
            "crop_x": x,
            "crop_y": y,
            "crop_height": height,
            "crop_width": width,
        }
