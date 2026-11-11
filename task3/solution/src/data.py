"""Dataset with center-crop -> square resize and keypoint-aware augmentation."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from common import canonical_crop

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_train_aug(size: int, level: str = "base", with_vessel: bool = False,
                    with_prior: bool = False):
    import albumentations as A

    if level == "strong":
        affine = A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.07, 0.07),
            rotate=(-15, 15),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.9,
        )
    else:
        affine = A.Affine(
            scale=(0.90, 1.10),
            translate_percent=(-0.03, 0.03),
            rotate=(-10, 10),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.8,
        )

    additional_targets = {}
    if with_vessel:
        additional_targets["vessel"] = "image"
    if with_prior:
        additional_targets["prior"] = "image"

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            affine,
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.5),
            A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=10, val_shift_limit=8, p=0.2),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.15),
            A.GaussianBlur(blur_limit=(3, 5), p=0.12),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        additional_targets=additional_targets or None,
    )


def gaussian_heatmap(out_size: int, u: float, v: float, sigma: float = 2.0) -> torch.Tensor:
    ys = torch.arange(out_size, dtype=torch.float32) + 0.5
    xs = torch.arange(out_size, dtype=torch.float32) + 0.5
    cx, cy = u * out_size, v * out_size
    g = torch.exp(-((xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2) / (2 * sigma ** 2))
    return g.unsqueeze(0)


def circle_heatmap(out_size: int, u: float, v: float, radius: float = 6.0) -> torch.Tensor:
    ys = torch.arange(out_size, dtype=torch.float32) + 0.5
    xs = torch.arange(out_size, dtype=torch.float32) + 0.5
    cx, cy = u * out_size, v * out_size
    dist = torch.sqrt((xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2)
    # 1-pixel soft edge keeps gradients stable.
    g = torch.sigmoid((radius - dist) / 0.75)
    return g.unsqueeze(0)


class FoveaDataset(Dataset):
    def __init__(self, df, ids, input_size: int = 512, sigma: float = 2.0,
                 train: bool = True, augment: bool = True, target: str = "gaussian",
                 radius: float = 6.0, aug_level: str = "base", instance_norm: bool = False,
                 aux_od: bool = False, mask_dir: str | None = None,
                 aux_sigma: float = 3.0, vessel_dir: str | None = None,
                 prior_dir: str | None = None, crop_shift: tuple[int, int] = (0, 0)):
        self.df = df.set_index("id")
        self.ids = [int(i) for i in ids]
        self.input_size = int(input_size)
        self.out_size = self.input_size // 4
        self.sigma = float(sigma)
        self.target = target
        self.radius = float(radius)
        self.instance_norm = bool(instance_norm)
        self.aux_od = bool(aux_od)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.aux_sigma = float(aux_sigma)
        self.vessel_dir = Path(vessel_dir) if vessel_dir else None
        self.prior_dir = Path(prior_dir) if prior_dir else None
        self.crop_shift = (int(crop_shift[0]), int(crop_shift[1]))
        self.train = train
        self.aug = (build_train_aug(
                        self.input_size, aug_level, self.vessel_dir is not None,
                        self.prior_dir is not None)
                    if (train and augment) else None)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        i = self.ids[idx]
        row = self.df.loc[i]
        # cv2.imread does not handle non-ASCII Windows paths; decode from bytes.
        buf = np.fromfile(str(row["image_path"]), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x0, y0, size = canonical_crop(row["width"], row["height"])
        x0 = int(np.clip(x0 + self.crop_shift[0], 0, row["width"] - size))
        y0 = int(np.clip(y0 + self.crop_shift[1], 0, row["height"] - size))
        crop = img[y0:y0 + size, x0:x0 + size]
        crop = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        vessel = None
        if self.vessel_dir is not None:
            vessel_path = self.vessel_dir / f"{i:04d}.png"
            if not vessel_path.exists():
                vessel_path = self.vessel_dir / f"{i:05d}.png"
            vbuf = np.fromfile(str(vessel_path), dtype=np.uint8)
            vessel = cv2.imdecode(vbuf, cv2.IMREAD_GRAYSCALE)
            vessel = cv2.resize(vessel, (self.input_size, self.input_size),
                                interpolation=cv2.INTER_AREA)
            vessel = cv2.cvtColor(vessel, cv2.COLOR_GRAY2RGB)
        prior = None
        if self.prior_dir is not None:
            prior_path = self.prior_dir / f"{i:04d}.png"
            if not prior_path.exists():
                prior_path = self.prior_dir / f"{i:05d}.png"
            pbuf = np.fromfile(str(prior_path), dtype=np.uint8)
            prior = cv2.imdecode(pbuf, cv2.IMREAD_GRAYSCALE)
            prior = cv2.resize(prior, (self.input_size, self.input_size),
                               interpolation=cv2.INTER_LINEAR)
            prior = cv2.cvtColor(prior, cv2.COLOR_GRAY2RGB)

        u = (float(row["Fovea_X"]) - x0) / size
        v = (float(row["Fovea_Y"]) - y0) / size
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            raise ValueError(f"Fovea outside canonical crop for id {i}: {u:.4f},{v:.4f}")

        od_u = od_v = 0.5
        if self.aux_od:
            if self.mask_dir is None:
                raise ValueError("aux_od requires mask_dir")
            buf_m = np.fromfile(str(self.mask_dir / f"{i:04d}.png"), dtype=np.uint8)
            mask = cv2.imdecode(buf_m, cv2.IMREAD_UNCHANGED)
            mc = mask[y0:y0 + size, x0:x0 + size]
            vals, cnt = np.unique(mc, return_counts=True)
            bg = vals[np.argmax(cnt)]
            fg = [(v, n) for v, n in zip(vals, cnt) if v != bg]
            # larger foreground region is the optic disc
            disc = max(fg, key=lambda t: t[1])[0]
            oy, ox = np.where(mc == disc)
            od_u = float(ox.mean() / size)
            od_v = float(oy.mean() / size)

        if self.aug is not None:
            kps = [(u * self.input_size, v * self.input_size)]
            if self.aux_od:
                kps.append((od_u * self.input_size, od_v * self.input_size))
            aug_input = {"image": crop}
            if vessel is not None:
                aug_input["vessel"] = vessel
            if prior is not None:
                aug_input["prior"] = prior
            res = self.aug(keypoints=kps, **aug_input)
            crop = res["image"]
            if vessel is not None:
                vessel = res["vessel"][:, :, 0]
            if prior is not None:
                prior = res["prior"][:, :, 0]
            kx, ky = res["keypoints"][0]
            u = float(np.clip(kx / self.input_size, 0.0, 1.0))
            v = float(np.clip(ky / self.input_size, 0.0, 1.0))
            if self.aux_od:
                okx, oky = res["keypoints"][1]
                od_u = float(np.clip(okx / self.input_size, 0.0, 1.0))
                od_v = float(np.clip(oky / self.input_size, 0.0, 1.0))

        x = crop.astype(np.float32) / 255.0
        if self.instance_norm:
            x = (x - x.mean(axis=(0, 1), keepdims=True)) / (
                x.std(axis=(0, 1), keepdims=True) + 1e-6)
        else:
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        if vessel is not None:
            if vessel.ndim == 3:
                vessel = vessel[:, :, 0]
            vmap = vessel.astype(np.float32) / 255.0
            x = np.concatenate([x, vmap[:, :, None]], axis=2)
        if prior is not None:
            if prior.ndim == 3:
                prior = prior[:, :, 0]
            pmap = prior.astype(np.float32) / 255.0
            x = np.concatenate([x, pmap[:, :, None]], axis=2)
        x = torch.from_numpy(x.transpose(2, 0, 1)).contiguous()

        if self.target == "circle":
            hm = circle_heatmap(self.out_size, u, v, self.radius)
        else:
            hm = gaussian_heatmap(self.out_size, u, v, self.sigma)
        od_hm = (gaussian_heatmap(self.out_size, od_u, od_v, self.aux_sigma)
                 if self.aux_od else torch.zeros_like(hm))
        meta = torch.tensor([x0, y0, size], dtype=torch.float32)
        return {
            "image": x,
            "heatmap": hm,
            "coord": torch.tensor([u, v], dtype=torch.float32),
            "od_heatmap": od_hm,
            "od_coord": torch.tensor([od_u, od_v], dtype=torch.float32),
            "id": i,
            "meta": meta,
        }
