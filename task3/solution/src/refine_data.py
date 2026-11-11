"""ROI-crop dataset for the coarse-to-fine second stage."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import CACHE
from data import IMAGENET_MEAN, IMAGENET_STD, gaussian_heatmap


def build_refine_aug(size: int):
    import albumentations as A

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03),
                     rotate=(-6, 6), border_mode=cv2.BORDER_REFLECT_101, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=6, p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


class RefineDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ids: list[int], coarse: pd.DataFrame,
                 crop: int = 512, input_size: int = 256, train: bool = True,
                 samples: int = 8, jitter: float = 40.0, sigma: float = 1.5):
        self.df = df.set_index("id")
        self.meta = pd.read_csv(CACHE / "canon_meta.csv").set_index("id")
        self.arr = np.load(CACHE / "canon_2000.npy", mmap_mode="r")
        self.ids = [int(i) for i in ids]
        self.coarse = coarse.set_index("id")
        self.crop = int(crop)
        self.input_size = int(input_size)
        self.out_size = self.input_size // 4
        self.train = train
        self.samples = int(samples) if train else 1
        self.jitter = float(jitter)
        self.sigma = float(sigma)
        self.aug = build_refine_aug(self.input_size) if train else None

    def __len__(self):
        return len(self.ids) * self.samples

    def _cache_xy(self, i: int, x: float, y: float) -> tuple[float, float]:
        m = self.meta.loc[i]
        return (x - m.x0) * m.scale, (y - m.y0) * m.scale

    def __getitem__(self, idx: int):
        i = self.ids[idx // self.samples]
        row = self.df.loc[i]
        gx, gy = self._cache_xy(i, float(row["Fovea_X"]), float(row["Fovea_Y"]))
        if i in self.coarse.index:
            cx, cy = self._cache_xy(i, float(self.coarse.loc[i, "pred_x"]),
                                    float(self.coarse.loc[i, "pred_y"]))
        else:
            # Training ids have no out-of-fold coarse prediction; use GT as the
            # crop centre and rely on jitter to simulate coarse-stage error.
            cx, cy = gx, gy

        if self.train:
            crop_size = int(round(np.random.uniform(448, 576)))
            if np.random.rand() < 0.5:
                # Uniform offsets teach the model to search the whole ROI,
                # not to trust the crop centre as a prior.
                cx = gx + np.random.uniform(-self.jitter, self.jitter)
                cy = gy + np.random.uniform(-self.jitter, self.jitter)
            else:
                cx = gx + np.random.randn() * self.jitter * 0.25
                cy = gy + np.random.randn() * self.jitter * 0.25
        else:
            crop_size = self.crop

        # Keep the fovea inside the crop with a safety margin.
        margin = 40
        cx = float(np.clip(cx, gx - crop_size / 2 + margin, gx + crop_size / 2 - margin))
        cy = float(np.clip(cy, gy - crop_size / 2 + margin, gy + crop_size / 2 - margin))
        x0 = int(round(cx - crop_size / 2))
        y0 = int(round(cy - crop_size / 2))
        x0 = int(np.clip(x0, 0, 2000 - crop_size))
        y0 = int(np.clip(y0, 0, 2000 - crop_size))

        patch = np.asarray(self.arr[i, y0:y0 + crop_size, x0:x0 + crop_size])
        patch = cv2.resize(patch, (self.input_size, self.input_size),
                           interpolation=cv2.INTER_AREA)

        u = (gx - x0) / crop_size
        v = (gy - y0) / crop_size
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            raise ValueError(f"Fovea outside ROI for id {i}: {u:.3f},{v:.3f}")

        if self.aug is not None:
            res = self.aug(image=patch,
                           keypoints=[(u * self.input_size, v * self.input_size)])
            patch = res["image"]
            kx, ky = res["keypoints"][0]
            u = float(np.clip(kx / self.input_size, 0, 1))
            v = float(np.clip(ky / self.input_size, 0, 1))

        x = patch.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).contiguous()
        hm = gaussian_heatmap(self.out_size, u, v, self.sigma)
        m = self.meta.loc[i]
        meta = torch.tensor([x0, y0, crop_size, m.scale, m.x0, m.y0], dtype=torch.float32)
        return {
            "image": x,
            "heatmap": hm,
            "coord": torch.tensor([u, v], dtype=torch.float32),
            "id": i,
            "meta": meta,
        }


def roi_coords_to_pixels(coords: np.ndarray, metas: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    metas = np.asarray(metas, dtype=np.float64)
    cx = metas[:, 0] + coords[:, 0] * metas[:, 2]
    cy = metas[:, 1] + coords[:, 1] * metas[:, 2]
    return np.stack([cx / metas[:, 3] + metas[:, 4],
                     cy / metas[:, 3] + metas[:, 5]], axis=1)
