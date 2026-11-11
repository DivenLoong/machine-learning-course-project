"""Inference helpers with flip TTA."""

from __future__ import annotations

import numpy as np
import torch

from model import heatmap_to_coord


@torch.no_grad()
def predict_tensor(model, images: torch.Tensor, tta: bool = True,
                   window: int | None = None) -> torch.Tensor:
    model.eval()
    kw = {} if window is None else {"window": window}
    if not tta:
        return heatmap_to_coord(model(images), **kw)

    acc = None
    for flip in ("none", "h", "v", "hv"):
        x = images
        if "h" in flip:
            x = torch.flip(x, dims=[3])
        if "v" in flip:
            x = torch.flip(x, dims=[2])
        c = heatmap_to_coord(model(x), **kw)
        if "h" in flip:
            c[:, 0] = 1.0 - c[:, 0]
        if "v" in flip:
            c[:, 1] = 1.0 - c[:, 1]
        acc = c if acc is None else acc + c
    return acc / 4.0


@torch.no_grad()
def predict_loader(model, loader, device, tta: bool = True, window: int | None = None):
    ids, coords, metas = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        c = predict_tensor(model, images, tta=tta, window=window)
        ids.extend([int(v) for v in batch["id"]])
        coords.append(c.float().cpu().numpy())
        metas.append(batch["meta"].numpy())
    return np.asarray(ids), np.concatenate(coords), np.concatenate(metas)


@torch.no_grad()
def predict_dual_loader(model, loader, device, tta: bool = True, window: int = 17,
                        fuse_threshold: float = 0.015):
    """Return heatmap coords, regression coords and MedICAL-style fused coords."""
    ids, hm_list, reg_list, fused_list, metas = [], [], [], [], []
    kw = {"window": window}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        hm_acc = torch.zeros(images.size(0), 2, device=device)
        reg_acc = torch.zeros_like(hm_acc)
        flips = ("none", "h", "v", "hv") if tta else ("none",)
        for flip in flips:
            x = images
            if "h" in flip:
                x = torch.flip(x, dims=[3])
            if "v" in flip:
                x = torch.flip(x, dims=[2])
            hm, reg = model(x, return_reg=True)
            c = heatmap_to_coord(hm, **kw)
            r = reg
            if "h" in flip:
                c[:, 0] = 1 - c[:, 0]
                r = r.clone(); r[:, 0] = 1 - r[:, 0]
            if "v" in flip:
                c[:, 1] = 1 - c[:, 1]
                r = r.clone(); r[:, 1] = 1 - r[:, 1]
            hm_acc += c
            reg_acc += r
        n = len(flips)
        hm_c = hm_acc / n
        reg_c = reg_acc / n
        dist = torch.linalg.vector_norm(hm_c - reg_c, dim=1).unsqueeze(1)
        fused = torch.where(dist > fuse_threshold, reg_c, 0.5 * (hm_c + reg_c))
        ids.extend([int(v) for v in batch["id"]])
        hm_list.append(hm_c.cpu().numpy())
        reg_list.append(reg_c.cpu().numpy())
        fused_list.append(fused.cpu().numpy())
        metas.append(batch["meta"].numpy())
    return (np.asarray(ids), np.concatenate(hm_list), np.concatenate(reg_list),
            np.concatenate(fused_list), np.concatenate(metas))


def coords_to_pixels(coords: np.ndarray, metas: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    metas = np.asarray(metas, dtype=np.float64)
    return coords * metas[:, 2:3] + metas[:, :2]
