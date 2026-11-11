"""Shared paths, data indexing, fold creation and metrics."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import KFold

PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("FOVEA_DATA_DIR", PROJECT.parent / "2026task3" / "detection")).expanduser().resolve()
TRAIN_DIR = DATA / "train"
TEST_DIR = DATA / "test"
GT_CSV = DATA / "fovea_localization_train_GT.csv"
SAMPLE_SUB = DATA / "sample_submission.csv"
CACHE = PROJECT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

# Competition test ids are 81..100. Public GAMMA labels for these ids exist;
# they must never be read or used. This guard is repeated in every loader.
MAX_TRAIN_ID = 80


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def load_train_df() -> pd.DataFrame:
    df = pd.read_csv(GT_CSV)
    df["id"] = df["data"].astype(int)
    df = df[df["id"] <= MAX_TRAIN_ID].copy()
    df = df.sort_values("id").reset_index(drop=True)
    if df["id"].max() > MAX_TRAIN_ID or len(df) != MAX_TRAIN_ID:
        raise RuntimeError("Training label guard failed: only ids 1..80 are allowed.")

    size_cache = CACHE / "train_image_sizes.json"
    sizes = json.loads(size_cache.read_text()) if size_cache.exists() else {}
    for i in df["id"]:
        key = f"{int(i):04d}"
        if key not in sizes:
            sizes[key] = list(_image_size(TRAIN_DIR / f"{key}.jpg"))
    size_cache.write_text(json.dumps(sizes, indent=1))
    df["width"] = [sizes[f"{int(i):04d}"][0] for i in df["id"]]
    df["height"] = [sizes[f"{int(i):04d}"][1] for i in df["id"]]
    df["image_path"] = [str(TRAIN_DIR / f"{int(i):04d}.jpg") for i in df["id"]]
    return df


def load_test_df() -> pd.DataFrame:
    ids = list(range(81, 101))
    rows = []
    for i in ids:
        w, h = _image_size(TEST_DIR / f"{i:04d}.jpg")
        rows.append({"id": i, "width": w, "height": h,
                     "image_path": str(TEST_DIR / f"{i:04d}.jpg")})
    return pd.DataFrame(rows)


def canonical_crop(width: int, height: int) -> tuple[int, int, int]:
    """Center square crop that removes the left/right black border."""
    size = min(int(width), int(height))
    x0 = (int(width) - size) // 2
    y0 = (int(height) - size) // 2
    return x0, y0, size


def _md5(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> list[dict]:
    """Group exact duplicate files so a duplicate never spans train/val."""
    hashes = {int(r.id): _md5(r.image_path) for r in df.itertuples()}
    groups: dict[int, int] = {}
    by_hash: dict[str, int] = {}
    for i in df["id"]:
        h = hashes[int(i)]
        if h in by_hash:
            groups[int(i)] = by_hash[h]
        else:
            by_hash[h] = int(i)
            groups[int(i)] = int(i)

    unique_groups = sorted(set(groups.values()))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (tr_idx, va_idx) in enumerate(kf.split(unique_groups)):
        tr_groups = {unique_groups[i] for i in tr_idx}
        va_groups = {unique_groups[i] for i in va_idx}
        folds.append({
            "fold": k,
            "train_ids": [int(i) for i in df["id"] if groups[int(i)] in tr_groups],
            "val_ids": [int(i) for i in df["id"] if groups[int(i)] in va_groups],
        })
    return folds


def load_or_make_folds(df: pd.DataFrame, path: Path | None = None,
                       n_splits: int = 5, seed: int = 42) -> list[dict]:
    path = path or (PROJECT / "folds.json")
    if path.exists():
        folds = json.loads(path.read_text())
    else:
        folds = make_folds(df, n_splits=n_splits, seed=seed)
        path.write_text(json.dumps(folds, indent=1))
    covered = sorted(i for f in folds for i in f["val_ids"])
    if covered != sorted(df["id"].tolist()):
        raise RuntimeError("Fold file does not cover all training ids exactly once.")
    return folds


def pixels_rmse(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    """Competition metric: RMSE over all X and Y values."""
    d = np.asarray(pred_xy, dtype=np.float64) - np.asarray(gt_xy, dtype=np.float64)
    return float(np.sqrt(np.mean(d ** 2)))


def coord_rmse(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    d = np.asarray(pred_xy, dtype=np.float64) - np.asarray(gt_xy, dtype=np.float64)
    return np.sqrt(np.mean(d ** 2, axis=1))
