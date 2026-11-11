from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def encode_mask(mask: np.ndarray) -> str:
    """Encode a binary mask using one-based, inclusive (start, length) RLE.

    The empty mask is encoded as an empty string, matching the supplied sample
    submission and Kaggle's run-length format.
    """
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if not np.isin(flat, [0, 1]).all():
        raise ValueError("RLE input must be binary after thresholding")
    if not flat.any():
        return ""

    padded = np.pad(flat, (1, 1), mode="constant")
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    if starts.size != ends.size:
        raise RuntimeError("Could not determine contiguous RLE runs")
    pairs = np.column_stack((starts + 1, ends - starts))
    return " ".join(f"{start} {length}" for start, length in pairs)


def decode_mask(rle: str, shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    """Decode a Kaggle-style RLE string and validate its pixel ranges."""
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    values = str(rle or "").split()
    if len(values) % 2:
        raise ValueError("RLE must contain an even number of integers")

    previous_end = 0
    for offset in range(0, len(values), 2):
        start = int(values[offset])
        length = int(values[offset + 1])
        if start < 1 or length < 1:
            raise ValueError("RLE starts and lengths must be positive")
        end = start + length - 1
        if start <= previous_end:
            raise ValueError("RLE runs must be sorted and non-overlapping")
        if end > flat.size:
            raise ValueError("RLE run extends beyond the mask")
        flat[start - 1 : end] = 1
        previous_end = end
    return flat.reshape(shape)


def masks_to_dataframe(masks: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for image_id in sorted(masks, key=lambda value: int(Path(value).stem)):
        mask = np.asarray(masks[image_id], dtype=np.uint8)
        rle = encode_mask(mask)
        decoded = decode_mask(rle, shape=mask.shape)
        if not np.array_equal(decoded, mask):
            raise ValueError(f"RLE round-trip failed for {image_id}")
        rows.append({"Id": Path(image_id).stem, "Predicted": rle})
    return pd.DataFrame(rows, columns=["Id", "Predicted"])
