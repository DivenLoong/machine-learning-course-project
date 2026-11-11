"""Shared library for the machine-learning assignment "Task 1: plant classification".

Contents
--------
* paths / config dataclasses
* dataset scanning
* plant-vs-soil cropping (HSV green mask, borrowed from published solutions)
* datasets with train-time augmentation and multi-view TTA
* backbone construction with ImageNet weights
* training / evaluation / inference engine
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "dataset-for-task2")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
SAMPLE_SUBMISSION = os.path.join(ROOT, "submission-for-task2.csv")
OUT_DIR = os.path.join(ROOT, "outputs")

CLASSES = [
    "Black-grass",
    "Common wheat",
    "Loose Silky-bent",
    "Scentless Mayweed",
    "Sugar beet",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Colour used to fill the transparent border / background so it matches the soil rather
# than the black that a naive RGBA->RGB conversion would produce.
PAD_COLOUR = (124, 116, 104)
CACHE_VERSION = "v3"   # v2 caches were built with the crop/resize order bug


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_rgb(path: str) -> Image.Image:
    """Open an image as RGB, compositing any alpha channel onto the soil colour.

    Nine of the supplied PNGs are RGBA with transparent borders; converting those
    straight to RGB turns the border pure black and wrecks them.
    """
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, PAD_COLOUR)
            bg.paste(rgba, mask=rgba.split()[-1])
            return bg
        return im.convert("RGB")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------


@dataclass
class CropConfig:
    """Settings for the green-mask crop that normalises plant scale."""

    enabled: bool = True
    hue_center: float = 60.0
    hue_tol: float = 35.0
    sat_min: float = 0.20
    val_min: float = 0.12
    min_area_frac: float = 0.01
    margin: float = 0.10
    pad_to_square: bool = True
    mask_background: bool = False
    max_side: int = 1024
    cache_name: str = "crop_boxes.json"


@dataclass
class TrainConfig:
    arch: str = "efficientnet_b0"
    size: int = 256
    folds: int = 5
    seed: int = 42
    batch_size: int = 32
    eval_batch_size: int = 64
    head_epochs: int = 4
    finetune_epochs: int = 36
    head_lr: float = 1e-3
    lr: float = 2e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    raug: bool = True
    raug_ops: int = 2
    raug_mag: int = 9
    ema_decay: float = 0.0
    num_workers: int = 2
    amp: bool = True
    tag: str = "baseline"


# --------------------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------------------


def scan_train() -> List[Tuple[str, int]]:
    """Return a deterministic list of (image path, class index)."""
    samples: List[Tuple[str, int]] = []
    for cls in CLASSES:
        d = os.path.join(TRAIN_DIR, cls)
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(".png"):
                samples.append((os.path.join(d, name), CLASS_TO_IDX[cls]))
    return samples
    return samples


def scan_test() -> List[Tuple[str, str]]:
    """Return a deterministic, id-sorted list of (image path, submission id)."""
    names = sorted(n for n in os.listdir(TEST_DIR) if n.lower().endswith(".png"))
    return [(os.path.join(TEST_DIR, n), n) for n in names]


# --------------------------------------------------------------------------------------
# plant / soil segmentation
# --------------------------------------------------------------------------------------


def rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    """Vectorised RGB->HSV. Input float array in [0, 1]; H in [0, 360), S/V in [0, 1]."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    safe = diff > 1e-8
    # where max is r
    m = safe & (mx == r)
    h[m] = (60 * ((g[m] - b[m]) / diff[m])) % 360
    m = safe & (mx == g)
    h[m] = 60 * ((b[m] - r[m]) / diff[m]) + 120
    m = safe & (mx == b)
    h[m] = 60 * ((r[m] - g[m]) / diff[m]) + 240
    s = np.where(mx > 1e-8, diff / np.maximum(mx, 1e-8), 0.0)
    return np.stack([h, s, mx], axis=-1)


def green_mask(img: Image.Image, cfg: CropConfig, work_side: int = 256) -> np.ndarray:
    """Boolean mask of plant pixels, computed on a downscaled copy for speed."""
    small = img.copy()
    small.thumbnail((work_side, work_side), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    hsv = rgb_to_hsv(arr)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    dh = np.abs(((hue - cfg.hue_center + 180) % 360) - 180)
    mask = (dh <= cfg.hue_tol) & (sat >= cfg.sat_min) & (val >= cfg.val_min)
    if mask.sum() < 16:
        return mask
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((7, 7)))
    mask = ndimage.binary_fill_holes(mask)
    return mask


def plant_box(img: Image.Image, cfg: CropConfig) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box (left, top, right, bottom) of the plant, or None when undetected."""
    if not cfg.enabled:
        return None
    mask = green_mask(img, cfg)
    if mask.sum() < cfg.min_area_frac * mask.size:
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    sy = img.height / mask.shape[0]
    sx = img.width / mask.shape[1]
    top, bottom = rows[0] * sy, (rows[-1] + 1) * sy
    left, right = cols[0] * sx, (cols[-1] + 1) * sx
    h, w = bottom - top, right - left
    top, bottom = top - cfg.margin * h, bottom + cfg.margin * h
    left, right = left - cfg.margin * w, right + cfg.margin * w
    left = int(max(0, left))
    top = int(max(0, top))
    right = int(min(img.width, right))
    bottom = int(min(img.height, bottom))
    if right - left < 8 or bottom - top < 8:
        return None
    return (left, top, right, bottom)


def apply_crop(img: Image.Image, box: Optional[Tuple[int, int, int, int]],
               cfg: CropConfig, pad_colour: Tuple[int, int, int] = PAD_COLOUR
               ) -> Image.Image:
    """Crop to ``box``, optionally pad to a square so leaf shape is preserved.

    Order matters and used to be wrong: the crop box is expressed in the coordinates of
    the *original* image, so cropping must happen before any downscaling.  Downscaling
    first made PIL fill everything outside the resized canvas with black, which silently
    corrupted 97 of the 500 training images (every plant whose bounding box covered most
    of the frame, i.e. exactly the confusable grass species).
    """
    if box is not None:
        img = img.crop(box)
    if cfg.max_side and max(img.size) > cfg.max_side:
        ratio = cfg.max_side / max(img.size)
        img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))),
                         Image.BILINEAR)
    if cfg.mask_background:
        mask = green_mask(img, cfg)
        if mask.any():
            up = Image.fromarray((mask * 255).astype(np.uint8)).resize(img.size,
                                                                      Image.NEAREST)
            arr = np.asarray(img).copy()
            arr[np.asarray(up) <= 127] = pad_colour
            img = Image.fromarray(arr)
    if cfg.pad_to_square and img.width != img.height:
        side = max(img.width, img.height)
        square = Image.new("RGB", (side, side), pad_colour)
        square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        img = square
    return img


def build_crop_cache(files: Sequence[str], cfg: CropConfig) -> Dict[str, Optional[List[int]]]:
    """Compute (and persist) the plant bounding box for every image."""
    cache_path = os.path.join(ensure_dir(OUT_DIR), cfg.cache_name)
    cache: Dict[str, Optional[List[int]]] = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            cache = json.load(fh)
    missing = [f for f in files if os.path.relpath(f, ROOT) not in cache]
    for path in missing:
        try:
            with Image.open(path) as im:
                box = plant_box(im.convert("RGB"), cfg)
        except Exception:  # noqa: BLE001 - corrupt file => fall back to full image
            box = None
        cache[os.path.relpath(path, ROOT)] = list(box) if box else None
    if missing:
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh)
    return cache


def cache_key(path: str) -> str:
    return os.path.relpath(path, ROOT).replace(os.sep, "__").rsplit(".", 1)[0] + ".jpg"


def image_cache_dir(cfg: CropConfig, side: int) -> str:
    tag = "mask" if cfg.mask_background else ("plain" if cfg.enabled else "raw")
    return os.path.join(OUT_DIR, f"imgcache_{tag}_{CACHE_VERSION}_{side}")


def build_image_cache(files: Sequence[str], cfg: CropConfig, side: int = 448) -> str:
    """Decode every image once and store the cropped version.

    The source PNGs go up to 2840x2132, so decoding them inside the training loop costs
    far more than the forward/backward pass. Caching removes that bottleneck entirely.
    """
    out_dir = ensure_dir(image_cache_dir(cfg, side))
    boxes = build_crop_cache(files, cfg)
    cache_cfg = replace(cfg, max_side=side)
    for path in files:
        out = os.path.join(out_dir, cache_key(path))
        if os.path.exists(out):
            continue
        box = boxes.get(os.path.relpath(path, ROOT))
        try:
            img = apply_crop(load_rgb(path), tuple(box) if box else None, cache_cfg)
            img.save(out, quality=95)
        except Exception:  # noqa: BLE001 - keep a placeholder so the key always resolves
            Image.new("RGB", (side, side), (124, 116, 104)).save(out, quality=95)
    return out_dir


# --------------------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------------------


def train_transform(size: int, mean: Sequence[float], std: Sequence[float],
                    raug: bool = False, raug_ops: int = 2, raug_mag: int = 9,
                    erasing: float = 0.25) -> Callable:
    ops: List[Callable] = [
        T.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(0.8, 1.25),
                            interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomRotation(180, interpolation=InterpolationMode.BILINEAR, fill=(124, 116, 104)),
    ]
    if raug:
        ops.append(T.RandAugment(num_ops=raug_ops, magnitude=raug_mag,
                                 interpolation=InterpolationMode.BILINEAR))
    else:
        ops.append(T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    if erasing > 0:
        ops.append(T.RandomErasing(p=erasing, scale=(0.02, 0.18), value="random"))
    return T.Compose(ops)


class _Rot90:
    """Picklable 90-degree rotation (``T.Lambda`` cannot cross process boundaries)."""

    def __init__(self, k: int):
        self.k = k % 4

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.rotate(90 * self.k, expand=False) if self.k else img


def eval_transform(size: int, mean: Sequence[float], std: Sequence[float],
                   scale: float = 1.0, rot90: int = 0, hflip: bool = False) -> Callable:
    """A deterministic view; ``scale`` > 1 zooms in, ``rot90`` is a lossless rotation.

    The shortest side is always resized to at least ``1.14 * size`` before centre cropping,
    so small crops are upscaled instead of being padded with black pixels.
    """
    resize_to = int(round(size * max(scale, 1.14)))
    ops: List[Callable] = [
        T.Resize(resize_to, interpolation=InterpolationMode.BILINEAR),
        T.CenterCrop(size),
    ]
    if rot90:
        ops.append(_Rot90(rot90))
    if hflip:
        ops.append(T.RandomHorizontalFlip(1.0))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


def tta_views(size: int, mean: Sequence[float], std: Sequence[float],
              scales: Sequence[float] = (1.0, 1.15), rotations: int = 4,
              flips: bool = False) -> List[Callable]:
    """Deterministic views for test-time augmentation.

    ``flips`` doubles the view count with a horizontal mirror, which the model has seen
    during training, so it is a free (if modest) gain.
    """
    views: List[Callable] = []
    for s in scales:
        for k in range(rotations):
            for flip in ((False, True) if flips else (False,)):
                views.append(eval_transform(size, mean, std, scale=s, rot90=k,
                                            hflip=flip))
    return views


# --------------------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------------------


class _PlantMixin:
    cfg: CropConfig
    boxes: Dict[str, Optional[List[int]]]
    cache_dir: Optional[str] = None

    def _load(self, path: str) -> Image.Image:
        if self.cache_dir:
            cached = os.path.join(self.cache_dir, cache_key(path))
            if os.path.exists(cached):
                with Image.open(cached) as im:
                    return im.convert("RGB")
        img = load_rgb(path)
        box = self.boxes.get(os.path.relpath(path, ROOT))
        return apply_crop(img, tuple(box) if box else None, self.cfg)


class TrainDS(_PlantMixin, Dataset):
    def __init__(self, samples: Sequence[Tuple[str, int]], tfm: Callable,
                 cfg: CropConfig, boxes: Dict[str, Optional[List[int]]],
                 cache_dir: Optional[str] = None):
        self.samples, self.tfm, self.cfg, self.boxes = list(samples), tfm, cfg, boxes
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return self.tfm(self._load(path)), label


class EvalDS(_PlantMixin, Dataset):
    """Validation dataset: one deterministic view plus its label."""

    def __init__(self, samples: Sequence[Tuple[str, int]], tfm: Callable,
                 cfg: CropConfig, boxes: Dict[str, Optional[List[int]]],
                 cache_dir: Optional[str] = None):
        self.samples, self.tfm, self.cfg, self.boxes = list(samples), tfm, cfg, boxes
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return self.tfm(self._load(path)), label


class InferDS(_PlantMixin, Dataset):
    """Serves ``len(views)`` TTA crops per image as one stacked tensor."""

    def __init__(self, samples: Sequence[Tuple[str, object]], views: Sequence[Callable],
                 cfg: CropConfig, boxes: Dict[str, Optional[List[int]]],
                 cache_dir: Optional[str] = None):
        self.samples, self.views, self.cfg, self.boxes = list(samples), list(views), cfg, boxes
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, tag = self.samples[idx]
        img = self._load(path)
        return torch.stack([v(img) for v in self.views]), idx


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int,
                drop_last: bool = False, pin_memory: bool = True) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=drop_last,
                      pin_memory=pin_memory, persistent_workers=num_workers > 0)


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------


_ARCHS = {
    "resnet18": (torchvision.models.resnet18, "ResNet18_Weights"),
    "resnet50": (torchvision.models.resnet50, "ResNet50_Weights"),
    "efficientnet_b0": (torchvision.models.efficientnet_b0, "EfficientNet_B0_Weights"),
    "efficientnet_b3": (torchvision.models.efficientnet_b3, "EfficientNet_B3_Weights"),
    "convnext_tiny": (torchvision.models.convnext_tiny, "ConvNeXt_Tiny_Weights"),
    "convnext_small": (torchvision.models.convnext_small, "ConvNeXt_Small_Weights"),
}


def build_model(arch: str, num_classes: int, pretrained: bool = True,
                img_size: Optional[int] = None) -> nn.Module:
    if arch not in _ARCHS:
        return _build_timm_model(arch, num_classes, pretrained, img_size)
    ctor, weights_name = _ARCHS[arch]
    weights = None
    if pretrained:
        weights = getattr(torchvision.models, weights_name).IMAGENET1K_V1
    model = ctor(weights=weights)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):  # resnets
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Linear):
            model.classifier = nn.Linear(clf.in_features, num_classes)
        elif isinstance(clf, nn.Sequential):
            for i in range(len(clf) - 1, -1, -1):
                if isinstance(clf[i], nn.Linear):
                    clf[i] = nn.Linear(clf[i].in_features, num_classes)
                    break
            else:
                raise ValueError(f"no Linear layer found in {arch} classifier")
        else:
            raise ValueError(f"unsupported classifier type for {arch}")
    else:
        raise ValueError(f"cannot locate head for {arch}")
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = trainable
    head = get_head_params(model)
    for param in head:
        param.requires_grad = True


def get_head_params(model: nn.Module) -> List[nn.Parameter]:
    if hasattr(model, "get_classifier"):  # timm models expose their head explicitly
        head = model.get_classifier()
        return list(head.parameters()) if head is not None else []
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return list(model.fc.parameters())
    if isinstance(model.classifier, nn.Linear):
        return list(model.classifier.parameters())
    params: List[nn.Parameter] = []
    for module in model.classifier.modules():
        if isinstance(module, nn.Linear):
            params += list(module.parameters())
    return params


def _build_timm_model(name: str, num_classes: int, pretrained: bool = True,
                      img_size: Optional[int] = None) -> nn.Module:
    """Fall back to timm for architectures torchvision does not ship (e.g. in21k weights)."""
    import timm  # imported lazily: only needed for timm backbones

    if img_size:
        # Transformers (Swin/ViT) assert an exact input size unless told about it up front.
        try:
            return timm.create_model(name, pretrained=pretrained, num_classes=num_classes,
                                     img_size=img_size)
        except TypeError:
            pass
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


# --------------------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------------------


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam


def cutmix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    h, w = x.shape[-2:]
    ratio = float(np.sqrt(1 - lam))
    cut_h, cut_w = int(round(h * ratio)), int(round(w * ratio))
    if cut_h < 1 or cut_w < 1:
        return x, y, y, 1.0
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
    lam = 1 - ((y2 - y1) * (x2 - x1) / (h * w))
    return x, y, y[perm], lam


class EMA:
    """Exponential moving average of the model weights (evaluated instead of the raw ones)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        model.load_state_dict({k: (self.shadow[k].to(state[k].dtype)
                                   if k in self.shadow else state[k])
                               for k in state})


@torch.no_grad()
def logits_for(model: nn.Module, loader: DataLoader, device: torch.device,
               tta: bool = False, chunk: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean probabilities over TTA views, sample indices) in dataset order.

    ``chunk`` bounds how many views are pushed through the network at once, so TTA on a
    batch of images does not blow up the 8 GB of VRAM.
    """
    model.eval()
    all_logits: Dict[int, List[np.ndarray]] = {}
    for batch, idx in loader:
        batch = batch.to(device, non_blocking=True)
        if batch.dim() == 5:
            b, n_views = batch.shape[0], batch.shape[1]
            flat = batch.flatten(0, 1)
            parts = []
            for start in range(0, flat.shape[0], chunk):
                parts.append(model(flat[start:start + chunk]).float().softmax(dim=1))
            out = torch.cat(parts).reshape(b, n_views, -1).mean(dim=1)
        else:
            out = model(batch).float().softmax(dim=1)
        out = out.cpu().numpy()
        for row, i in zip(out, idx.tolist()):
            all_logits.setdefault(i, []).append(row)
    order = sorted(all_logits)
    return np.stack([np.mean(all_logits[i], axis=0) for i in order]), np.array(order)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if x.dim() == 5:  # TTA-shaped batch (B, views, C, H, W)
            b, v = x.shape[0], x.shape[1]
            out = model(x.flatten(0, 1)).reshape(b, v, -1).mean(dim=1)
        else:
            out = model(x)
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum())
        total += y.numel()
    return correct / max(total, 1)


def train_one_fold(tr_loader: DataLoader, va_loader: DataLoader, cfg: TrainConfig,
                   device: torch.device, num_classes: int = len(CLASSES),
                   log: Callable[[str], None] = print) -> Tuple[nn.Module, Dict]:
    seed_everything(cfg.seed)
    model = build_model(cfg.arch, num_classes, img_size=cfg.size).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    total_epochs = cfg.head_epochs + cfg.finetune_epochs
    best_acc, best_state, best_epoch = -1.0, None, -1

    def run_epoch(optimizer, scheduler) -> Tuple[float, float]:
        model.train()
        running, seen, correct = 0.0, 0, 0
        for x, y in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            ya = yb = None
            lam = 1.0
            if cfg.mixup_alpha > 0 and cfg.cutmix_alpha > 0:
                fn = mixup_batch if np.random.rand() < 0.5 else cutmix_batch
                x, ya, yb, lam = fn(x, y, cfg.mixup_alpha if fn is mixup_batch
                                    else cfg.cutmix_alpha)
            elif cfg.mixup_alpha > 0:
                x, ya, yb, lam = mixup_batch(x, y, cfg.mixup_alpha)
            elif cfg.cutmix_alpha > 0:
                x, ya, yb, lam = cutmix_batch(x, y, cfg.cutmix_alpha)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                out = model(x)
                loss = (criterion(out, y) if ya is None else
                        lam * criterion(out, ya) + (1 - lam) * criterion(out, yb))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            if scheduler is not None:
                scheduler.step()
            running += float(loss.detach()) * y.numel()
            seen += y.numel()
            correct += int((out.detach().argmax(1) == y).sum())
        return running / max(seen, 1), correct / max(seen, 1)

    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    set_backbone_trainable(model, False)
    opt = torch.optim.AdamW(get_head_params(model), lr=cfg.head_lr, weight_decay=cfg.weight_decay)
    for epoch in range(1, cfg.head_epochs + 1):
        loss, acc = run_epoch(opt, None)
        va = evaluate(model, va_loader, device)
        history["train_loss"].append(loss)
        history["train_acc"].append(acc)
        history["val_acc"].append(va)
        log(f"    [head {epoch:02d}/{cfg.head_epochs}] loss={loss:.4f} train_acc={acc:.4f} val_acc={va:.4f}")

    set_backbone_trainable(model, True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.finetune_epochs * max(len(tr_loader), 1),
        pct_start=0.15, anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0)
    for epoch in range(1, cfg.finetune_epochs + 1):
        loss, acc = run_epoch(opt, sched)
        va = evaluate(model, va_loader, device)
        history["train_loss"].append(loss)
        history["train_acc"].append(acc)
        history["val_acc"].append(va)
        log(f"    [ft {epoch:02d}/{cfg.finetune_epochs}] loss={loss:.4f} train_acc={acc:.4f} val_acc={va:.4f}")
        if va > best_acc:
            best_acc, best_epoch = va, epoch + cfg.head_epochs
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # The final weights are kept (not the best-val checkpoint) so that the out-of-fold
    # estimate stays honest: picking the epoch by validation accuracy would leak the
    # validation set into model selection.
    if ema is not None:
        ema.copy_to(model)
        history["ema_val_acc"] = evaluate(model, va_loader, device)
    history["final_val_acc"] = history["val_acc"][-1] if history["val_acc"] else float("nan")
    history["best_val_acc"] = best_acc
    history["best_epoch"] = best_epoch
    log(f"    final val_acc={history['final_val_acc']:.4f} "
        f"(best {best_acc:.4f} @ epoch {best_epoch}/{total_epochs})")
    return model, history
