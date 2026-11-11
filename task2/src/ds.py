"""Load the six-class course dataset and augment grayscale batches on device.

Images are cached as uint8 arrays. Training augments batches with tensor
operations after moving the cached images to the GPU.
"""
from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.abspath(os.path.expanduser(os.environ.get(
    "FER_PROJECT_ROOT", os.path.join(os.path.dirname(__file__), "..")
)))
COMP_DIR = os.path.join(ROOT, "2026task2")
DATA_DIR = os.path.join(COMP_DIR, "fer_data", "fer_data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
SAMPLE_SUB = os.path.join(COMP_DIR, "submission.csv")
OUT_DIR = os.path.join(ROOT, "outputs")
RUNS_DIR = os.path.join(OUT_DIR, "runs")
SUB_DIR = os.path.join(ROOT, "submissions")

# Competition label order, from the competition rules:
#   0 Angry | 1 Fear | 2 Happy | 3 Sad | 4 Surprise | 5 Neutral
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLS = len(CLASSES)
IMG_SIZE = 48
PIXELS = IMG_SIZE * IMG_SIZE

DATA_CACHE = os.path.join(OUT_DIR, "data_cache.npz")
FOLDS_PATH = os.path.join(OUT_DIR, "folds.json")
NORM_PATH = os.path.join(OUT_DIR, "norm.json")


def ensure_dirs() -> None:
    for d in (OUT_DIR, RUNS_DIR, SUB_DIR):
        os.makedirs(d, exist_ok=True)


def eval_batch_for(res: int) -> int:
    """Inference batch size: activations, not the input tensor, dominate memory."""
    if res >= 192:
        return 128
    if res >= 96:
        return 256
    if res >= 64:
        return 512
    return 1024


# --------------------------------------------------------------------------- #
# raw data
# --------------------------------------------------------------------------- #
def _decode(path: str) -> np.ndarray:
    with Image.open(path) as im:
        if im.size != (IMG_SIZE, IMG_SIZE):
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        a = np.asarray(im.convert("L"), dtype=np.uint8)
    return a.reshape(-1)


def _decode_many(paths, workers: int = 16, label: str = "") -> np.ndarray:
    out = np.empty((len(paths), PIXELS), np.uint8)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, a in enumerate(ex.map(_decode, paths)):
            out[i] = a
            if label and (i + 1) % 10000 == 0:
                print(f"    decoded {i + 1}/{len(paths)} {label}", flush=True)
    return out


def train_paths_labels():
    paths, labels = [], []
    for name in CLASSES:
        fs = sorted(glob.glob(os.path.join(TRAIN_DIR, name, "*.jpg")))
        if not fs:
            raise FileNotFoundError(os.path.join(TRAIN_DIR, name))
        paths.extend(fs)
        labels.extend([CLS2IDX[name]] * len(fs))
    return paths, np.asarray(labels, np.int64)


def test_paths():
    fs = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
    if not fs:
        raise FileNotFoundError(TEST_DIR)
    return fs


def load_raw(rebuild: bool = False):
    ensure_dirs()
    if not rebuild and os.path.exists(DATA_CACHE):
        z = np.load(DATA_CACHE, allow_pickle=False)
        return z["tr_px"], z["tr_y"], z["te_px"], z["te_ids"]
    tr_paths, tr_y = train_paths_labels()
    te_paths = test_paths()
    print(f"decoding {len(tr_paths)} train + {len(te_paths)} test images ...", flush=True)
    tr_px = _decode_many(tr_paths, label="train")
    te_px = _decode_many(te_paths, label="test")
    te_ids = np.asarray([os.path.basename(p) for p in te_paths])
    np.savez(DATA_CACHE, tr_px=tr_px, tr_y=tr_y, te_px=te_px, te_ids=te_ids)
    return tr_px, tr_y, te_px, te_ids


def norm_stats(rebuild: bool = False):
    if not rebuild and os.path.exists(NORM_PATH):
        with open(NORM_PATH) as fh:
            d = json.load(fh)
        return d["mean"], d["std"]
    tr_px, _, _, _ = load_raw()
    x = tr_px.astype(np.float32) / 255.0
    mean, std = float(x.mean()), float(x.std())
    ensure_dirs()
    with open(NORM_PATH, "w") as fh:
        json.dump({"mean": mean, "std": std}, fh, indent=2)
    print(f"normalisation: mean={mean:.4f} std={std:.4f}")
    return mean, std


def make_folds(n_splits: int = 5, seed: int = 42, rebuild: bool = False):
    if not rebuild and os.path.exists(FOLDS_PATH):
        with open(FOLDS_PATH) as fh:
            d = json.load(fh)
        if d.get("n_splits") == n_splits and d.get("seed") == seed:
            return d["folds"]
    from sklearn.model_selection import StratifiedKFold

    _, tr_y, _, _ = load_raw()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [
        {"train": tr.tolist(), "val": va.tolist()}
        for tr, va in skf.split(np.zeros(len(tr_y)), tr_y)
    ]
    ensure_dirs()
    with open(FOLDS_PATH, "w") as fh:
        json.dump({"n_splits": n_splits, "seed": seed, "folds": folds}, fh)
    return folds


# --------------------------------------------------------------------------- #
# GPU resident tensors
# --------------------------------------------------------------------------- #
class Device:
    def __init__(self, device: str = "cuda"):
        tr_px, tr_y, te_px, te_ids = load_raw()
        self.device = torch.device(device)
        self.train_px = torch.from_numpy(tr_px).to(self.device)
        self.train_y = torch.from_numpy(tr_y).to(self.device)
        self.test_px = torch.from_numpy(te_px).to(self.device)
        self.test_ids = list(te_ids)
        self.n_train = self.train_px.shape[0]
        self.n_test = self.test_px.shape[0]

    def train_batch(self, idx):
        return self.train_px[idx], self.train_y[idx]

    def val_batch(self, idx):
        return self.train_px[idx], self.train_y[idx]


# --------------------------------------------------------------------------- #
# batched GPU augmentation
# --------------------------------------------------------------------------- #
class GpuAug:
    """Augmentation for a uint8 [B,2304] batch, entirely on the GPU.

    order: float -> upsample to `res` -> affine -> flip -> brightness/contrast ->
           random erasing -> normalise -> (mixup | cutmix)
    All per-sample randomness is drawn as one [B,16] tensor and read back with a
    single `.tolist()`, so there is no per-sample device synchronisation.
    """

    R_AFF = 5      # ang, shear, scale, tx, ty
    R_FLIP = 1
    R_PHOT = 2     # brightness, contrast
    R_ERASE = 5    # hit, area, ratio, y, x
    R_MIX = 4      # mode, lam, box-y, box-x
    N_RAND = R_AFF + R_FLIP + R_PHOT + R_ERASE + R_MIX

    def __init__(
        self,
        res: int,
        mean: float,
        std: float,
        device="cuda",
        train: bool = True,
        rot: float = 12.0,
        trans: float = 0.08,
        scale: float = 0.10,
        shear: float = 5.0,
        hflip: float = 0.5,
        erase_p: float = 0.25,
        erase_area=(0.02, 0.15),
        erase_ratio=(0.3, 1.2),
        brightness: float = 0.15,
        contrast: float = 0.15,
        mix_p: float = 0.5,
        mixup_a: float = 0.2,
        cutmix_a: float = 1.0,
        use_affine: bool = True,
        use_erase: bool = True,
        use_mix: bool = True,
        in_ch: int = 3,
        seed: int = 0,
        np_rng: np.random.Generator | None = None,
        **extra,
    ):
        self.res = int(res)
        self.mean = float(mean)
        self.std = float(std)
        self.device = torch.device(device)
        self.train = train
        self.rot = float(rot)
        self.trans = float(trans)
        self.scale = float(scale)
        self.shear = float(shear)
        self.hflip = float(hflip)
        self.erase_p = float(erase_p)
        self.erase_area = erase_area
        self.erase_ratio = erase_ratio
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.mix_p = float(mix_p)
        self.mixup_a = float(mixup_a)
        self.cutmix_a = float(cutmix_a)
        self.use_affine = use_affine
        self.use_erase = use_erase
        self.use_mix = use_mix
        self.in_ch = int(in_ch)
        self.mix_off = False
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))
        self.np = np_rng or np.random.default_rng(seed)

    # -- steps ------------------------------------------------------------ #
    def _theta(self, rl):
        b = len(rl)
        rows = []
        deg = np.pi / 180.0
        for i in range(b):
            ang = (rl[i][0] * 2 - 1) * self.rot * deg
            shx = (rl[i][1] * 2 - 1) * self.shear * deg
            sc = 1.0 + (rl[i][2] * 2 - 1) * self.scale
            tx = (rl[i][3] * 2 - 1) * self.trans
            ty = (rl[i][4] * 2 - 1) * self.trans
            c, s = np.cos(ang), np.sin(ang)
            m00, m01 = c * sc, (c * shx - s) * sc
            m10, m11 = s * sc, (s * shx + c) * sc
            det = m00 * m11 - m01 * m10
            rows.append([m11 / det, -m01 / det, tx, -m10 / det, m00 / det, ty])
        return torch.tensor(rows, dtype=torch.float32, device=self.device).view(b, 2, 3)

    def _affine(self, x, rl):
        b = x.shape[0]
        theta = self._theta(rl)
        grid = F.affine_grid(theta, (b, 1, self.res, self.res), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _flip(self, x, rl):
        base = self.R_AFF
        m = torch.tensor([1.0 if rl[i][base] >= self.hflip else 0.0 for i in range(len(rl))],
                         device=self.device).view(-1, 1, 1, 1)
        return x * (1 - m) + torch.flip(x, dims=[3]) * m

    def _photometric(self, x, rl):
        base = self.R_AFF + self.R_FLIP
        b = x.shape[0]
        if self.brightness > 0:
            f = torch.tensor([1.0 + (rl[i][base] * 2 - 1) * self.brightness for i in range(b)],
                             device=self.device).view(b, 1, 1, 1)
            x = x * f
        if self.contrast > 0:
            f = torch.tensor([1.0 + (rl[i][base + 1] * 2 - 1) * self.contrast for i in range(b)],
                             device=self.device).view(b, 1, 1, 1)
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            x = (x - m) * f + m
        return x.clamp_(0.0, 1.0)

    def _erase(self, x, rl):
        base = self.R_AFF + self.R_FLIP + self.R_PHOT
        _, _, h, w = x.shape
        for i in range(x.shape[0]):
            if rl[i][base] >= self.erase_p:
                continue
            area = self.erase_area[0] + rl[i][base + 1] * (self.erase_area[1] - self.erase_area[0])
            ratio = self.erase_ratio[0] + rl[i][base + 2] * (self.erase_ratio[1] - self.erase_ratio[0])
            eh = min(h - 1, max(1, int(round((area * h * w * ratio) ** 0.5))))
            ew = min(w - 1, max(1, int(round((area * h * w / ratio) ** 0.5))))
            y0 = int(rl[i][base + 3] * (h - eh))
            x0 = int(rl[i][base + 4] * (w - ew))
            x[i, :, y0 : y0 + eh, x0 : x0 + ew] = 0.0
        return x

    def _mix(self, x, y, rl):
        b = x.shape[0]
        base = self.R_AFF + self.R_FLIP + self.R_PHOT + self.R_ERASE
        if b < 2 or not self.use_mix or self.mix_off or rl[0][base] >= self.mix_p:
            return x, y, y, 1.0
        perm = torch.randperm(b, generator=self.gen, device=self.device)
        xb, yb = x[perm], y[perm]
        if rl[0][base + 1] < 0.5:  # mixup
            lam = float(self.np.beta(self.mixup_a, self.mixup_a))
            lam = max(lam, 1 - lam)
            x = x * lam + xb * (1 - lam)
        else:  # cutmix
            lam = float(self.np.beta(self.cutmix_a, self.cutmix_a))
            lam = max(lam, 1 - lam)
            h = w = self.res
            rh, rw = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
            if rh > 0 and rw > 0:
                cy = int(rl[0][base + 2] * (h - rh))
                cx = int(rl[0][base + 3] * (w - rw))
                x[:, :, cy : cy + rh, cx : cx + rw] = xb[:, :, cy : cy + rh, cx : cx + rw]
                lam = 1.0 - (rh * rw) / (h * w)
        return x, y, yb, float(lam)

    # -- public ----------------------------------------------------------- #
    def __call__(self, px_u8, y):
        b = px_u8.shape[0]
        rl = torch.rand((b, self.N_RAND), generator=self.gen, device=self.device).tolist()
        x = px_u8.view(-1, 1, IMG_SIZE, IMG_SIZE).float().div_(255.0)
        if self.res != IMG_SIZE:
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        if self.train:
            if self.use_affine:
                x = self._affine(x, rl)
            x = self._flip(x, rl)
            x = self._photometric(x, rl)
            if self.use_erase:
                x = self._erase(x, rl)
        x = (x - self.mean) / self.std
        if self.in_ch == 3:
            x = x.repeat(1, 3, 1, 1)
        if self.train:
            return self._mix(x, y, rl)
        return x, y, y, 1.0


def soft_ce(logits, ya, yb, lam, label_smoothing: float = 0.1):
    """Cross entropy for (possibly mixed) targets, with label smoothing."""
    loss = F.cross_entropy(logits, ya, label_smoothing=label_smoothing)
    if yb is not None and lam < 1.0:
        loss = lam * loss + (1 - lam) * F.cross_entropy(
            logits, yb, label_smoothing=label_smoothing
        )
    return loss


class EMA:
    def __init__(self, model, decay: float = 0.999):
        self.module = model
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point
        }
        self.backup = {}

    @torch.no_grad()
    def update(self):
        for k, v in self.module.state_dict().items():
            s = self.shadow.get(k)
            if s is not None:
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply(self):
        self.backup = {}
        for k, v in self.module.state_dict().items():
            if k in self.shadow:
                self.backup[k] = v.detach().clone()
                v.copy_(self.shadow[k])

    def restore(self):
        for k, v in self.backup.items():
            self.module.state_dict()[k].copy_(v)
        self.backup = {}
