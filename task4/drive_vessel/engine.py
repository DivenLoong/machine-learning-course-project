from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import (
    RetinaDataset,
    list_drive_samples,
    list_external_samples,
    list_extra_samples,
)
from .losses import build_loss
from .metrics import binary_metrics, summarize_metrics
from .models import build_model, load_initial_weights
from .splits import load_or_create_folds
from .utils import get_device, seed_everything, seed_worker, write_json


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.module = deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for name, value in self.module.state_dict().items():
            source = model_state[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                value.copy_(source)


def _dataset(samples, config, training: bool) -> RetinaDataset:
    return RetinaDataset(
        samples,
        size=int(config["data"]["size"]),
        training=training,
        preprocessing=str(config["data"].get("preprocessing", "raw")),
        augment=bool(config["train"].get("augment", True)),
        augment_profile=str(config["train"].get("augment_profile", "strong")),
        fov_center_crop=bool(config["train"].get("fov_center_crop", False)),
        patches_per_image=int(config["train"].get("patches_per_image", 1)),
        train_patch_size=config["data"].get("train_patch_size"),
        cache=training,
        input_channels=int(config["data"].get("input_channels", 3)),
        normalization=str(config["data"].get("normalization", "imagenet")),
    )


def _loader(dataset, config, training: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=training,
        num_workers=int(config["train"].get("workers", 2)),
        pin_memory=True,
        persistent_workers=int(config["train"].get("workers", 2)) > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    max_batches: int | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray], list[dict]]:
    model.eval()
    probabilities: dict[str, np.ndarray] = {}
    rows = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
            logits = model(images)
        probability = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
        target = targets.cpu().numpy()[:, 0]
        for index, image_id in enumerate(batch["image_id"]):
            prediction = (probability[index] >= threshold).astype(np.uint8)
            rows.append({"id": image_id, **binary_metrics(target[index] > 0.5, prediction)})
            probabilities[image_id] = probability[index]
    summary = summarize_metrics(rows)
    return summary, probabilities, rows


def pixel_auc(target: np.ndarray, probability: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if np.unique(target).size < 2:
        return float("nan")
    return float(roc_auc_score(target, probability))


def train_fold(config: dict, fold: int, run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"]) + fold
    seed_everything(seed, bool(config["train"].get("deterministic", False)))
    device = get_device(config)
    samples = list_drive_samples(config["data"]["root"], "train")
    folds_path = Path(config["data"]["folds_file"])
    fold_data = load_or_create_folds(
        folds_path, samples, int(config["data"]["size"]), int(config["data"]["num_folds"])
    )
    validation_ids = set(fold_data["folds"][fold])
    train_all = bool(config["data"].get("train_all", False))
    train_samples = (
        list(samples)
        if train_all
        else [sample for sample in samples if sample.image_id not in validation_ids]
    )
    validation_samples = [sample for sample in samples if sample.image_id in validation_ids]
    external_config = config["data"].get("external", {})
    external_samples = (
        list_external_samples(
            external_config.get("chase_root"),
            external_config.get("stare_root"),
            external_config.get("fives_root"),
        )
        if external_config.get("enabled", False)
        else []
    )
    external_limit = int(external_config.get("max_samples", 0))
    if external_limit > 0 and len(external_samples) > external_limit:
        stride = max(1, len(external_samples) // external_limit)
        external_samples = external_samples[::stride][:external_limit]
    local_repeat = int(external_config.get("local_repeat", 3))
    if external_samples:
        if external_config.get("external_only", False):
            train_samples = list(external_samples)
        else:
            train_samples = train_samples * local_repeat + external_samples
    pseudo_config = config["data"].get("pseudo", {})
    pseudo_samples = (
        list_extra_samples(pseudo_config["root"], "pseudo")
        if pseudo_config.get("enabled", False)
        else []
    )
    if pseudo_samples:
        train_samples.extend(
            pseudo_samples * int(pseudo_config.get("repeat", 1))
        )

    train_loader = _loader(_dataset(train_samples, config, True), config, True, seed)
    validation_loader = _loader(
        _dataset(validation_samples, config, False), config, False, seed
    )

    model = build_model(config["model"]).to(device)
    initialization = (
        load_initial_weights(model, config["model"]["init_checkpoint"])
        if config["model"].get("init_checkpoint")
        else None
    )
    criterion = build_loss(str(config["loss"]["name"]))
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    epochs = int(config["train"]["epochs"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(config["train"]["lr"]) * float(config["train"].get("min_lr_ratio", 0.01)),
    )
    ema = (
        ModelEMA(model, float(config["train"]["ema_decay"]))
        if float(config["train"].get("ema_decay", 0.0)) > 0
        else None
    )
    accumulation = int(config["train"].get("gradient_accumulation", 1))
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and config["train"].get("amp_dtype", "bf16") == "bf16"
        else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is None and device.type == "cuda")
    best_dice = -1.0
    best_epoch = -1
    stale_epochs = 0
    history = []
    limit_batches = config["train"].get("limit_train_batches")
    limit_val_batches = config["train"].get("limit_val_batches")

    print(
        f"[fold {fold}] training={len(train_samples)} validation={len(validation_samples)} "
        f"model={config['model']['name']} preprocess={config['data']['preprocessing']}"
        f" external={len(external_samples)} pseudo={len(pseudo_samples)}"
    )
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        steps = 0
        progress = tqdm(train_loader, leave=False, desc=f"fold {fold} epoch {epoch}")
        pending_backward = False
        for batch_index, batch in enumerate(progress):
            if limit_batches is not None and batch_index >= int(limit_batches):
                break
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets) / accumulation
            scaler.scale(loss).backward()
            pending_backward = True
            if (batch_index + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_backward = False
                if ema is not None:
                    ema.update(model)
            loss_sum += float(loss.detach()) * accumulation
            steps += 1
            progress.set_postfix(loss=f"{loss_sum / steps:.4f}")
        if pending_backward:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        if steps == 0:
            raise RuntimeError("No training batches were processed")

        scheduler.step()
        evaluation_model = ema.module if ema is not None else model
        validation_summary, probabilities, per_image = evaluate_model(
            evaluation_model,
            validation_loader,
            device,
            threshold=float(config["eval"].get("threshold", 0.5)),
            amp_dtype=amp_dtype,
            max_batches=int(limit_val_batches) if limit_val_batches is not None else None,
        )
        duration = time.time() - started
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / steps,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": duration,
            **{f"val_{key}": value for key, value in validation_summary.items()},
        }
        history.append(row)
        print(
            f"[fold {fold}] epoch={epoch} loss={row['train_loss']:.4f} "
            f"dice={validation_summary['dice']:.4f} "
            f"se={validation_summary['sensitivity']:.4f} "
            f"sp={validation_summary['specificity']:.4f} time={duration:.1f}s"
        )

        if validation_summary["dice"] > best_dice:
            best_dice = validation_summary["dice"]
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model": evaluation_model.state_dict(),
                "config": config,
                "fold": fold,
                "epoch": epoch,
                "validation_dice": best_dice,
            }
            torch.save(checkpoint, run_dir / "best.pt")
            np.savez_compressed(
                run_dir / "oof_probs.npz", **{key: value.astype(np.float16) for key, value in probabilities.items()}
            )
            write_json(run_dir / "per_image_metrics.json", {"fold": fold, "best_epoch": best_epoch, "rows": per_image})
        else:
            stale_epochs += 1
        write_json(run_dir / "history.json", {"fold": fold, "history": history})
        if stale_epochs >= int(config["train"].get("early_stopping_patience", 40)):
            print(f"[fold {fold}] early stop after {stale_epochs} stale epochs")
            break

    result = {
        "fold": fold,
        "best_dice": best_dice,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "validation_ids": sorted(validation_ids, key=int),
        "model": config["model"]["name"],
        "preprocessing": config["data"]["preprocessing"],
        "loss": config["loss"]["name"],
        "external_samples": len(external_samples),
        "pseudo_samples": len(pseudo_samples),
        "initialization": initialization,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "final_info.json", result)
    return result
