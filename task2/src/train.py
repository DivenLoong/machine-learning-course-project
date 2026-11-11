"""Train one (backbone, resolution, fold) run.

Example:
    python src/train.py --backbone convnext_tiny.fb_in22k_ft_in1k --res 224 \
        --batch 32 --epochs 60 --fold 0 --tag ctn224_f0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds  # noqa: E402
import models as M  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k")
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eval-batch", type=int, default=0, help="0 = auto from resolution")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--min-lr-factor", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--limit", type=int, default=0, help="smoke test: use only N train images")
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--mix-p", type=float, default=0.5)
    p.add_argument("--mix-off-epochs", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=2,
                   help="validate every N epochs; the final epoch is always validated")
    p.add_argument("--channels-last", type=int, default=1,
                   help="1 = NHWC memory format (usually faster for conv nets on Ampere+)")
    p.add_argument("--swa-epochs", default="6,8,10,12,14,16",
                   help="epoch numbers whose EMA weights are averaged into an SWA model; "
                        "empty string disables it")
    p.add_argument("--no-mix", action="store_true")
    p.add_argument("--no-affine", action="store_true")
    p.add_argument("--no-erase", action="store_true")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--rot", type=float, default=12.0)
    p.add_argument("--scale", type=float, default=0.10)
    p.add_argument("--erase-p", type=float, default=0.25)
    p.add_argument("--brightness", type=float, default=0.15)
    p.add_argument("--contrast", type=float, default=0.15)
    p.add_argument("--pretrained", type=int, default=1)
    p.add_argument("--full-data", action="store_true", help="train on all images (no val split)")
    return p.parse_args(argv)


def lr_at(step: int, total: int, warmup: int, base: float, min_factor: float) -> float:
    if step < warmup:
        return base * (0.05 + 0.95 * step / max(1, warmup))
    t = (step - warmup) / max(1, total - warmup)
    return base * (min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * t)))


def main(argv=None):
    a = parse_args(argv)
    ds.ensure_dirs()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    # Fixed input shapes -> let cuDNN pick the fastest kernel once per shape.
    torch.backends.cudnn.benchmark = True
    # TF32 for fp32 matmul / cudnn convolutions (default off for matmul on this build).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"

    tag = a.tag or f"{a.backbone.split('.')[0]}_{a.res}_f{a.fold}"
    run_dir = os.path.join(ds.RUNS_DIR, tag)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(vars(a), fh, indent=2)

    dev = ds.Device(device)
    mean, std = ds.norm_stats()
    folds = ds.make_folds(a.n_folds)

    if a.full_data:
        tr_idx = np.arange(dev.n_train)
        va_idx = np.array([], np.int64)
    else:
        tr_idx = np.asarray(folds[a.fold]["train"], np.int64)
        va_idx = np.asarray(folds[a.fold]["val"], np.int64)
    if a.limit:
        tr_idx = tr_idx[: a.limit]

    model, in_ch, native = M.build_model(
        a.backbone, num_classes=ds.N_CLS, drop_path=a.drop_path, pretrained=bool(a.pretrained)
    )
    model = model.to(device)
    if a.channels_last:
        model = model.to(memory_format=torch.channels_last)
    n_par = sum(p.numel() for p in model.parameters())
    eval_batch = a.eval_batch or ds.eval_batch_for(a.res)
    print(f"[{tag}] backbone={a.backbone} res={a.res} in_ch={in_ch} params={n_par/1e6:.2f}M "
          f"train={len(tr_idx)} val={len(va_idx)} eval_batch={eval_batch}", flush=True)

    aug = ds.GpuAug(a.res, mean, std, device=device, train=True, in_ch=in_ch,
                    rot=a.rot, scale=a.scale, mix_p=a.mix_p, erase_p=a.erase_p,
                    brightness=a.brightness, contrast=a.contrast,
                    use_mix=not a.no_mix, use_affine=not a.no_affine,
                    use_erase=not a.no_erase, seed=a.seed * 1000 + a.fold,
                    np_rng=np.random.default_rng(a.seed * 7919 + a.fold))
    evalaug = ds.GpuAug(a.res, mean, std, device=device, train=False, in_ch=in_ch)

    def cl(t):
        return t.contiguous(memory_format=torch.channels_last) if a.channels_last else t

    head, backbone = M.head_and_backbone_params(model)
    groups = [
        {"params": backbone, "lr": a.lr_backbone},
        {"params": head, "lr": a.lr_head},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    base_lrs = [a.lr_backbone, a.lr_head]
    scaler = torch.amp.GradScaler("cuda")
    ema = None if a.no_ema else ds.EMA(model, a.ema_decay)

    tr_t = torch.from_numpy(tr_idx).to(device)
    n_tr = len(tr_idx)
    steps_per_epoch = max(1, n_tr // a.batch)
    total_steps = steps_per_epoch * a.epochs
    warmup_steps = steps_per_epoch * a.warmup

    log_path = os.path.join(run_dir, "log.jsonl")
    log_fh = open(log_path, "a", encoding="utf-8")
    t_start = time.time()
    step = 0
    best = {"acc": -1.0, "epoch": -1, "which": "raw"}
    # SWA over the early EMA window: single-epoch selection on a 5.6k val set is
    # noise-dominated, averaging several neighbouring EMA snapshots is steadier.
    swa_eps = {int(x) for x in a.swa_epochs.split(",") if x.strip()} if a.swa_epochs else set()
    swa_sum, swa_count = None, 0

    @torch.no_grad()
    def evaluate(idx, batch):
        model.eval()
        correct = torch.zeros((), device=device)
        for s in range(0, len(idx), batch):
            sub = torch.from_numpy(np.asarray(idx[s : s + batch])).to(device)
            px, y = dev.val_batch(sub)
            x, _, _, _ = evalaug(px, y)
            x = cl(x)
            with torch.amp.autocast("cuda"):
                out = model(x)
            correct += (out.float().argmax(1) == y).sum()
        model.train()
        return float(correct / max(1, len(idx)))

    for epoch in range(a.epochs):
        aug.mix_off = a.no_mix or epoch >= a.epochs - a.mix_off_epochs
        perm = torch.randperm(n_tr, device=device)
        run_loss, seen, corr = 0.0, 0, 0
        t_ep = time.time()
        for it in range(steps_per_epoch):
            sel = tr_t[perm[it * a.batch : (it + 1) * a.batch]]
            if sel.numel() == 0:
                continue
            px, y = dev.train_batch(sel)
            x, ya, yb, lam = aug(px, y)
            x = cl(x)
            for gi, g in enumerate(opt.param_groups):
                g["lr"] = lr_at(step, total_steps, warmup_steps, base_lrs[gi], a.min_lr_factor)
            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = ds.soft_ce(out, ya, yb, lam, a.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update()
            run_loss += float(loss.detach()) * sel.numel()
            seen += sel.numel()
            # training accuracy against the primary (unmixed) label
            corr += int((out.detach().float().argmax(1) == ya).sum())
            step += 1

        rec = {
            "epoch": epoch,
            "step": step,
            "loss": run_loss / max(1, seen),
            "train_acc": corr / max(1, seen),
            "lr": opt.param_groups[0]["lr"],
            "sec": round(time.time() - t_ep, 1),
            "elapsed_min": round((time.time() - t_start) / 60, 2),
        }
        if ema is not None and epoch in swa_eps:
            if swa_sum is None:
                swa_sum = {k: v.detach().clone() for k, v in ema.shadow.items()}
            else:
                for k, v in swa_sum.items():
                    v.add_(ema.shadow[k])
            swa_count += 1
            rec["swa_count"] = swa_count
        do_eval = bool(len(va_idx)) and (
            epoch % max(1, a.eval_every) == 0 or epoch == a.epochs - 1
        )
        if do_eval:
            rec["evaluated"] = True
            rec["val_acc"] = evaluate(va_idx, eval_batch)
            if ema is not None:
                ema.apply()
                rec["val_acc_ema"] = evaluate(va_idx, eval_batch)
                ema.restore()
            cand = {"raw": rec["val_acc"], "ema": rec.get("val_acc_ema", -1)}
            which = max(cand, key=cand.get)
            if cand[which] > best["acc"]:
                best = {"acc": cand[which], "epoch": epoch, "which": which}
                state = ema.shadow if (which == "ema" and ema is not None) else model.state_dict()
                torch.save({k: v.clone() for k, v in state.items()} if which == "ema" else state,
                           os.path.join(run_dir, "best.pt"))
        print(json.dumps(rec), flush=True)
        log_fh.write(json.dumps(rec) + "\n")
        log_fh.flush()
        if a.max_hours and (time.time() - t_start) / 3600 > a.max_hours:
            print(f"[{tag}] time budget reached at epoch {epoch}", flush=True)
            break

    log_fh.close()

    # ---- optional SWA checkpoint (kept only if it beats the best single epoch) ----
    ckpt = os.path.join(run_dir, "best.pt")
    if swa_count >= 2 and len(va_idx):
        swa = {k: (v / swa_count).to(device) for k, v in swa_sum.items()}
        model.load_state_dict(swa, strict=True)
        acc_swa = evaluate(va_idx, eval_batch)
        print(f"[{tag}] SWA({swa_count} snapshots) val={acc_swa:.4f} "
              f"vs best_single={best['acc']:.4f}", flush=True)
        if acc_swa > best["acc"]:
            best = {"acc": acc_swa, "epoch": -1, "which": "swa", "swa_count": swa_count,
                    "swa_epochs": sorted(swa_eps)}
            torch.save(swa, ckpt)

    # final artefacts: val probabilities from the best checkpoint
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    if len(va_idx):
        probs = torch.zeros(len(va_idx), ds.N_CLS)
        with torch.no_grad():
            for s in range(0, len(va_idx), eval_batch):
                sub = torch.from_numpy(np.asarray(va_idx[s : s + eval_batch])).to(device)
                px, y = dev.val_batch(sub)
                x, _, _, _ = evalaug(px, y)
                x = cl(x)
                with torch.amp.autocast("cuda"):
                    out = model(x)
                probs[s : s + len(sub)] = F.softmax(out.float(), dim=1).cpu()
        np.save(os.path.join(run_dir, "val_probs.npy"), probs.numpy())
        np.save(os.path.join(run_dir, "val_idx.npy"), np.asarray(va_idx))
        y_true = dev.train_y[torch.from_numpy(np.asarray(va_idx)).to(device)].cpu()
        final_acc = float((probs.argmax(1) == y_true).float().mean())
        best["final_val_acc"] = final_acc
    best["minutes"] = round((time.time() - t_start) / 60, 2)
    best["params_m"] = round(n_par / 1e6, 3)
    best["backbone"] = a.backbone
    best["res"] = a.res
    best["fold"] = a.fold
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(best, fh, indent=2)
    print(f"[{tag}] done in {best['minutes']}min  best={best}", flush=True)


if __name__ == "__main__":
    main()
