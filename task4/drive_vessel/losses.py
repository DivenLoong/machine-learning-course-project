from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    probabilities = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probabilities * targets).sum(dims)
    denominator = probabilities.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce + soft_dice_loss(logits, targets)


def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
):
    probabilities = torch.sigmoid(logits)
    dims = (1, 2, 3)
    true_positive = (probabilities * targets).sum(dims)
    false_positive = (probabilities * (1.0 - targets)).sum(dims)
    false_negative = ((1.0 - probabilities) * targets).sum(dims)
    score = (true_positive + eps) / (
        true_positive + alpha * false_positive + beta * false_negative + eps
    )
    return 1.0 - score.mean()


def tversky_bce_loss(logits: torch.Tensor, targets: torch.Tensor):
    return F.binary_cross_entropy_with_logits(logits, targets) + tversky_loss(
        logits, targets
    )


def build_loss(name: str):
    if name == "bce_dice":
        return bce_dice_loss
    if name == "tversky_bce":
        return tversky_bce_loss
    raise ValueError(f"Unknown loss: {name}")
