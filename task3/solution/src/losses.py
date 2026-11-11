"""Loss functions for landmark heatmaps."""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveWingLoss(nn.Module):
    """Adaptive Wing Loss (Wang et al., ICCV 2019).

    Reference implementation follows MMPose's heatmap_loss.py.
    """

    def __init__(self, omega: float = 14.0, theta: float = 0.5,
                 epsilon: float = 1.0, alpha: float = 2.1):
        super().__init__()
        self.omega = float(omega)
        self.theta = float(theta)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = (target - pred).abs()
        A = self.omega * (
            1.0 / (1.0 + (self.theta / self.epsilon) ** (self.alpha - target))
        ) * (self.alpha - target) * (
            (self.theta / self.epsilon) ** (self.alpha - target - 1.0)
        ) / self.epsilon
        C = self.theta * A - self.omega * torch.log(
            1.0 + (self.theta / self.epsilon) ** (self.alpha - target))
        losses = torch.where(
            delta < self.theta,
            self.omega * torch.log(1.0 + (delta / self.epsilon) ** (self.alpha - target)),
            A * delta - C,
        )
        return losses.mean()
