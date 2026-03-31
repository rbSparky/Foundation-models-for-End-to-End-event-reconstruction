from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConservationLoss(nn.Module):
    """Baseline masked reconstruction loss on [pT, eta, phi, E]."""

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 1.0,
        gamma: float = 0.5,
        loss_coef: List[float] = [0.25, 0.25, 0.25, 0.25],
        reduction: str = "mean",
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.loss_coef = loss_coef
        self.reduction = reduction

    def _pT_loss(self, pT_pred: Tensor, pT_true: Tensor) -> Tensor:
        return torch.sqrt(F.mse_loss(pT_pred, pT_true, reduction=self.reduction))

    def _eta_loss(self, eta_pred: Tensor, eta_true: Tensor) -> Tensor:
        return F.l1_loss(eta_pred, eta_true, reduction=self.reduction)

    def _phi_loss(self, phi_pred: Tensor, phi_true: Tensor) -> Tensor:
        sin_pred, cos_pred = torch.sin(phi_pred), torch.cos(phi_pred)
        sin_true, cos_true = torch.sin(phi_true), torch.cos(phi_true)
        cos_sim = cos_true * cos_pred + sin_true * sin_pred
        return (1.0 - cos_sim).mean()

    def _energy_loss(self, E_pred: Tensor, E_true: Tensor) -> Tensor:
        return torch.sqrt(F.mse_loss(E_pred, E_true, reduction=self.reduction))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pT_pred, eta_pred, phi_pred, E_pred = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        pT_true, eta_true, phi_true, E_true = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        pT_loss = self._pT_loss(pT_pred, pT_true)
        eta_loss = self._eta_loss(eta_pred, eta_true)
        phi_loss = self._phi_loss(phi_pred, phi_true)
        energy_loss = self._energy_loss(E_pred, E_true)

        loss = 0.0
        loss += self.loss_coef[0] * pT_loss
        loss += self.loss_coef[1] * eta_loss
        loss += self.loss_coef[2] * phi_loss
        loss += self.loss_coef[3] * energy_loss

        return loss, (pT_loss, eta_loss, phi_loss, energy_loss)


class StableReconstructionLoss(nn.Module):
    """Numerically-stable masked reconstruction loss on [log1p(pT), eta, sin(phi), cos(phi), log1p(E)]."""

    def __init__(
        self,
        loss_coef: List[float] = [0.3, 0.2, 0.2, 0.3],
        robust: str = "huber",
        huber_delta: float = 0.1,
        charbonnier_eps: float = 1e-3,
        reduction: str = "mean",
    ):
        super().__init__()
        self.loss_coef = loss_coef
        self.robust = robust
        self.huber_delta = huber_delta
        self.charbonnier_eps = charbonnier_eps
        self.reduction = reduction

    def _robust(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.robust == "huber":
            return F.huber_loss(pred, target, reduction=self.reduction, delta=self.huber_delta)
        if self.robust == "charbonnier":
            diff = pred - target
            return torch.sqrt(diff * diff + self.charbonnier_eps ** 2).mean()
        raise ValueError(f"Unknown robust loss: {self.robust}")

    def forward(self, pred: Tensor, target: Tensor):
        if pred.size(-1) != 5 or target.size(-1) != 5:
            raise ValueError(
                "StableReconstructionLoss expects prediction/target shape (*, 5): "
                "[log1p(pT), eta, sin(phi), cos(phi), log1p(E)]"
            )

        p_t_loss = self._robust(pred[:, 0], target[:, 0])
        eta_loss = F.l1_loss(pred[:, 1], target[:, 1], reduction=self.reduction)
        phi_loss = 0.5 * (
            F.mse_loss(pred[:, 2], target[:, 2], reduction=self.reduction)
            + F.mse_loss(pred[:, 3], target[:, 3], reduction=self.reduction)
        )
        energy_loss = self._robust(pred[:, 4], target[:, 4])

        loss = 0.0
        loss += self.loss_coef[0] * p_t_loss
        loss += self.loss_coef[1] * eta_loss
        loss += self.loss_coef[2] * phi_loss
        loss += self.loss_coef[3] * energy_loss

        return loss, (p_t_loss, eta_loss, phi_loss, energy_loss)
