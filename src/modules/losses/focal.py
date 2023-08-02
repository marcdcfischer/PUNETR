import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FocalLoss(nn.Module):
    def __init__(self,
                 out_channels: int,
                 loss_weight: float = 1.,
                 alpha_background: float = 0.1,
                 alpha_background_end: float = 0.1,
                 alpha_foreground: float = 0.1,
                 additive_alpha: Tuple[float, ...] = (0.0, 0.9),
                 gamma: float = 1.5,
                 normalized: bool = True,
                 alpha_blending: bool = False):
        super().__init__()
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.out_channels = out_channels
        self.alpha_background = alpha_background
        self.alpha_background_end = alpha_background_end
        self.alpha_foreground = alpha_foreground
        self.additive_alpha = additive_alpha
        self.normalized = normalized
        self.alpha_blending = alpha_blending

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                label_indices_active: Optional[torch.Tensor] = None,
                current_epoch: float | None = None,
                max_epochs: float | None = None,
                tag: str = 'seg_focal'):
        """
        :param predictions: logits [B, C, H, W, D]
        :param targets: int tensor [B, H, W, D]
        :param label_indices_active: [B, C]
        :return:
        """
        assert predictions[:, 0, ...].shape == targets.shape

        losses = dict()
        log_softmax = torch.clamp(F.log_softmax(predictions, dim=1), min=-1e3)
        log_prob_weighted, log_prob_nonweighted = list(), list()
        current_alpha_background_ = self.alpha_background
        if self.alpha_blending:  # Progressively move from initial alpha background to alpha background end value based on epoch
            current_alpha_background_ = (1. - (current_epoch / max_epochs)) * self.alpha_background + (current_epoch / max_epochs) * self.alpha_background_end
        for idx_batch in range(targets.shape[0]):
            loss_weight_alpha = torch.tensor([current_alpha_background_] + [self.alpha_foreground for _ in range(predictions.shape[1] - 1)], dtype=torch.float16, device=predictions.device)
            loss_weight_alpha += torch.tensor(self.additive_alpha, dtype=torch.float16, device=predictions.device)[torch.nonzero(label_indices_active[idx_batch, :], as_tuple=False).squeeze()] if label_indices_active is not None else torch.tensor(self.additive_alpha, dtype=torch.float16, device=predictions.device)
            assert predictions.shape[1] == loss_weight_alpha.shape[0]
            log_prob_weighted.append(F.nll_loss(input=log_softmax[idx_batch: idx_batch+1, ...],
                                                target=targets[idx_batch: idx_batch+1, ...],
                                                weight=loss_weight_alpha,
                                                reduction='none')[0, ...])
            log_prob_nonweighted.append(F.nll_loss(input=log_softmax[idx_batch: idx_batch+1, ...],
                                                   target=targets[idx_batch: idx_batch+1, ...],
                                                   weight=None,
                                                   reduction='none')[0, ...])
        log_prob_weighted, log_prob_nonweighted = torch.stack(log_prob_weighted, dim=0), torch.stack(log_prob_nonweighted)
        prob = torch.exp(-log_prob_nonweighted)
        one_minus_prob = torch.clamp(1.0 - prob, min=0.0, max=1.0)**self.gamma

        normalization = 1.0
        if self.normalized:
            with torch.no_grad():
                normalization = torch.clamp(one_minus_prob.detach().mean(dim=(1, 2, 3)), min=1e-3, max=1.0)  # (clamped) focal loss normalization
        losses[tag] = self.loss_weight * ((1.0 / normalization) * (one_minus_prob * log_prob_weighted).mean(dim=(1, 2, 3))).mean()

        return losses
