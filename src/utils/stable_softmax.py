import torch
from typing import Sequence


def stable_softmax(logits: torch.Tensor, temperature: float, dim: int | Sequence[int]):
    exp = torch.exp((logits - logits.detach().max(dim=dim, keepdim=True).values) / temperature)
    return exp / exp.sum(dim=dim, keepdim=True)
