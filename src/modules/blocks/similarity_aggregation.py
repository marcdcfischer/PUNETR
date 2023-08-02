import torch
import torch.nn.functional as F
from src.utils.stable_softmax import stable_softmax


def similarity_aggregation(latents: torch.Tensor,
                           instructions: torch.Tensor,
                           mean_aggregation: bool = False,
                           top_k_selection: bool = False,
                           soft_selection_sigma: float = 0.1,
                           normalization: bool = True,
                           legacy: bool = True):
    """
    :param latents: [B, H*W*D, C]
    :param instructions: [B, I, N, C]
    :return: [B, I, H*W*D]
    """
    if normalization:
        latents = F.normalize(latents, p=2, dim=-1, eps=1e-12)
        instructions = F.normalize(instructions, p=2, dim=-1, eps=1e-12)

    if legacy:
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            x_sim = (torch.einsum('b m c, b i n c -> b i n m', latents, instructions) + 1.) / 2.  # Calculate similarities in range [0, 1] between instructions and content
    else:
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            x_sim = torch.einsum('b m c, b i n c -> b i n m', latents, instructions)  # Calculate similarities in range [-1, 1] between instructions and content

    # (Post) selection of instructions
    assert mean_aggregation is False or top_k_selection is False  # Both can't be true at once
    if mean_aggregation:
        x_sim = torch.mean(x_sim, dim=2)
    elif top_k_selection:
        # Top k selection with k=3. 1. Doesn't have to align to all, 2. Single outlier (max) is prevented due to top k averaging.
        x_sim = torch.topk(x_sim, k=3, dim=2)[0]
        x_sim = torch.mean(x_sim, dim=2)  # Average similarities of top k tokens of the respective mask
    else:
        # Re-weight by relative importance (detached softmaxed similarities). 1. Doesn't have to align to all, 2. All instructions receive a (weighted) gradient.
        x_sim = stable_softmax(x_sim.detach(), soft_selection_sigma, dim=2) * x_sim
        x_sim = torch.sum(x_sim, dim=2)  # Aggregate weighted similarities

    return x_sim
