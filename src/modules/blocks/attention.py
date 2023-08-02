import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
import einops
from torch.utils.checkpoint import checkpoint
from torch.backends.cuda import sdp_kernel, SDPBackend
# Helpful arg mapper
backend_map = {
    SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
    SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
}

class WindowedMaskedAttention(nn.Module):
    def __init__(self,
                 q_channels: int,
                 kv_channels: Optional[int] = None,
                 heads_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 separate_norms: bool = False,
                 qkv_bias: bool = True,
                 heads: int = 4,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 use_checkpoint: bool = False,
                 use_fused_sdpa: bool = True):
        super().__init__()
        kv_channels = kv_channels if kv_channels is not None else q_channels
        heads_channels = heads_channels if heads_channels is not None else q_channels // heads
        out_channels = out_channels if out_channels is not None else q_channels
        inner_channels = q_channels
        assert inner_channels % heads == 0

        self.use_checkpoint = use_checkpoint
        self.use_fused_sdpa = use_fused_sdpa
        self.heads = heads
        self.attn_drop_flag = attn_drop
        self.proj_drop_flag = proj_drop
        self.eps = 1e-8

        self.pre_norm_q, self.pre_norm_k, self.pre_norm_v = None, None, None
        if separate_norms:
            self.pre_norm_q = nn.LayerNorm(q_channels)
            self.pre_norm_k = nn.LayerNorm(kv_channels)
            self.pre_norm_v = nn.LayerNorm(kv_channels)

        self.qkv = nn.Linear(q_channels, 3 * inner_channels, bias=qkv_bias)
        # self.to_k = nn.Linear(kv_channels, inner_channels, bias=qkv_bias)
        # self.to_v = nn.Linear(kv_channels, inner_channels, bias=qkv_bias)
        self.masked_attention_calc = WindowedMaskedAttentionCalculation(inv_temperature=heads_channels ** -0.5)
        self.proj = nn.Linear(inner_channels, out_channels)
        self.proj_drop = nn.Dropout(self.proj_drop_flag)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                pos_q: Optional[torch.Tensor] = None,
                pos_k: Optional[torch.Tensor] = None,
                pos_scores: Optional[torch.Tensor] = None):
        # Transform to q, k, v and add pos info
        n_batch, n_patch = x.shape[0], x.shape[1]
        qkv = einops.rearrange(self.qkv(x), 'b p n (v h d) -> v (b p) h n d', v=3, h=self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Add pos
        q = q + pos_q if pos_q is not None else q
        k = k + pos_k if pos_k is not None else k

        # Masked Attention
        pos_scores = einops.rearrange(pos_scores.expand(n_batch, n_patch, self.heads, -1, -1), 'b p h n d -> (b p) h n d')
        if self.use_fused_sdpa:
            if mask is not None:
                raise NotImplementedError  # Shouldn't be the case anymore, since mask is currently fused with pos_scores before being passed to this function
            else:
                # with sdp_kernel(**backend_map[SDPBackend.MATH]):  # Note: new kernels do not work with attention bias yet. Especially the gradient on the att. bias could remain problematic.
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=pos_scores, dropout_p=self.attn_drop_flag, is_causal=False)  # uses std. scale
        elif self.use_checkpoint:
            out = checkpoint(self.masked_attention_calc, q, k, v, pos_scores, mask, preserve_rng_state=False, use_reentrant=False)
        else:
            out = self.masked_attention_calc(q, k, v, pos_scores, mask)
        out = einops.rearrange(out, '(b p) h n d -> b p n (h d)', b=n_batch, p=n_patch, h=self.heads)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class WindowedMaskedAttentionCalculation(nn.Module):
    def __init__(self,
                 inv_temperature: float):
        super().__init__()
        self.inv_temperature = inv_temperature

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos_scores: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        sim *= self.inv_temperature     # Scaled prior to pos_scores (since they are already pre-scaled atm)
        sim = sim + pos_scores if pos_scores is not None else sim
        sim = mask * sim if mask is not None else sim
        attn_score = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn_score, v)

        return out
