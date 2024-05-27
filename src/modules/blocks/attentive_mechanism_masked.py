import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from src.modules.blocks.attention import WindowedMaskedAttention
from torch.utils.checkpoint import checkpoint
from monai.networks.blocks import MLPBlock as Mlp
import einops


# see https://github.com/KMnP/vpt/blob/e2dd70a5ee291d398d002e6963ddbe0f66f58038/src/models/vit_adapter/adapter_block.py#L25 for adapter adaptation
class WindowedMaskedAttentionBlock(nn.Module):
    """
    Generic attention block for masked attention
    """
    def __init__(self,
                 hidden_channels: int,
                 heads: int = 4,
                 mlp_ratio: float = 4.0,
                 act_layer: str = "GELU",
                 dropout_mlp: float = 0.1,  # atm hardcoded
                 dropout_proj: float = 0.1,  # atm hardcoded
                 dropout_attn: float = 0.1,  # atm hardcoded
                 reduction_factor: int = 4,  # atm hardcoded
                 qkv_bias: bool = True,
                 adapter: bool = False):
        super().__init__()
        self.mlp_ratio = mlp_ratio
        self.adapter = adapter

        self.norm1 = nn.GroupNorm(hidden_channels, hidden_channels)
        self.attn = WindowedMaskedAttention(q_channels=hidden_channels,
                                            heads=heads,
                                            attn_drop=dropout_attn,
                                            proj_drop=dropout_proj,
                                            separate_norms=False,
                                            qkv_bias=qkv_bias)
        self.norm2 = nn.GroupNorm(hidden_channels, hidden_channels)
        self.mlp = Mlp(hidden_size=hidden_channels,
                       mlp_dim=int(hidden_channels * self.mlp_ratio),
                       act=act_layer,
                       dropout_rate=dropout_mlp,
                       dropout_mode="swin")

        if self.adapter:
            self.adapter_block = Mlp(hidden_size=hidden_channels,
                                     mlp_dim=int(hidden_channels // reduction_factor),
                                     act=act_layer,
                                     dropout_rate=dropout_mlp,
                                     dropout_mode="swin")
            self.adapter_norm = nn.GroupNorm(hidden_channels, hidden_channels)

    def forward(self,
                x: torch.Tensor,
                x_instructions: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                pos_scores: Optional[torch.Tensor] = None):
        """

        :param x: Key / value content [B, P, N, C]
        :return:
        """
        b_ = x.size(0)

        # Attention block
        x_att = einops.rearrange(self.norm1(einops.rearrange(x, 'b p n c -> (b p) c n')), '(b p) c n -> b p n c', b=b_)
        x_att = torch.cat([x_instructions, x_att], dim=2) if x_instructions is not None else x_att  # Concat instructions
        x_att = self.attn(x=x_att, mask=mask, pos_scores=pos_scores)
        x_att = x_att[:, :, x_instructions.shape[2]:, :] if x_instructions is not None else x_att
        x = x + x_att

        # Residual and MLP
        x_mlp = self.mlp(einops.rearrange(self.norm2(einops.rearrange(x, 'b p n c -> (b p) c n')), '(b p) c n -> b p n c', b=b_))
        x = x + x_mlp

        # Adapter - with pre- and post- residual connection
        if self.adapter:
            x = x + self.adapter_block(einops.rearrange(self.adapter_norm(einops.rearrange(x, 'b p n c -> (b p) c n')), '(b p) c n -> b p n c', b=b_))

        return x

    def load_from(self, weights, n_block, prefix, layer):
        root = f"{prefix}.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",

        ]
        if prefix == 'module':
            block_names += [
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
            ]
        elif prefix == 'swinViT':
            block_names += [
            "mlp.linear1.weight",
            "mlp.linear1.bias",
            "mlp.linear2.weight",
            "mlp.linear2.bias",
            ]
        else:
            raise ValueError()

        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            # self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])  # Done in outer function
            # self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])  # Done in outer function
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])

    def named_parameters_body(self):

        return [x_ for x_ in list(self.named_parameters()) if any(y_ in x_[0] for y_ in ['norm1', 'attn', 'norm2', 'mlp'])]

    def named_parameters_adapter(self):

        params_ = list()
        if self.adapter:
            params_ += [x_ for x_ in list(self.named_parameters()) if any(y_ in x_[0] for y_ in ['adapter'])]

        return params_

