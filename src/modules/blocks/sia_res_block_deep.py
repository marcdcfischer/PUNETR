import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.blocks.sia_block_deep import DeepShiftedInstructedAttentionBlock
from typing import Sequence, Optional, List, Union, Tuple
from monai.networks.blocks import Convolution
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from itertools import chain


class DeepSIAResBlock(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 instruction_pool_size: int,
                 tokens_per_instruction: int,
                 separate_background: bool = True,
                 kernel_size: Sequence[int] = (3, 3, 1),
                 strides: Sequence[int] = (1, 1, 1),
                 heads: int = 4,
                 window_size: Sequence[int] = (8, 8, 1),
                 norm_name: Union[Tuple, str] = "instance",
                 act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                 unique_instruction_bias: bool = True,
                 unique_token_bias: bool = True,
                 no_bias_instructions: bool = False,
                 no_bias_content: bool = False,
                 adapter: bool = False,
                 legacy_bias: bool = False,
                 half_res: bool = False,
                 pre_scale: bool = True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.half_res = half_res

        self.act = get_act_layer(name=act_name)
        self.sia = DeepShiftedInstructedAttentionBlock(hidden_channels=self.hidden_channels,
                                                       instruction_pool_size=instruction_pool_size,
                                                       tokens_per_instruction=tokens_per_instruction,
                                                       separate_background=separate_background,
                                                       heads=heads,
                                                       window_size=window_size,
                                                       unique_instruction_bias=unique_instruction_bias,
                                                       unique_token_bias=unique_token_bias,
                                                       no_bias_instructions=no_bias_instructions,
                                                       no_bias_content=no_bias_content,
                                                       adapter=adapter,
                                                       legacy_bias=legacy_bias,
                                                       pre_scale=pre_scale)
        self.norm_1 = get_norm_layer(name=norm_name, spatial_dims=3, channels=self.hidden_channels)

        if self.half_res:
            self.down = nn.Upsample(scale_factor=(0.5, 0.5, 0.5), mode='trilinear')
            self.up = nn.Upsample(scale_factor=(2.0, 2.0, 2.0), mode='trilinear')

    def forward(self,
                x: torch.Tensor,
                x_instructions: Optional[Sequence[torch.Tensor]] = None,
                label_indices: Optional[torch.Tensor] = None):
        residual = x

        # SIA
        if self.half_res:
            x = self.down(x)
        x = self.norm_1(self.sia(x, x_instructions, label_indices=label_indices))  # Normed output since UnetrUpBlock directly starts with a conv. Also more similar to UnetResBlock
        if self.half_res:
            x = self.up(x)

        # Residuals
        x = self.act(x + residual)  # adjustments to residual are only down in case of in_channels != out_channels in UnetResBlock

        return x

    def named_parameters_body(self):

        parameters_res = self.norm_1.named_parameters()
        parameters_sia_att = self.sia.named_parameters_attention()
        parameters_bias_content = self.sia.named_parameters_bias_content()

        return list(chain(*[parameters_res, parameters_sia_att, parameters_bias_content]))

    def named_parameters_adapter(self):

        return self.sia.named_parameters_adapter()

    def named_parameters_bias_instructions(self):

        parameters_bias_instructions = list(self.sia.named_parameters_bias_instructions())

        return parameters_bias_instructions
