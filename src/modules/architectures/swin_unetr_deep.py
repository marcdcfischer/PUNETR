import einops
import torch
import torch.nn as nn
from argparse import ArgumentParser, Namespace
from typing import Union, Dict, Optional, List
from monai.networks.blocks import UnetrBasicBlock, UnetOutBlock
from src.modules.blocks.monai_unetr_block_custom import UnetrUpBlockCustom
from src.modules.architectures.monai_swin_unetr_custom import SwinTransformer
from src.modules.blocks.sia_res_block_deep import DeepSIAResBlock
from src.modules.blocks.instruction_pool import InstructionPool
from src.modules.blocks.similarity_aggregation import similarity_aggregation
from src.modules.blocks.sia_block_deep import DeepShiftedInstructedAttentionBlock
import collections
import warnings
from itertools import chain


class DeepSIAUNetr(nn.Module):
    def __init__(self,
                 conf: Union[Dict, Namespace]):
        super().__init__()
        self.conf = conf

        # Segmentation instructions (atm one token per class)
        assert self.conf.instruction_pool_size >= self.conf.out_channels
        self.separate_background = self.conf.separate_background and self.conf.label_indices_max_active < 2
        if self.separate_background:
            print('Using separate background tokens for each foreground class.')
        else:
            print('Using shared background tokens for all foreground classes.')
        self.tokens_per_instruction_combined = 2 * self.conf.tokens_per_instruction if self.separate_background else self.conf.tokens_per_instruction
        self.tokens_per_instruction_seg_combined = 2 * self.conf.tokens_per_instruction_seg if self.separate_background else self.conf.tokens_per_instruction_seg

        # Make sure variants are configured (somewhat) properly
        assert self.conf.instruction_channels == self.conf.hidden_channels[0]
        if 'prompting' in self.conf.adaptation_variant.lower():
            assert self.conf.fixed_output is False
        else:
            assert self.conf.fixed_output is True

        # Make sure dimensions are divisible at least by 2 or are equal to 1
        for patch_size_ in ([self.conf.patch_size_teacher, *self.conf.patch_size_students]):
            assert all([x_ % 2 == 0 or x_ == 1 for x_ in patch_size_])

        # Architecture
        # UNet Encoder (with SIA blocks)
        self.depth_kernel_size_unet = [3 if self.conf.patch_size_students[0][2] / 2 ** idx_block >= 3. else 1 for idx_block in range(6)]
        self.depth_stride_unet = [2 if self.conf.patch_size_students[0][2] / 2 ** idx_block > 1. else 1 for idx_block in range(6)]
        # self.depth_scale_factor_unet = [2. if self.conf.patch_size_students[0][2] / 2 ** idx_block > 1. else 1 for idx_block in range(self.conf.depth_sia_encoder)]
        print(f'Using kernel sizes {(3, 3, self.depth_kernel_size_unet)}, strides {(2, 2, self.depth_stride_unet)}.')

        # Encoder
        self.sia_instruction_blocks_encoder = nn.ModuleList()
        if self.conf.monai_swin_vit:
            self.swinunetr_encoder_vit = SwinTransformer(
                in_chans=self.conf.in_channels,
                embed_dim=self.conf.hidden_channels[0],
                window_size=self.conf.monai_attn_window_size,
                patch_size=(2, 2, 2 if self.conf.patch_size_students[0][2] > 1 else 1),
                depths=self.conf.monai_swin_vit_depths,
                num_heads=self.conf.attention_heads[1:],
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=self.conf.monai_drop_rate,
                attn_drop_rate=self.conf.monai_attn_drop_rate,
                drop_path_rate=self.conf.monai_dropout_path_rate,
                norm_layer=nn.LayerNorm,  # hardcoded in their SwinUNETR implementation
                use_checkpoint=self.conf.monai_use_checkpoint,
                spatial_dims=3,
                downsample="mergingv2",
                use_promptable_attention=self.conf.monai_use_promptable_attention,
                instruction_pool_size=self.conf.instruction_pool_size,
                tokens_per_instruction=self.tokens_per_instruction_combined,
                separate_background=self.separate_background,
                unique_instruction_bias=self.conf.unique_instruction_bias,
                unique_token_bias=self.conf.unique_token_bias,
                no_bias_instructions=self.conf.no_bias_instructions,
                no_bias_content=self.conf.no_bias_content,
                adapter=self.conf.adaptation_variant.lower() == 'adapter',
                legacy_bias=self.conf.monai_legacy_bias,
                pre_scale=self.conf.pre_scale,
            )
            for idx_block in range(4):
                self.sia_instruction_blocks_encoder.extend([nn.Module(), nn.Module()])
                if self.conf.prompting_variant.lower() in ['encoder', 'full'] and not self.conf.fixed_output:
                    for idx_sub_block_ in range(2):  # windowed and shifted windowed block
                        self.sia_instruction_blocks_encoder[2 * idx_block + idx_sub_block_] = InstructionPool(
                            instruction_pool_size=self.conf.instruction_pool_size,
                            hidden_channels=self.conf.hidden_channels[0] * 2**idx_block,
                            default_instructions=self.conf.out_channels,
                            tokens_per_instruction=self.tokens_per_instruction_combined,
                            separate_background=self.separate_background,
                            use_norm=self.conf.instructions_use_norm,
                            elementwise_affine=self.conf.instructions_elementwise_affine,
                            dropout=self.conf.instruction_dropout,
                        )

        # Skip connections
        self.swinunetr_skip_blocks = nn.ModuleList()
        self.sia_instruction_blocks_decoder = nn.ModuleList()
        for idx_block in range(5):
            self.sia_instruction_blocks_decoder.extend([nn.Module(), nn.Module()])
            if self.conf.model_variant.lower() == 'swin_unetr':
                self.swinunetr_skip_blocks.append(UnetrBasicBlock(
                    spatial_dims=3,
                    in_channels=self.conf.hidden_channels[0] * 2**(idx_block-1) if idx_block > 0 else self.conf.in_channels,
                    out_channels=self.conf.hidden_channels[0] * 2**(idx_block-1) if idx_block > 0 else self.conf.hidden_channels[0],
                    kernel_size=(3, 3, self.depth_kernel_size_unet[idx_block]),
                    stride=1,
                    norm_name=self.conf.monai_conv_norm_name,
                    res_block=True,
                ))
            elif self.conf.model_variant.lower() == 'punet':
                if idx_block == 0:  # extra residual block, since DeepSIAResBlock would need to change amount of channels within attention block otherwise ...
                    self.swinunetr_skip_blocks.append(UnetrBasicBlock(
                        spatial_dims=3,
                        in_channels=self.conf.hidden_channels[0] * 2 ** (idx_block - 1) if idx_block > 0 else self.conf.in_channels,
                        out_channels=self.conf.hidden_channels[0] * 2 ** (idx_block - 1) if idx_block > 0 else self.conf.hidden_channels[0],
                        kernel_size=(3, 3, self.depth_kernel_size_unet[idx_block]),
                        stride=1,
                        norm_name=self.conf.monai_conv_norm_name,
                        res_block=True,
                    ))
                else:
                    self.swinunetr_skip_blocks.append(DeepSIAResBlock(
                        hidden_channels=self.conf.hidden_channels[0] * 2 ** (idx_block - 1) if idx_block > 0 else self.conf.hidden_channels[0],
                        instruction_pool_size=self.conf.instruction_pool_size,
                        tokens_per_instruction=self.tokens_per_instruction_combined,
                        separate_background=self.separate_background,
                        kernel_size=(3, 3, self.depth_kernel_size_unet[idx_block]),
                        strides=(1, 1, 1),
                        heads=self.conf.attention_heads[idx_block],
                        window_size=self.conf.attn_window_size,
                        norm_name=self.conf.monai_conv_norm_name,
                        unique_instruction_bias=self.conf.unique_instruction_bias,
                        unique_token_bias=self.conf.unique_token_bias,
                        no_bias_instructions=self.conf.no_bias_instructions,
                        no_bias_content=self.conf.no_bias_content,
                        adapter=self.conf.adaptation_variant.lower() == 'adapter',
                        pre_scale=self.conf.pre_scale,
                    ))
                if self.conf.prompting_variant.lower() in ['decoder', 'full'] and not self.conf.fixed_output:
                    for idx_sub_block_ in range(2):  # windowed and shifted windowed block
                        self.sia_instruction_blocks_decoder[2 * idx_block + idx_sub_block_] = InstructionPool(
                            instruction_pool_size=self.conf.instruction_pool_size,
                            hidden_channels=self.conf.hidden_channels[0] * 2**(idx_block-1) if idx_block > 0 else self.conf.hidden_channels[0],
                            default_instructions=self.conf.out_channels,
                            tokens_per_instruction=self.tokens_per_instruction_combined,
                            separate_background=self.separate_background,
                            use_norm=self.conf.instructions_use_norm,
                            elementwise_affine=self.conf.instructions_elementwise_affine,
                            dropout=self.conf.instruction_dropout,
                        )
            elif any([self.conf.model_variant.lower() == x_ for x_ in ['punet_decoder', 'punet_decoder_high_res']]):
                if idx_block == 0:
                    self.swinunetr_skip_blocks.append(UnetrBasicBlock(
                        spatial_dims=3,
                        in_channels=self.conf.hidden_channels[0] * 2 ** (idx_block - 1) if idx_block > 0 else self.conf.in_channels,
                        out_channels=self.conf.hidden_channels[0] * 2 ** (idx_block - 1) if idx_block > 0 else self.conf.hidden_channels[0],
                        kernel_size=(3, 3, self.depth_kernel_size_unet[idx_block]),
                        stride=1,
                        norm_name=self.conf.monai_conv_norm_name,
                        res_block=True,
                    ))
                else:
                    self.swinunetr_skip_blocks.append(nn.Identity())
            elif self.conf.model_variant.lower() == 'punet_decoder_low_res':
                self.swinunetr_skip_blocks.append(nn.Identity())
            else:
                raise NotImplementedError

        # SwinUNETR Decoder
        self.swinunetr_decoder_blocks = nn.ModuleList()
        self.swinunetr_decoder_blocks_late = nn.ModuleList()
        self.sia_instruction_blocks_decoder_late = nn.ModuleList()
        for idx_block in range(5):
            self.sia_instruction_blocks_decoder_late.extend([nn.Module(), nn.Module()])
            self.swinunetr_decoder_blocks.append(UnetrUpBlockCustom(
                spatial_dims=3,
                in_channels=self.conf.hidden_channels[0] * 2**idx_block if idx_block > 0 else self.conf.hidden_channels[0],
                out_channels=self.conf.hidden_channels[0] * 2**(idx_block - 1) if idx_block > 0 else self.conf.hidden_channels[0],
                kernel_size=(3, 3, self.depth_kernel_size_unet[idx_block]),
                upsample_kernel_size=(2, 2, self.depth_stride_unet[idx_block]),
                norm_name=self.conf.monai_conv_norm_name,
                res_block=True,
                skip=False if idx_block == 0 and self.conf.model_variant.lower() == 'punet_decoder_low_res' else True,
            ))
            if any([self.conf.model_variant.lower() == x_ for x_ in ['punet_decoder', 'punet_decoder_low_res', 'punet_decoder_high_res']]):
                if idx_block == 0 and not self.conf.model_variant.lower() == 'punet_decoder_high_res':
                    self.swinunetr_decoder_blocks_late.append(nn.Identity())
                else:
                    self.swinunetr_decoder_blocks_late.append(DeepSIAResBlock(
                        hidden_channels=self.conf.hidden_channels[0] * 2 ** (idx_block - 1) if idx_block > 0 else self.conf.hidden_channels[0],
                        instruction_pool_size=self.conf.instruction_pool_size,
                        tokens_per_instruction=self.tokens_per_instruction_combined,
                        separate_background=self.separate_background,
                        kernel_size=(3, 3, self.depth_kernel_size_unet[idx_block]),
                        strides=(1, 1, 1),
                        heads=self.conf.attention_heads[idx_block],
                        window_size=self.conf.attn_window_size,
                        norm_name=self.conf.monai_conv_norm_name,
                        unique_instruction_bias=self.conf.unique_instruction_bias,
                        unique_token_bias=self.conf.unique_token_bias,
                        no_bias_instructions=self.conf.no_bias_instructions,
                        no_bias_content=self.conf.no_bias_content,
                        adapter=self.conf.adaptation_variant.lower() == 'adapter',
                        half_res=True if idx_block == 0 and self.conf.model_variant.lower() == 'punet_decoder_high_res' else False,
                        pre_scale=self.conf.pre_scale,
                    ))
                if self.conf.prompting_variant.lower() in ['decoder', 'full'] and not self.conf.fixed_output:
                    for idx_sub_block_ in range(2):  # windowed and shifted windowed block
                        self.sia_instruction_blocks_decoder_late[2 * idx_block + idx_sub_block_] = InstructionPool(
                            instruction_pool_size=self.conf.instruction_pool_size,
                            hidden_channels=self.conf.hidden_channels[0] * 2 ** (idx_block - 1) if idx_block > 0 else self.conf.hidden_channels[0],
                            default_instructions=self.conf.out_channels,
                            tokens_per_instruction=self.tokens_per_instruction_combined,
                            separate_background=self.separate_background,
                            use_norm=self.conf.instructions_use_norm,
                            elementwise_affine=self.conf.instructions_elementwise_affine,
                            dropout=self.conf.instruction_dropout,
                        )

        # Last pool is always active (otherwise no sim comparison is possible)
        # I.e. this pool exists regardless of start, end, encoder, decoder, full variants
        if not self.conf.fixed_output:
            # With add. out block (so content is not normed or limited by an activation func.)
            if self.conf.final_conv:
                self.conv_out = UnetOutBlock(spatial_dims=3,
                                             in_channels=self.conf.hidden_channels[0],
                                             out_channels=self.conf.hidden_channels[0])
            self.instruction_pool = InstructionPool(instruction_pool_size=self.conf.instruction_pool_size,
                                                    hidden_channels=self.conf.instruction_channels,
                                                    default_instructions=self.conf.out_channels,
                                                    tokens_per_instruction=self.tokens_per_instruction_seg_combined,
                                                    separate_background=self.separate_background,
                                                    use_norm=self.conf.instructions_use_norm_final,
                                                    elementwise_affine=self.conf.instructions_elementwise_affine,
                                                    dropout=self.conf.instruction_dropout)
        # Fixed output (ablation)
        else:
            self.conv_fixed = UnetOutBlock(spatial_dims=3,
                                           in_channels=self.conf.hidden_channels[0],
                                           out_channels=self.conf.label_indices_max_active + 1 if self.conf.label_indices_max_active > 0 else self.conf.out_channels)

        # Mean rep initialization
        if self.conf.mean_initialization and self.conf.downstream:
            self.set_downstream_instruction_parameters(label_indices_base=self.conf.label_indices_base,
                                                       label_indices_downstream=self.conf.label_indices_downstream_active)

    def load_swin_vit(self, weights):
        with torch.no_grad():
            print('Using pre-trained SWinViT encoder.')
            print(f'Available weights: {[weights["state_dict"].keys()]}.')
            prefix = "module"

            self.swinunetr_encoder_vit.patch_embed.proj.weight.copy_(weights["state_dict"][f"{prefix}.patch_embed.proj.weight"])
            self.swinunetr_encoder_vit.patch_embed.proj.bias.copy_(weights["state_dict"][f"{prefix}.patch_embed.proj.bias"])
            if self.conf.monai_use_promptable_attention:
                self.swinunetr_encoder_vit.layers1[0].load_from(weights, n_block=0, prefix=prefix, layer="layers1")
                self.swinunetr_encoder_vit.layers2[0].load_from(weights, n_block=0, prefix=prefix, layer="layers2")
                self.swinunetr_encoder_vit.layers3[0].load_from(weights, n_block=0, prefix=prefix, layer="layers3")
                self.swinunetr_encoder_vit.layers4[0].load_from(weights, n_block=0, prefix=prefix, layer="layers4")
            else:
                for bname, block in self.swinunetr_encoder_vit.layers1[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, prefix=prefix, layer="layers1")
                self.swinunetr_encoder_vit.layers1[0].downsample.reduction.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers1.0.downsample.reduction.weight"]
                )
                self.swinunetr_encoder_vit.layers1[0].downsample.norm.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers1.0.downsample.norm.weight"]
                )
                self.swinunetr_encoder_vit.layers1[0].downsample.norm.bias.copy_(
                    weights["state_dict"][f"{prefix}.layers1.0.downsample.norm.bias"]
                )
                for bname, block in self.swinunetr_encoder_vit.layers2[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, prefix=prefix, layer="layers2")
                self.swinunetr_encoder_vit.layers2[0].downsample.reduction.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers2.0.downsample.reduction.weight"]
                )
                self.swinunetr_encoder_vit.layers2[0].downsample.norm.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers2.0.downsample.norm.weight"]
                )
                self.swinunetr_encoder_vit.layers2[0].downsample.norm.bias.copy_(
                    weights["state_dict"][f"{prefix}.layers2.0.downsample.norm.bias"]
                )
                for bname, block in self.swinunetr_encoder_vit.layers3[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, prefix=prefix, layer="layers3")
                self.swinunetr_encoder_vit.layers3[0].downsample.reduction.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers3.0.downsample.reduction.weight"]
                )
                self.swinunetr_encoder_vit.layers3[0].downsample.norm.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers3.0.downsample.norm.weight"]
                )
                self.swinunetr_encoder_vit.layers3[0].downsample.norm.bias.copy_(
                    weights["state_dict"][f"{prefix}.layers3.0.downsample.norm.bias"]
                )
                for bname, block in self.swinunetr_encoder_vit.layers4[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, prefix=prefix, layer="layers4")
                self.swinunetr_encoder_vit.layers4[0].downsample.reduction.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers4.0.downsample.reduction.weight"]
                )
                self.swinunetr_encoder_vit.layers4[0].downsample.norm.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers4.0.downsample.norm.weight"]
                )
                self.swinunetr_encoder_vit.layers4[0].downsample.norm.bias.copy_(
                    weights["state_dict"][f"{prefix}.layers4.0.downsample.norm.bias"]
                )

    def load_swin_unetr(self, weights):
        with torch.no_grad():
            print('Using pre-trained SWinUNETR model.')
            print(f'Available weights: {[weights["state_dict"].keys()]}.')
            prefix = "swinViT"

            # Encoder
            self.swinunetr_encoder_vit.patch_embed.proj.weight.copy_(weights["state_dict"][f"{prefix}.patch_embed.proj.weight"])
            self.swinunetr_encoder_vit.patch_embed.proj.bias.copy_(weights["state_dict"][f"{prefix}.patch_embed.proj.bias"])
            if self.conf.monai_use_promptable_attention:
                self.swinunetr_encoder_vit.layers1[0].load_from(weights, n_block=0, prefix=prefix, layer="layers1")
                self.swinunetr_encoder_vit.layers2[0].load_from(weights, n_block=0, prefix=prefix, layer="layers2")
                self.swinunetr_encoder_vit.layers3[0].load_from(weights, n_block=0, prefix=prefix, layer="layers3")
                self.swinunetr_encoder_vit.layers4[0].load_from(weights, n_block=0, prefix=prefix, layer="layers4")
            else:
                for bname, block in self.swinunetr_encoder_vit.layers1[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, prefix=prefix, layer="layers1")
                self.swinunetr_encoder_vit.layers1[0].downsample.reduction.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers1.0.downsample.reduction.weight"]
                )
                self.swinunetr_encoder_vit.layers1[0].downsample.norm.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers1.0.downsample.norm.weight"]
                )
                self.swinunetr_encoder_vit.layers1[0].downsample.norm.bias.copy_(
                    weights["state_dict"][f"{prefix}.layers1.0.downsample.norm.bias"]
                )
                for bname, block in self.swinunetr_encoder_vit.layers2[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, prefix=prefix, layer="layers2")
                self.swinunetr_encoder_vit.layers2[0].downsample.reduction.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers2.0.downsample.reduction.weight"]
                )
                self.swinunetr_encoder_vit.layers2[0].downsample.norm.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers2.0.downsample.norm.weight"]
                )
                self.swinunetr_encoder_vit.layers2[0].downsample.norm.bias.copy_(
                    weights["state_dict"][f"{prefix}.layers2.0.downsample.norm.bias"]
                )
                for bname, block in self.swinunetr_encoder_vit.layers3[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, prefix=prefix, layer="layers3")
                self.swinunetr_encoder_vit.layers3[0].downsample.reduction.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers3.0.downsample.reduction.weight"]
                )
                self.swinunetr_encoder_vit.layers3[0].downsample.norm.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers3.0.downsample.norm.weight"]
                )
                self.swinunetr_encoder_vit.layers3[0].downsample.norm.bias.copy_(
                    weights["state_dict"][f"{prefix}.layers3.0.downsample.norm.bias"]
                )
                for bname, block in self.swinunetr_encoder_vit.layers4[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, prefix=prefix, layer="layers4")
                self.swinunetr_encoder_vit.layers4[0].downsample.reduction.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers4.0.downsample.reduction.weight"]
                )
                self.swinunetr_encoder_vit.layers4[0].downsample.norm.weight.copy_(
                    weights["state_dict"][f"{prefix}.layers4.0.downsample.norm.weight"]
                )
                self.swinunetr_encoder_vit.layers4[0].downsample.norm.bias.copy_(
                    weights["state_dict"][f"{prefix}.layers4.0.downsample.norm.bias"]
                )

            # Skip connections
            if isinstance(self.swinunetr_skip_blocks[0], UnetrBasicBlock):
                self.swinunetr_skip_blocks[0].layer.conv1.conv.weight.copy_(
                    weights["state_dict"][f"encoder1.layer.conv1.conv.weight"][..., slice(1, 2) if self.swinunetr_skip_blocks[0].layer.conv1.conv.weight.shape[-1] == 1 else slice(3)]
                )
                self.swinunetr_skip_blocks[0].layer.conv2.conv.weight.copy_(
                    weights["state_dict"][f"encoder1.layer.conv2.conv.weight"][..., slice(1, 2) if self.swinunetr_skip_blocks[0].layer.conv2.conv.weight.shape[-1] == 1 else slice(3)]
                )
                # A downsample ckpt may be available (why?) but isn't used.
                # self.swinunetr_skip_blocks[0].layer.conv3.conv.weight.copy_(
                #    weights["state_dict"][f"encoder1.layer.conv3.conv.weight"]
                # )
            if isinstance(self.swinunetr_skip_blocks[1], UnetrBasicBlock):
                self.swinunetr_skip_blocks[1].layer.conv1.conv.weight.copy_(
                    weights["state_dict"][f"encoder2.layer.conv1.conv.weight"][..., slice(1, 2) if self.swinunetr_skip_blocks[1].layer.conv1.conv.weight.shape[-1] == 1 else slice(3)]
                )
                self.swinunetr_skip_blocks[1].layer.conv2.conv.weight.copy_(
                    weights["state_dict"][f"encoder2.layer.conv2.conv.weight"][..., slice(1, 2) if self.swinunetr_skip_blocks[1].layer.conv2.conv.weight.shape[-1] == 1 else slice(3)]
                )
            if isinstance(self.swinunetr_skip_blocks[2], UnetrBasicBlock):
                self.swinunetr_skip_blocks[2].layer.conv1.conv.weight.copy_(
                    weights["state_dict"][f"encoder3.layer.conv1.conv.weight"][..., slice(1, 2) if self.swinunetr_skip_blocks[2].layer.conv1.conv.weight.shape[-1] == 1 else slice(3)]
                )
                self.swinunetr_skip_blocks[2].layer.conv2.conv.weight.copy_(
                    weights["state_dict"][f"encoder3.layer.conv2.conv.weight"][..., slice(1, 2) if self.swinunetr_skip_blocks[2].layer.conv2.conv.weight.shape[-1] == 1 else slice(3)]
                )
            if isinstance(self.swinunetr_skip_blocks[3], UnetrBasicBlock):
                self.swinunetr_skip_blocks[3].layer.conv1.conv.weight.copy_(
                    weights["state_dict"][f"encoder4.layer.conv1.conv.weight"][..., slice(1, 2) if self.swinunetr_skip_blocks[3].layer.conv1.conv.weight.shape[-1] == 1 else slice(3)]
                )
                self.swinunetr_skip_blocks[3].layer.conv2.conv.weight.copy_(
                    weights["state_dict"][f"encoder4.layer.conv2.conv.weight"][..., slice(1, 2) if self.swinunetr_skip_blocks[3].layer.conv2.conv.weight.shape[-1] == 1 else slice(3)]
                )
            if isinstance(self.swinunetr_skip_blocks[4], UnetrBasicBlock):
                self.swinunetr_skip_blocks[4].layer.conv1.conv.weight.copy_(
                    weights["state_dict"][f"encoder10.layer.conv1.conv.weight"][..., slice(1, 2) if self.swinunetr_skip_blocks[4].layer.conv1.conv.weight.shape[-1] == 1 else slice(3)][:384, :384, ...]  # Uses encoder 10
                )
                self.swinunetr_skip_blocks[4].layer.conv2.conv.weight.copy_(
                    weights["state_dict"][f"encoder10.layer.conv2.conv.weight"][..., slice(1, 2) if self.swinunetr_skip_blocks[4].layer.conv2.conv.weight.shape[-1] == 1 else slice(3)][:384, :384, ...]
                )
            # Decoder
            self.swinunetr_decoder_blocks[4].transp_conv.conv.weight.copy_(
                weights["state_dict"][f"decoder5.transp_conv.conv.weight"]
            )
            self.swinunetr_decoder_blocks[4].conv_block.conv1.conv.weight.copy_(
                weights["state_dict"][f"decoder5.conv_block.conv1.conv.weight"][..., slice(1, 2) if self.swinunetr_decoder_blocks[4].conv_block.conv1.conv.weight.shape[-1] == 1 else slice(3)]
            )
            self.swinunetr_decoder_blocks[4].conv_block.conv2.conv.weight.copy_(
                weights["state_dict"][f"decoder5.conv_block.conv2.conv.weight"][..., slice(1, 2) if self.swinunetr_decoder_blocks[4].conv_block.conv2.conv.weight.shape[-1] == 1 else slice(3)][:, :384, ...]  # slice weights due to connection differences
            )
            self.swinunetr_decoder_blocks[4].conv_block.conv3.conv.weight.copy_(
                weights["state_dict"][f"decoder5.conv_block.conv3.conv.weight"]
            )
            self.swinunetr_decoder_blocks[3].transp_conv.conv.weight.copy_(
                weights["state_dict"][f"decoder4.transp_conv.conv.weight"]
            )
            self.swinunetr_decoder_blocks[3].conv_block.conv1.conv.weight.copy_(
                weights["state_dict"][f"decoder4.conv_block.conv1.conv.weight"][..., slice(1, 2) if self.swinunetr_decoder_blocks[3].conv_block.conv1.conv.weight.shape[-1] == 1 else slice(3)]
            )
            self.swinunetr_decoder_blocks[3].conv_block.conv2.conv.weight.copy_(
                weights["state_dict"][f"decoder4.conv_block.conv2.conv.weight"][..., slice(1, 2) if self.swinunetr_decoder_blocks[3].conv_block.conv2.conv.weight.shape[-1] == 1 else slice(3)]
            )
            self.swinunetr_decoder_blocks[3].conv_block.conv3.conv.weight.copy_(
                weights["state_dict"][f"decoder4.conv_block.conv3.conv.weight"]
            )
            self.swinunetr_decoder_blocks[2].transp_conv.conv.weight.copy_(
                weights["state_dict"][f"decoder3.transp_conv.conv.weight"]
            )
            self.swinunetr_decoder_blocks[2].conv_block.conv1.conv.weight.copy_(
                weights["state_dict"][f"decoder3.conv_block.conv1.conv.weight"][..., slice(1, 2) if self.swinunetr_decoder_blocks[2].conv_block.conv1.conv.weight.shape[-1] == 1 else slice(3)]
            )
            self.swinunetr_decoder_blocks[2].conv_block.conv2.conv.weight.copy_(
                weights["state_dict"][f"decoder3.conv_block.conv2.conv.weight"][..., slice(1, 2) if self.swinunetr_decoder_blocks[2].conv_block.conv2.conv.weight.shape[-1] == 1 else slice(3)]
            )
            self.swinunetr_decoder_blocks[2].conv_block.conv3.conv.weight.copy_(
                weights["state_dict"][f"decoder3.conv_block.conv3.conv.weight"]
            )
            self.swinunetr_decoder_blocks[1].transp_conv.conv.weight.copy_(
                weights["state_dict"][f"decoder2.transp_conv.conv.weight"]
            )
            self.swinunetr_decoder_blocks[1].conv_block.conv1.conv.weight.copy_(
                weights["state_dict"][f"decoder2.conv_block.conv1.conv.weight"][..., slice(1, 2) if self.swinunetr_decoder_blocks[1].conv_block.conv1.conv.weight.shape[-1] == 1 else slice(3)]
            )
            self.swinunetr_decoder_blocks[1].conv_block.conv2.conv.weight.copy_(
                weights["state_dict"][f"decoder2.conv_block.conv2.conv.weight"][..., slice(1, 2) if self.swinunetr_decoder_blocks[1].conv_block.conv2.conv.weight.shape[-1] == 1 else slice(3)]
            )
            self.swinunetr_decoder_blocks[1].conv_block.conv3.conv.weight.copy_(
                weights["state_dict"][f"decoder2.conv_block.conv3.conv.weight"]
            )
            self.swinunetr_decoder_blocks[0].transp_conv.conv.weight.copy_(
                weights["state_dict"][f"decoder1.transp_conv.conv.weight"]
            )
            self.swinunetr_decoder_blocks[0].conv_block.conv1.conv.weight.copy_(
                weights["state_dict"][f"decoder1.conv_block.conv1.conv.weight"][..., slice(1, 2) if self.swinunetr_decoder_blocks[0].conv_block.conv1.conv.weight.shape[-1] == 1 else slice(3)]
            )
            self.swinunetr_decoder_blocks[0].conv_block.conv2.conv.weight.copy_(
                weights["state_dict"][f"decoder1.conv_block.conv2.conv.weight"][..., slice(1, 2) if self.swinunetr_decoder_blocks[0].conv_block.conv2.conv.weight.shape[-1] == 1 else slice(3)]
            )
            self.swinunetr_decoder_blocks[0].conv_block.conv3.conv.weight.copy_(
                weights["state_dict"][f"decoder1.conv_block.conv3.conv.weight"]
            )

            # Output layer (if fixed)
            if hasattr(self, "conv_fixed"):
                self.conv_fixed.conv.conv.weight.copy_(
                    weights["state_dict"][f"out.conv.conv.weight"]
                )
            elif hasattr(self, "conv_out"):
                self.conv_out.conv.conv.weight.copy_(
                    weights["state_dict"][f"out.conv.conv.weight"]
                )
            else:
                pass

    def get_named_fixed_parameters(self):

        params_ = []
        if self.conf.fixed_output:
            params_ = [self.conv_fixed.named_parameters()]

        return list(chain(*params_))

    def get_named_encoder_parameters(self):

        params_ = [self.swinunetr_encoder_vit.patch_embed.named_parameters()]
        params_ += [x_.named_parameters_attention() if isinstance(x_, DeepShiftedInstructedAttentionBlock) else x_.named_parameters() for x_ in [self.swinunetr_encoder_vit.layers1[0],
                                                                                                                                                 self.swinunetr_encoder_vit.layers2[0],
                                                                                                                                                 self.swinunetr_encoder_vit.layers3[0],
                                                                                                                                                 self.swinunetr_encoder_vit.layers4[0]]]
        params_ += [x_.named_parameters_bias_content() for x_ in [self.swinunetr_encoder_vit.layers1[0],
                                                                  self.swinunetr_encoder_vit.layers2[0],
                                                                  self.swinunetr_encoder_vit.layers3[0],
                                                                  self.swinunetr_encoder_vit.layers4[0]] if isinstance(x_, DeepShiftedInstructedAttentionBlock)]

        params_ += [x_.named_parameters_downsample() for x_ in [self.swinunetr_encoder_vit.layers1[0],
                                                                self.swinunetr_encoder_vit.layers2[0],
                                                                self.swinunetr_encoder_vit.layers3[0],
                                                                self.swinunetr_encoder_vit.layers4[0]] if isinstance(x_, DeepShiftedInstructedAttentionBlock)]

        return list(chain(*params_))

    def get_named_skip_parameters(self):

        params_ = []
        params_ += [x_.named_parameters_body() if isinstance(x_, DeepSIAResBlock) else x_.named_parameters() for x_ in self.swinunetr_skip_blocks]

        return list(chain(*params_))

    def get_named_decoder_parameters(self):

        params_ = []
        params_ += [x_.named_parameters() for x_ in self.swinunetr_decoder_blocks]
        params_ += [x_.named_parameters_body() if isinstance(x_, DeepSIAResBlock) else x_.named_parameters() for x_ in self.swinunetr_decoder_blocks_late]
        params_ += [self.get_named_fixed_parameters()]
        if self.conf.final_conv:
            params_ += [self.conv_out.named_parameters()]

        return list(chain(*params_))

    def get_named_body_parameters(self):

        params_ = [self.get_named_encoder_parameters(), self.get_named_skip_parameters(), self.get_named_decoder_parameters()]

        return list(chain(*params_))

    def get_named_instruction_bias_parameters(self):

        params_ = []
        params_ += [self.get_named_instruction_bias_encoder_parameters()]
        params_ += [self.get_named_instruction_bias_skip_parameters()]
        params_ += [self.get_named_instruction_bias_decoder_parameters()]

        return list(chain(*params_))

    def get_named_instruction_bias_encoder_parameters(self):

        params_ = []
        params_ += [x_.named_parameters_bias_instructions() for x_ in [self.swinunetr_encoder_vit.layers1[0],
                                                                       self.swinunetr_encoder_vit.layers2[0],
                                                                       self.swinunetr_encoder_vit.layers3[0],
                                                                       self.swinunetr_encoder_vit.layers4[0]] if isinstance(x_, DeepShiftedInstructedAttentionBlock)]

        return list(chain(*params_))

    def get_named_instruction_bias_skip_parameters(self):

        params_ = []
        params_ += [x_.named_parameters_bias_instructions() for x_ in self.swinunetr_skip_blocks if isinstance(x_, DeepSIAResBlock)]

        return list(chain(*params_))

    def get_named_instruction_bias_decoder_parameters(self):

        params_ = []
        params_ += [x_.named_parameters_bias_instructions() for x_ in self.swinunetr_decoder_blocks_late if isinstance(x_, DeepSIAResBlock)]

        return list(chain(*params_))

    def get_named_instruction_pool_parameters(self):

        params_ = []
        params_ += [self.get_named_instruction_pool_encoder_parameters()]
        params_ += [self.get_named_instruction_pool_skip_parameters()]
        params_ += [self.get_named_instruction_pool_decoder_parameters()]

        return list(chain(*params_))

    def get_named_instruction_pool_encoder_parameters(self):

        params_ = []
        params_ += [x_.named_parameters_instruction_tokens() for x_ in self.sia_instruction_blocks_encoder if isinstance(x_, InstructionPool)]
        params_ += [x_.named_parameters_instruction_norm() for x_ in self.sia_instruction_blocks_encoder if isinstance(x_, InstructionPool)]

        return list(chain(*params_))

    def get_named_instruction_pool_skip_parameters(self):

        params_ = []
        params_ += [x_.named_parameters_instruction_tokens() for x_ in self.sia_instruction_blocks_decoder if isinstance(x_, InstructionPool)]
        params_ += [x_.named_parameters_instruction_norm() for x_ in self.sia_instruction_blocks_decoder if isinstance(x_, InstructionPool)]

        return list(chain(*params_))

    def get_named_instruction_pool_decoder_parameters(self):

        params_ = []
        params_ += [x_.named_parameters_instruction_tokens() for x_ in self.sia_instruction_blocks_decoder_late if isinstance(x_, InstructionPool)]
        params_ += [x_.named_parameters_instruction_norm() for x_ in self.sia_instruction_blocks_decoder_late if isinstance(x_, InstructionPool)]
        if not self.conf.fixed_output:
            params_ += [self.instruction_pool.named_parameters_instruction_tokens() if isinstance(self.instruction_pool, InstructionPool) else []]
            params_ += [self.instruction_pool.named_parameters_instruction_norm() if isinstance(self.instruction_pool, InstructionPool) else []]

        return list(chain(*params_))

    def get_named_instruction_parameters(self):

        params_ = [self.get_named_instruction_pool_parameters(), self.get_named_instruction_bias_parameters()]

        return list(chain(*params_))

    def get_named_adapter_parameters(self):

        params_ = []
        params_ += [self.get_named_adapter_encoder_parameters()]
        params_ += [self.get_named_adapter_skip_parameters()]
        params_ += [self.get_named_adapter_decoder_parameters()]

        return list(chain(*params_))

    def get_named_adapter_encoder_parameters(self):

        params_ = []
        params_ += [x_.named_parameters_adapter() for x_ in [self.swinunetr_encoder_vit.layers1[0],
                                                             self.swinunetr_encoder_vit.layers2[0],
                                                             self.swinunetr_encoder_vit.layers3[0],
                                                             self.swinunetr_encoder_vit.layers4[0]] if isinstance(x_, DeepShiftedInstructedAttentionBlock)]

        return list(chain(*params_))

    def get_named_adapter_skip_parameters(self):

        params_ = []
        params_ += [x_.named_parameters_adapter() for x_ in self.swinunetr_skip_blocks if isinstance(x_, DeepSIAResBlock)]

        return list(chain(*params_))

    def get_named_adapter_decoder_parameters(self):

        params_ = []
        params_ += [x_.named_parameters_adapter() for x_ in self.swinunetr_decoder_blocks_late if isinstance(x_, DeepSIAResBlock)]

        return list(chain(*params_))

    def set_requires_gradient(self,
                              grad_instructions: torch.BoolTensor,
                              grad_instructions_norm: bool = True,
                              grad_instructions_scores: bool = True,
                              grad_body: bool = True,
                              grad_vit: bool = True):

        # Disable everything (as default)
        debug = True
        for (name_, param_student_) in self.named_parameters():
            param_student_.requires_grad = False

            if debug:
                # Make sure every parameter is included in any of the submodules
                if not any([param_student_.data_ptr() == x_[1].data_ptr() for x_ in [*self.get_named_body_parameters(), *self.get_named_instruction_parameters(), *self.get_named_adapter_parameters()]]):
                    raise ValueError(f'Missing parameter {name_}')

        # Enable or disable gradients for bulk of interpreter (student) - dependent on selective freezing
        for (name_, param_student_) in self.get_named_body_parameters():
            param_student_.requires_grad = grad_body

        # Encoder - now only the ViT part - overwrites body part
        for (name_, param_student_) in self.get_named_encoder_parameters():
            param_student_.requires_grad = grad_vit

        # Last layer
        if self.conf.adaptation_variant in ['fixed', 'decoder', 'bias', 'adapter']:
            for (name_, param_student_) in self.get_named_fixed_parameters():
                param_student_.requires_grad = True

        # Adapter layers
        if self.conf.adaptation_variant in ['adapter']:
            for (name_, param_student_) in self.get_named_adapter_parameters():
                param_student_.requires_grad = True

        # Decoder
        if self.conf.adaptation_variant in ['decoder']:
            for (name_, param_student_) in self.get_named_decoder_parameters():
                param_student_.requires_grad = True

        # Enable or disable gradients for bias params in bulk of interpreter
        if self.conf.adaptation_variant in ['bias', 'bias_prompting']:
            for (name_, param_student_) in self.get_named_body_parameters():
                if 'bias' in name_ or 'norm' in name_:  # Includes all bias parameters and in addition scale parameters of norms
                    param_student_.requires_grad = True

        # Enable or disable gradients for all instruction parameters
        if self.conf.adaptation_variant in ['prompting', 'bias_prompting']:
            # Only active instruction bias scores are adjusted.
            for (name_, param_student_) in self.get_named_instruction_bias_parameters():
                if 'encoding_cross_inst_content' in name_:
                    encoding_nr = int(name_[len(name_.rstrip('0123456789')):])
                    if encoding_nr < grad_instructions.shape[0]:
                        param_student_.requires_grad = grad_instructions[encoding_nr].item() & grad_instructions_scores if self.conf.unique_instruction_bias else grad_instructions_scores
                    else:
                        param_student_.requires_grad = False  # Excess instruction remain unused and therefore default False
                elif 'weights_cross_inst_content' in name_:
                    param_student_.requires_grad = grad_instructions_scores
                else:
                    raise ValueError(f'Unexpected name {name_} in named instruction bias parameters')

            # Fine-grained instruction pool adjustments
            for (name_, param_student_) in self.get_named_instruction_pool_parameters():
                # Set token parameter
                if 'weight' in name_ or 'bias' in name_:  # instruction norm weights and bias
                    param_student_.requires_grad = grad_instructions_norm
                elif name_.isnumeric():  # instructions.0+
                    if int(name_) < grad_instructions.shape[0]:
                        param_student_.requires_grad = grad_instructions[int(name_)].item()
                    else:
                        param_student_.requires_grad = False  # Excess instruction remain unused and therefore default False
                else:
                    raise ValueError(f'Unexpected name {name_} in named instruction pool parameters.')

        # Report frozen / nonfrozen
        def _report_trainable(key_, named_params):
            print(f"Trainable are {sum([p_[1].numel() for p_ in named_params if p_[1].requires_grad])}/{sum([p_[1].numel() for p_ in named_params])} {key_} parameters.")

        print(f"Trainable parameters for adaptation variant {self.conf.adaptation_variant}.")
        _report_trainable('encoder', self.get_named_encoder_parameters())
        _report_trainable('decoder', self.get_named_decoder_parameters())
        _report_trainable('fixed layer', self.get_named_fixed_parameters())
        _report_trainable('body', self.get_named_body_parameters())
        _report_trainable('instruction bias', self.get_named_instruction_bias_parameters())  # Note: amount of truly active bias parameters may be less (for position ablations).
        _report_trainable('instruction pool', self.get_named_instruction_pool_parameters())
        _report_trainable('instruction', self.get_named_instruction_parameters())
        _report_trainable('adapter', self.get_named_adapter_parameters())
        _report_trainable('all', list(self.named_parameters()))

    def set_downstream_instruction_parameters(self, label_indices_base: List[int], label_indices_downstream: List[int]):

        if len(label_indices_base) > 0:
            print('Performing initialization of instructions based on mean representation.')
            mean_rep = torch.mean(torch.stack(list(self.instruction_pool.instruction_tokens.instructions), dim=0)[label_indices_base, ...], dim=0)  # Mean rep of existing foreground categories
            for idx_instruction, tokens_ in enumerate(self.instruction_pool.instruction_tokens.instructions):
                if idx_instruction in label_indices_downstream:
                    if idx_instruction in label_indices_base:
                        warnings.warn(f'Initializing downstream instruction {idx_instruction} with mean rep despite it being present in label_indices_base. (Ignore if intended.)')
                    tokens_.data.copy_(mean_rep)
        else:
            print('Initialization of instructions remains random since no base labels are available.')

    def forward(self,
                x: torch.Tensor,
                label_indices: Optional[torch.Tensor] = None,
                pseudo_indices_subject: Optional[torch.Tensor] = None,
                pseudo_indices_label: Optional[torch.Tensor] = None,
                mode_label: str = 'pseudo',
                mode_loss: str = 'both'):  # pseudo or label
        dict_out = collections.defaultdict(dict)
        batch_size = x.shape[0]
        # Fetch instructions
        # Note: mode does not have an effect atm. Pseudo does not exist for deep (since it would be too large)

        # SwinViT Encoder
        if mode_loss == 'self' \
         or (self.conf.noninstructed_attention and not self.conf.downstream) \
         or (self.conf.noninstructed_attention_downstream and self.conf.downstream) \
         or self.conf.prompting_variant.lower() not in ['full', 'encoder'] \
         or not self.conf.monai_use_promptable_attention:
            vit_enc = self.swinunetr_encoder_vit(x, normalize=True)
        else:
            vit_enc = self.swinunetr_encoder_vit(x, self.sia_instruction_blocks_encoder, label_indices=label_indices, normalize=True)

        # UNETR-like skip blocks (part of the decoder)
        if any([self.conf.model_variant.lower() == x_ for x_ in ['swin_unetr', 'punet_decoder', 'punet_decoder_low_res', 'punet_decoder_high_res']]) \
            or mode_loss == 'self' \
            or (self.conf.noninstructed_attention and not self.conf.downstream) \
            or (self.conf.noninstructed_attention_downstream and self.conf.downstream) \
            or self.conf.prompting_variant.lower() not in ['full', 'decoder']:
            swinunetr_skip_0 = self.swinunetr_skip_blocks[0](x)
            swinunetr_skip_1 = self.swinunetr_skip_blocks[1](vit_enc[0])
            swinunetr_skip_2 = self.swinunetr_skip_blocks[2](vit_enc[1])
            swinunetr_skip_3 = self.swinunetr_skip_blocks[3](vit_enc[2])
            swinunetr_skip_4 = self.swinunetr_skip_blocks[4](vit_enc[3])
            swinunetr_skip_5 = vit_enc[4]  # last stage is not processed according to figure (different from monai's implementation)
        elif self.conf.model_variant == 'punet':
            swinunetr_skip_0 = self.swinunetr_skip_blocks[0](x)
            swinunetr_skip_1 = self.swinunetr_skip_blocks[1](vit_enc[0], [x_(label_indices, batch_size=batch_size) for x_ in self.sia_instruction_blocks_decoder[2:4]], label_indices=label_indices)
            swinunetr_skip_2 = self.swinunetr_skip_blocks[2](vit_enc[1], [x_(label_indices, batch_size=batch_size) for x_ in self.sia_instruction_blocks_decoder[4:6]], label_indices=label_indices)
            swinunetr_skip_3 = self.swinunetr_skip_blocks[3](vit_enc[2], [x_(label_indices, batch_size=batch_size) for x_ in self.sia_instruction_blocks_decoder[6:8]], label_indices=label_indices)
            swinunetr_skip_4 = self.swinunetr_skip_blocks[4](vit_enc[3], [x_(label_indices, batch_size=batch_size) for x_ in self.sia_instruction_blocks_decoder[8:10]], label_indices=label_indices)
            swinunetr_skip_5 = vit_enc[4]  # last stage is not processed according to figure (different from monai's implementation)
        else:
            raise NotImplementedError
        dict_out['dense']['skips'] = [swinunetr_skip_1, swinunetr_skip_2, swinunetr_skip_3, swinunetr_skip_4]

        # Final up / decoder blocks
        if any([self.conf.model_variant.lower() == x_ for x_ in ['swin_unetr', 'punet']]):
            swinunetr_dec_4 = self.swinunetr_decoder_blocks[4](swinunetr_skip_5, swinunetr_skip_4)
            swinunetr_dec_3 = self.swinunetr_decoder_blocks[3](swinunetr_dec_4, swinunetr_skip_3)
            swinunetr_dec_2 = self.swinunetr_decoder_blocks[2](swinunetr_dec_3, swinunetr_skip_2)
            swinunetr_dec_1 = self.swinunetr_decoder_blocks[1](swinunetr_dec_2, swinunetr_skip_1)
            swinunetr_dec_0 = self.swinunetr_decoder_blocks[0](swinunetr_dec_1, swinunetr_skip_0)
        elif any([self.conf.model_variant.lower() == x_ for x_ in ['punet_decoder', 'punet_decoder_low_res', 'punet_decoder_high_res']]):
            if mode_loss == 'self' \
                or (self.conf.noninstructed_attention and not self.conf.downstream) \
                or (self.conf.noninstructed_attention_downstream and self.conf.downstream) \
                or self.conf.prompting_variant.lower() not in ['full', 'decoder']:
                swinunetr_dec_4 = self.swinunetr_decoder_blocks[4](swinunetr_skip_5, swinunetr_skip_4)
                swinunetr_dec_4 = self.swinunetr_decoder_blocks_late[4](swinunetr_dec_4)
                swinunetr_dec_3 = self.swinunetr_decoder_blocks[3](swinunetr_dec_4, swinunetr_skip_3)
                swinunetr_dec_3 = self.swinunetr_decoder_blocks_late[3](swinunetr_dec_3)
                swinunetr_dec_2 = self.swinunetr_decoder_blocks[2](swinunetr_dec_3, swinunetr_skip_2)
                swinunetr_dec_2 = self.swinunetr_decoder_blocks_late[2](swinunetr_dec_2)
                swinunetr_dec_1 = self.swinunetr_decoder_blocks[1](swinunetr_dec_2, swinunetr_skip_1)
                swinunetr_dec_1 = self.swinunetr_decoder_blocks_late[1](swinunetr_dec_1)
                swinunetr_dec_0 = self.swinunetr_decoder_blocks[0](swinunetr_dec_1, swinunetr_skip_0)
                swinunetr_dec_0 = self.swinunetr_decoder_blocks_late[0](swinunetr_dec_0) if self.conf.model_variant.lower() == 'punet_decoder_high_res' else swinunetr_dec_0
            else:
                swinunetr_dec_4 = self.swinunetr_decoder_blocks[4](swinunetr_skip_5, swinunetr_skip_4)
                swinunetr_dec_4 = self.swinunetr_decoder_blocks_late[4](swinunetr_dec_4, [x_(label_indices, batch_size=batch_size) for x_ in self.sia_instruction_blocks_decoder_late[8:10]], label_indices=label_indices)
                swinunetr_dec_3 = self.swinunetr_decoder_blocks[3](swinunetr_dec_4, swinunetr_skip_3)
                swinunetr_dec_3 = self.swinunetr_decoder_blocks_late[3](swinunetr_dec_3, [x_(label_indices, batch_size=batch_size) for x_ in self.sia_instruction_blocks_decoder_late[6:8]], label_indices=label_indices)
                swinunetr_dec_2 = self.swinunetr_decoder_blocks[2](swinunetr_dec_3, swinunetr_skip_2)
                swinunetr_dec_2 = self.swinunetr_decoder_blocks_late[2](swinunetr_dec_2, [x_(label_indices, batch_size=batch_size) for x_ in self.sia_instruction_blocks_decoder_late[4:6]], label_indices=label_indices)
                swinunetr_dec_1 = self.swinunetr_decoder_blocks[1](swinunetr_dec_2, swinunetr_skip_1)
                swinunetr_dec_1 = self.swinunetr_decoder_blocks_late[1](swinunetr_dec_1, [x_(label_indices, batch_size=batch_size) for x_ in self.sia_instruction_blocks_decoder_late[2:4]], label_indices=label_indices)
                swinunetr_dec_0 = self.swinunetr_decoder_blocks[0](swinunetr_dec_1, swinunetr_skip_0)
                swinunetr_dec_0 = self.swinunetr_decoder_blocks_late[0](swinunetr_dec_0, [x_(label_indices, batch_size=batch_size) for x_ in self.sia_instruction_blocks_decoder_late[0:2]], label_indices=label_indices) if self.conf.model_variant.lower() == 'punet_decoder_high_res' else swinunetr_dec_0
        dict_out['patched']['embedded_latents'] = swinunetr_dec_0 if not self.conf.final_conv else self.conv_out(swinunetr_dec_0)

        # Segmentation recombination
        if not self.conf.fixed_output:
            x_instructions_final = self.instruction_pool(label_indices, batch_size=batch_size)
            dict_out['instructions']['segmentation_latents'] = x_instructions_final

            h_, w_, d_ = dict_out['patched']['embedded_latents'].shape[-3:]
            x_sim_latents = einops.rearrange(dict_out['patched']['embedded_latents'], 'b c h w d -> b (h w d) c')
            x_sim_instructions = einops.rearrange(dict_out['instructions']['segmentation_latents'], 'b (i n) c -> b i n c', n=self.conf.tokens_per_instruction_seg)  # [B, I, N, C]. Should add up to the same form regardless of self.separate_background (for binary case).
            x_sim = similarity_aggregation(latents=x_sim_latents,
                                           instructions=x_sim_instructions,
                                           mean_aggregation=self.conf.mean_aggregation,
                                           top_k_selection=self.conf.top_k_selection,
                                           soft_selection_sigma=self.conf.soft_selection_sigma,
                                           normalization=self.conf.sim_normalization,
                                           legacy=not self.conf.no_sim_legacy)
            x_sim = einops.rearrange(x_sim, 'b i (h w d) -> b i h w d', h=h_, w=w_, d=d_)
        else:
            dict_out['instructions']['segmentation_latents'] = None

            # assert self.conf.architecture == 'wip_simple'  # Should only be used in conjunction with simple (multiclass) case.
            x_sim = self.conv_fixed(dict_out['patched']['embedded_latents'])

        dict_out['dense']['embedded_latents'] = x_sim

        return dict_out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--instruction_pool_size', default=10, type=int)
        parser.add_argument('--instruction_pool_size_pseudo_subjects', default=100, type=int)
        parser.add_argument('--instruction_pool_size_pseudo_labels', default=51, type=int)
        parser.add_argument('--instruction_channels', default=48, type=int)
        parser.add_argument('--tokens_per_instruction', default=24, type=int)
        parser.add_argument('--tokens_per_instruction_seg', default=24, type=int)
        # parser.add_argument('--tokens_per_background', default=20, type=int)
        parser.add_argument('--attention_heads', default=[2, 3, 6, 12, 24], type=int)
        parser.add_argument('--attn_window_size', default=[7, 7, 5], nargs=3, type=int)
        parser.add_argument('--hidden_channels', default=[48, 96, 192, 384, 768], nargs='*', type=int)
        parser.add_argument('--monai_attn_window_size', default=[7, 7, 5], nargs=3, type=int)
        parser.add_argument('--monai_conv_norm_name', default="instance", type=str)
        parser.add_argument('--monai_swin_vit', default=True, type=bool)
        parser.add_argument('--monai_swin_vit_depths', default=[2, 2, 2, 2], type=int)
        parser.add_argument('--monai_swin_vit_pretrained', action='store_true')
        parser.add_argument('--monai_swin_vit_ckpt', default='/my/data/swinvit.pt', type=str)  # (EDIT ME IF NEEDED)
        parser.add_argument('--monai_swin_unetr_pretrained', action='store_true')
        parser.add_argument('--monai_swin_unetr_ckpt', default='/my/data//swin_unetr_btcv_base_5000ep_f48_lr2e-4_pretrained.pt', type=str)  # (EDIT ME IF NEEDED)
        parser.add_argument('--monai_drop_rate', default=0.1, type=float),
        parser.add_argument('--monai_attn_drop_rate', default=0.1, type=float)
        parser.add_argument('--monai_dropout_path_rate', default=0.0, type=float)
        parser.add_argument('--monai_legacy_bias', action='store_true')  # Whether to use own relative bias (known from promptable blocks) or the more simple relative bias that comes with the SwinUNETR
        parser.add_argument('--monai_use_checkpoint', default=True, type=bool)
        parser.add_argument('--monai_use_promptable_attention', default=True, type=bool)  # Promptable blocks do not support drop path (at least not yet).

        # Instruction initialization / aggregation
        parser.add_argument('--noninstructed_attention', action='store_true')  # Attention layers are not instructed
        parser.add_argument('--noninstructed_attention_downstream', action='store_true')
        parser.add_argument('--top_k_selection', action='store_true')  # True: Aggregate via softmax re-weighting, False: Mean over topk (atm 3)
        parser.add_argument('--soft_selection_sigma', default=0.0625, type=float)  # Temperature for softmax re-weighting. Roughly 1 / #tokens
        parser.add_argument('--mean_aggregation', action='store_true')  # Aggregate instructions without any sophisticated selection
        parser.add_argument('--mean_initialization', default=False, type=bool)  # Use mean representation of learned (base) categories for initialization of new (unseen) downstream category
        parser.add_argument('--fixed_output', action='store_true')  # Fixed linear output layer instead of cosine similarity matching with instructions.
        parser.add_argument('--instructions_use_norm', default=True, type=bool)  # Norm all instructions in a pool by a common norm.
        parser.add_argument('--instructions_use_norm_final', default=True, type=bool)
        parser.add_argument('--no_instructions_use_norm_final', dest='instructions_use_norm_final', action='store_false')
        parser.add_argument('--instructions_elementwise_affine', default=True, type=bool)  # Enable / disable elementwise affine params for instruction norm.
        parser.add_argument('--prompting_variant', default='full', type=str, choices=['start', 'end', 'encoder', 'decoder', 'full'])
        parser.add_argument('--adaptation_variant', default='prompting', type=str, choices=['prompting', 'fixed', 'decoder', 'bias', 'adapter', 'bias_prompting'])
        parser.add_argument('--model_variant', default='punet_decoder', type=str, choices=['punet', 'punet_decoder', 'punet_decoder_low_res', 'punet_decoder_high_res', 'swin_unetr'])
        parser.add_argument('--sim_normalization', action='store_true')  # Enable normalization prior to similarity aggregation
        parser.add_argument('--no_sim_legacy', action='store_true')  # Legacy shift in sim agg
        parser.add_argument('--final_conv', default=True, type=bool)
        parser.add_argument('--instruction_dropout', default=0.05, type=float)

        # Attention bias scheme
        parser.add_argument('--unique_instruction_bias', default=False, type=bool)  # If True each Instruction has a unique bias score.
        parser.add_argument('--unique_token_bias', default=True, type=bool)  # If True each Token (across all instructions) has a unique bias score. IF False all bias scores are the same regardless of the token.
        parser.add_argument('--no_bias_instructions', action='store_true')
        parser.add_argument('--no_bias_content', action='store_true')
        parser.add_argument('--pre_scale', default=True, type=bool)  # Use pre-scaled bias scores or not. Non-scaled ones may influence the attention scheme more (at least for the first few epochs).

        return parser
