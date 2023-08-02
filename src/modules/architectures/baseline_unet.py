import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Dict
from argparse import Namespace, ArgumentParser
# from monai.networks.nets import UNet
from src.modules.architectures.monai_unet_custom import UNet
from monai.networks.layers.utils import get_act_layer, get_norm_layer


# Monai UNet model
class MonaiUNet(nn.Module):
    def __init__(self,
                 conf: Union[Dict, Namespace]):
        super().__init__()
        self.conf = conf
        self.norm_name = 'instance'  # atm hardcoded.

        self.net = UNet(spatial_dims=3,
                        in_channels=self.conf.in_channels,
                        out_channels=48,
                        channels=(48, 96, 192, 384, 768),
                        strides=(2, 2, 2, 2),
                        kernel_size=3,
                        up_kernel_size=3,
                        num_res_units=2,
                        act='PRELU',
                        norm=self.norm_name,
                        dropout=0.1,
                        bias=True)

        self.norm_seg = get_norm_layer(name=self.norm_name, spatial_dims=3, channels=48)
        self.conv_seg = nn.Conv3d(in_channels=48,
                                  out_channels=self.conf.out_channels,
                                  kernel_size=(1, 1, 1))

        self.norm_emb = get_norm_layer(name=self.norm_name, spatial_dims=3, channels=48)
        self.conv_emb = nn.Conv3d(in_channels=48,
                                  out_channels=48,
                                  kernel_size=(1, 1, 1))

    def forward(self, x: torch.Tensor):

        x = self.net(x)
        x_seg = self.conv_seg(F.leaky_relu(self.norm_seg(x)))
        x_emb = self.conv_emb(F.leaky_relu(self.norm_emb(x)))

        dict_out = {'dense': {'embedded_latents': x_seg},
                    'patched':  {'embedded_latents': x_emb}}

        return dict_out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        return parser
