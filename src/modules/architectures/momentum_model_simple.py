import torch
import torch.nn as nn
from argparse import ArgumentParser, Namespace
from typing import Union, Dict, Optional, List
from src.modules.architectures.baseline_unet import MonaiUNet
from src.modules.architectures.baseline_swin_unetr import MonaiSwinUNETR
from src.modules.architectures.swin_unetr_deep import DeepSIAUNetr
import pathlib as plb


class MomentumModelSimple(nn.Module):
    def __init__(self,
                 conf: Union[Dict, Namespace]):
        super().__init__()
        self.conf = conf
        self.current_tau_instructions = conf.initial_tau_instructions
        self.current_tau_body = conf.initial_tau_body

        # Architecture
        if self.conf.architecture.lower() == 'wip_simple':
            self.architecture = DeepSIAUNetr
        elif self.conf.architecture.lower() == 'unet':
            self.architecture = MonaiUNet
        elif self.conf.architecture.lower() == 'swin_unetr':
            self.architecture = MonaiSwinUNETR
        else:
            raise NotImplementedError(f'The selected architecture {self.conf.architecture} is not available.')

        # Base architecture (x2). Anything in there will be replicated 2x.
        self.network_student = self.architecture(conf=self.conf)
        self.network_teacher = self.architecture(conf=self.conf)

        if self.conf.architecture.lower() == 'wip_simple':
            # Loading of (externally) pre-trained models
            if self.conf.monai_swin_vit_pretrained and self.conf.monai_swin_unetr_pretrained:
                raise ValueError('Loading parameters for the encoder (swin vit) as well as the whole model (swin unetr) is not sensible.')

            # Load pre-trained params of SwinViT
            if self.conf.monai_swin_vit_pretrained:
                if self.conf.ckpt is None and self.conf.ckpt_run_name is None:
                    assert plb.Path(self.conf.monai_swin_vit_ckpt).exists()
                    weights = torch.load(self.conf.monai_swin_vit_ckpt)
                    self.network_student.load_swin_vit(weights=weights)

            # Load pre-trained params of SwinUNETR
            if self.conf.monai_swin_unetr_pretrained:
                if self.conf.ckpt is None and self.conf.ckpt_run_name is None:
                    assert plb.Path(self.conf.monai_swin_unetr_ckpt).exists()
                    weights = torch.load(self.conf.monai_swin_unetr_ckpt)
                    self.network_student.load_swin_unetr(weights=weights)

        # Overwrite teacher initialization with teachers one and disable gradients
        for (name, param_student_), (_, param_teacher_) in zip(
            self.network_student.named_parameters(),
            self.network_teacher.named_parameters(),
        ):
            param_teacher_.data.copy_(param_student_.data)  # initialize teacher with identical data as student
            param_teacher_.requires_grad = False  # Do not update by gradient

    def forward(self,
                x: List[torch.Tensor],
                x_teacher: Optional[torch.Tensor] = None):

        dict_out_students = [self.network_student(x_) for x_ in x]
        dict_out_teacher = None
        if x_teacher is not None:
            with torch.no_grad():
                dict_out_teacher = self.network_teacher(x_teacher)

        return dict_out_students, dict_out_teacher

    def update_teacher(self):
        # Apply momentum weight update
        # Note: batch norms are in general left untouched (i.e. are updated separately)
        for (name, param_student_), (_, param_teacher_) in zip(
            self.network_student.named_parameters(),
            self.network_teacher.named_parameters(),
        ):
            if 'instruction_pool' in name:
                param_teacher_.data = self.current_tau_instructions * param_teacher_.data + (1 - self.current_tau_instructions) * param_student_.data
            else:
                param_teacher_.data = self.current_tau_body * param_teacher_.data + (1 - self.current_tau_body) * param_student_.data

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        hparams_tmp = parser.parse_known_args()[0]
        if hparams_tmp.architecture.lower() == 'wip_simple':
            parser = DeepSIAUNetr.add_model_specific_args(parser)
        elif hparams_tmp.architecture.lower() == 'unet':
            parser = MonaiUNet.add_model_specific_args(parser)
        elif hparams_tmp.architecture.lower() == 'swin_unetr':
            parser = MonaiSwinUNETR.add_model_specific_args(parser)
        else:
            raise NotImplementedError(f'The selected architecture {hparams_tmp.architecture} is not available.')

        # Momentum
        parser.add_argument('--initial_tau_instructions', default=0.99, type=float)
        parser.add_argument('--initial_tau_body', default=0.99, type=float)

        return parser
