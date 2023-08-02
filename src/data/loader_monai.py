from torch.utils.data import WeightedRandomSampler, RandomSampler  # Would this (random_split) be a good option for ddp?
from argparse import ArgumentParser, Namespace
from src.data.distributed_wrapper import DistributedSamplerWrapper
import torch
import pytorch_lightning as pl
from typing import Union, Dict, Optional, Type
import numpy as np

from src.data.transforms_monai import generate_transforms, generate_test_transforms
from src.data.conversion_monai import convert_subjects
# from src.data.datasets.gather_tiny_ixi import gather_data
from src.data.datasets import gather_tcia_btcv, gather_tseg
from monai.data import DataLoader, Dataset, CacheDataset, set_track_meta  # Wrapper around torch DataLoader and Dataset


class BasicDataModule(pl.LightningDataModule):
    def __init__(self, conf: Union[Dict, Namespace]):
        super().__init__()
        self.conf = conf  # Model and DataModule hparams are now the same. So only one save_hyperparameters is allowed
        self.transform_train, self.transform_val, self.transform_test, self.transform_test_post = None, None, None, None
        self.df_train, self.df_val, self.df_test = None, None, None,
        self.ds_train, self.ds_val, self.ds_test = None, None, None

    def prepare_data(self):
        # Nothing to do here - best to do it offline
        pass

    def _get_max_shape_train(self, ds_subjects: Type[Dataset]):
        shapes = np.array([crop_['image'].shape for sub_ in ds_subjects for crop_ in sub_])
        return shapes.max(axis=0)

    def _get_max_shape_val(self, ds_subjects: Type[Dataset]):
        shapes = np.array([sub_['image'].shape for sub_ in ds_subjects])
        return shapes.max(axis=0)

    def _get_foreground_background_ratio(self, ds_subjects: Type[Dataset]):
        ratios = np.zeros((self.conf.out_channels - 1,))
        for sub_ in ds_subjects:
            background_ = (sub_['label'] == 0.).float().count_nonzero().item()
            for idx_ in range(1, self.conf.out_channels):
                ratios[idx_ - 1] += (sub_['label'] == idx_).float().count_nonzero().item() / background_
        ratios /= len(ds_subjects)
        ratios_inv = 1 / ratios
        ratios_bound = np.concatenate([[0], self.conf.additive_alpha_factor * ratios_inv], axis=0)
        return ratios_bound

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage == 'test' or stage is None:  # Atm. only one stage is used for all routines.

            # Data gathering / preparation
            if self.conf.dataset.lower() == 'tcia_btcv':
                self.df_train, self.df_val, self.df_test = gather_tcia_btcv.generate_dataframes(self.conf)
            elif self.conf.dataset.lower() == 'tseg':
                if self.conf.domain == 'lobes':
                    assert self.conf.out_channels == 6
                if self.conf.domain == 'abdomen':
                    assert self.conf.out_channels == 7
                if self.conf.domain == 'all':
                    assert self.conf.out_channels == 11
                self.df_train, self.df_val, self.df_test = gather_tseg.generate_dataframes(self.conf)
            else:
                raise NotImplementedError()
            dict_subjects_train, content_keys, aux_keys = convert_subjects(self.df_train)
            dict_subjects_val, _, _ = convert_subjects(self.df_val)
            dict_subjects_test, _, _ = convert_subjects(self.df_test)
            # set_track_meta(True)
            self.transform_train, self.transform_val = generate_transforms(patch_size_students=self.conf.patch_size_students,
                                                                           patch_size_teacher=self.conf.patch_size_teacher,
                                                                           content_keys=content_keys,
                                                                           aux_keys=aux_keys,
                                                                           num_samples=self.conf.num_samples,
                                                                           n_transforms=self.conf.num_transforms,
                                                                           a_min=self.conf.a_min,
                                                                           a_max=self.conf.a_max,
                                                                           b_min=self.conf.b_min,
                                                                           b_max=self.conf.b_max,
                                                                           variant=self.conf.augmentation_variant,
                                                                           orientation='xy',
                                                                           crop_foreground=self.conf.crop_foreground and self.conf.downstream,
                                                                           crop_label=self.conf.crop_label and self.conf.downstream,
                                                                           crop_label_fine=self.conf.crop_label_fine and self.conf.downstream,
                                                                           label_indices=self.conf.label_indices_downstream_active if self.conf.architecture == 'wip' else (),
                                                                           normalize=self.conf.normalize_data)  # Cropping or padding screws with the coord grid; so prepare data beforehand properly.
            self.transform_test = generate_test_transforms(content_keys=content_keys,
                                                           aux_keys=aux_keys,
                                                           a_min=self.conf.a_min,
                                                           a_max=self.conf.a_max,
                                                           b_min=self.conf.b_min,
                                                           b_max=self.conf.b_max,
                                                           normalize=self.conf.normalize_data)

            # Datasets
            cls_dataset = CacheDataset if self.conf.cache_dataset else Dataset
            self.ds_train = cls_dataset(data=dict_subjects_train, transform=self.transform_train)  # (monai's) CacheDataset may bring some speed up (for deterministic transforms)
            self.ds_val = cls_dataset(data=dict_subjects_val, transform=self.transform_val)
            self.ds_test = cls_dataset(data=dict_subjects_test, transform=self.transform_test)

            recalc_ratios = False  # atm hardcoded
            if recalc_ratios:
                self.ds_train_dummy = Dataset(data=dict_subjects_train, transform=self.transform_test)
                additive_alpha = self._get_foreground_background_ratio(self.ds_train_dummy)
                print(f'Recalced additive alpha based on foreground / background ratio: {additive_alpha}')

            max_shape_train = self._get_max_shape_train(self.ds_train)
            max_shape_val = self._get_max_shape_val(self.ds_val)

            print(f'Amount of training samples: {len(self.ds_train)}, and validation samples: {len(self.ds_val)}.')
            print(f'Max shapes for train: {max_shape_train} and val: {max_shape_val}.')
            [print(f'Key: {k_}, Value: {type(v_)}') for k_, v_ in self.ds_train[0][0].items()]
            print(f'Using additive alpha: {self.conf.additive_alpha}.')

        else:
            raise ValueError(f'Stage {stage} is not available.')

    # For self-sup augmentation see: https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/simclr/transforms.py#L17-L91
    def train_dataloader(self):
        # See site-packages/pytorch_lightning/trainer for replace_sampler_ddp
        num_samples = int(self.conf.num_samples_epoch / self.conf.num_transforms / self.conf.num_samples)
        print(f'Drawing {num_samples} of {len(self.ds_train)} training samples.')
        if self.conf.weighting:
            weights = torch.tensor(self.df_train['weights'].values)

            if self.conf.accelerator == 'ddp':
                sampler = DistributedSamplerWrapper(WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=self.conf.replacement), shuffle=True)
            else:
                sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=self.conf.replacement)  # should always shuffle (due to random draw)
        else:
            sampler = RandomSampler(self.ds_train, replacement=True, num_samples=num_samples)
        assert (self.conf.batch_size / self.conf.num_transforms / self.conf.num_samples).is_integer()

        loader_xy = DataLoader(self.ds_train,
                               batch_size=self.conf.batch_size // self.conf.num_transforms // self.conf.num_samples,
                               num_workers=self.conf.num_workers,
                               pin_memory=self.conf.pin_memory,
                               shuffle=False,  # Has to be False for given sampler
                               sampler=sampler,
                               # collate_fn=collate_list,
                               drop_last=True)
        loaders = [loader_xy]
        return loaders

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          batch_size=1,  # Note: Can be smaller than batch_size if len(ds_val) < batch_size.
                          num_workers=self.conf.num_workers,
                          pin_memory=self.conf.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.ds_test,
                          batch_size=1,
                          num_workers=self.conf.num_workers,
                          pin_memory=self.conf.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.ds_test,
                          batch_size=1,
                          num_workers=self.conf.num_workers,
                          pin_memory=self.conf.pin_memory)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--pin_memory', action='store_true')
        parser.add_argument('--replacement', default=True, type=bool)  # Use sampling with replacement if annotation is only sparsely available
        parser.add_argument('--weighting', default=True, type=bool)  # Custom weighting
        parser.add_argument('--sample_factor', default=1., type=float)  # Amount of samples drawn per epoch for weighted sampling
        parser.add_argument('--num_transforms', default=1, type=int)  # Amount of transforms applied to a selected (cropped) sample. For > 1, this produces a "positive" pair. Keep it high-ish so crops find some overlapping regions
        parser.add_argument('--num_samples', default=2, type=int)  # Samples drawn per subject
        parser.add_argument('--num_samples_epoch', default=2000, type=int)  # Amount of (overall) samples in an epoch
        parser.add_argument('--queue_max_length', default=36, type=int)
        parser.add_argument('--max_subjects_train', default=-1, type=int)
        parser.add_argument('--cache_dataset', default=False, type=bool)
        parser.add_argument('--dataset', default='tcia_btcv', type=str, choices=['tcia_btcv', 'ctorg', 'tseg'])
        parser.add_argument('--a_min', default=-175, type=float)  # only used if no data normalization is performed
        parser.add_argument('--a_max', default=250, type=float)
        parser.add_argument('--b_min', default=0, type=float)
        parser.add_argument('--b_max', default=1, type=float)
        parser.add_argument('--crop_foreground', default=False, type=float)
        parser.add_argument('--crop_label', default=True, type=float)
        parser.add_argument('--crop_label_fine', default=True, type=float)
        parser.add_argument('--augmentation_variant', default=0, type=int, choices=[0])  # Select augmentation variant by index

        hparams_tmp = parser.parse_known_args()[0]
        if hparams_tmp.dataset.lower() == 'tcia_btcv':
            parser = gather_tcia_btcv.add_data_specific_args(parser)
        elif hparams_tmp.dataset.lower() == 'tseg':
            parser = gather_tseg.add_data_specific_args(parser)
        else:
            raise NotImplementedError(f'The selected architecture {hparams_tmp.architecture} is not available.')

        return parser
