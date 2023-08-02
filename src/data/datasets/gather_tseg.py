from typing import Union, Dict, Tuple
from argparse import ArgumentParser, Namespace
import pathlib as plb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from typing import List
from src.data.datasets.gather_data import _generate_splits, _mask_domains


def generate_dataframes(conf: Union[Dict, Namespace]):

    df_dirs = _gather_data(dir_images=conf.dir_images,
                           dir_masks=conf.dir_masks,
                           domain=conf.domain)

    df_train, df_val, df_test = _generate_splits(df_dirs,
                                                 num_annotated=conf.num_annotated,
                                                 domains=(conf.domain,),
                                                 max_subjects_train=conf.max_subjects_train)

    df_train = _mask_domains(df_train,
                             modality=conf.masked_modality,
                             valid_choices=(conf.domain,))

    return df_train, df_val, df_test


def _gather_data(dir_images: str,
                 dir_masks: str,
                 domain: str,
                 pseudo_segmentation: bool = False):

    data_dirs = dict()
    entries = ['names', 'frames', 'domains', 'images', 'masks']
    entries = entries + ['pseudos'] if pseudo_segmentation else entries
    for key_ in entries:
        data_dirs[key_] = list()

    if 'lobes' == domain:
        paths_images_lobes = sorted((plb.Path(dir_images) / 'processed_tseg_lobes').rglob('*_img.nii.gz'))
        paths_masks_lobes = [x_ for x_ in sorted((plb.Path(dir_masks) / 'processed_tseg_lobes').rglob('*_lbl.nii.gz')) if '_pseudo' not in x_.name]
        data_dirs['names'].extend([x_.name.split('.')[0] + '_lobes' for x_ in paths_images_lobes])
        data_dirs['frames'].extend([x_.name.split('.')[0] + '_lobes' for x_ in paths_images_lobes])
        data_dirs['domains'].extend(['lobes' for _ in paths_images_lobes])
        data_dirs['images'].extend(paths_images_lobes)
        data_dirs['masks'].extend(paths_masks_lobes)

        if pseudo_segmentation:
            paths_masks_pseudo_tcia = sorted((plb.Path(dir_masks) / 'processed_tcia').rglob('label*_pseudo.nii.gz'))
            data_dirs['pseudos'].extend(paths_masks_pseudo_tcia)

    if 'abdomen' == domain:
        paths_images_abdomen = sorted((plb.Path(dir_images) / 'processed_tseg_abdomen').rglob('*_img.nii.gz'))
        paths_masks_abdomen = [x_ for x_ in sorted((plb.Path(dir_masks) / 'processed_tseg_abdomen').rglob('*_lbl.nii.gz')) if '_pseudo' not in x_.name]
        data_dirs['names'].extend([x_.name.split('.')[0] + '_abdomen' for x_ in paths_images_abdomen])
        data_dirs['frames'].extend([x_.name.split('.')[0] + '_abdomen' for x_ in paths_images_abdomen])
        data_dirs['domains'].extend(['abdomen' for _ in paths_images_abdomen])
        data_dirs['images'].extend(paths_images_abdomen)
        data_dirs['masks'].extend(paths_masks_abdomen)

    if 'all' == domain:
        paths_images_abdomen = sorted((plb.Path(dir_images) / 'processed_tseg_foreground').rglob('*_img.nii.gz'))
        paths_masks_abdomen = [x_ for x_ in sorted((plb.Path(dir_masks) / 'processed_tseg_foreground').rglob('*_lbl.nii.gz')) if '_pseudo' not in x_.name]
        data_dirs['names'].extend([x_.name.split('.')[0] + '_foreground' for x_ in paths_images_abdomen])
        data_dirs['frames'].extend([x_.name.split('.')[0] + '_foreground' for x_ in paths_images_abdomen])
        data_dirs['domains'].extend(['foreground' for _ in paths_images_abdomen])
        data_dirs['images'].extend(paths_images_abdomen)
        data_dirs['masks'].extend(paths_masks_abdomen)

    debug = False
    if debug:
        import matplotlib
        import nibabel as nib
        matplotlib.use('tkagg')
        viewer = nib.viewers.OrthoSlicer3D(np.array(np.stack([nib.load(paths_images_lobes[0]).get_fdata(),
                                                              nib.load(paths_masks_lobes[0]).get_fdata()], axis=-1)))
        viewer.show()
        viewer = nib.viewers.OrthoSlicer3D(np.array(np.stack([nib.load(paths_images_abdomen[0]).get_fdata(),
                                                              nib.load(paths_masks_abdomen[0]).get_fdata()], axis=-1)))
        viewer.show()

    df_dirs = pd.DataFrame(data_dirs)
    df_dirs = df_dirs.assign(annotated=True)
    df_dirs = df_dirs.assign(weights=1.0)  # Default weight is 1.0

    return df_dirs


def add_data_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--dir_images', default='/path/to/my/data/', type=str)
    parser.add_argument('--dir_masks', default='/path/to/my/data/', type=str)
    # parser.add_argument('--dir_scribbles', default='/mnt/SSD_SATA_03/data_med/scribbles/acdc_scribbles_2020_fixed', type=str)
    # parser.add_argument('--image_size', default=[64, 64, 48], nargs=3, type=int)
    parser.add_argument('--n_students', default=2, type=int)
    parser.add_argument('--patch_size_students', default="128,128,24; 96,96,16", type=list_of_tupels)
    parser.add_argument('--patch_size_teacher', default=[128, 128, 24], nargs=3, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--out_channels', default=6, type=int)  # lobes: 6, abdomen: 7
    parser.add_argument('--out_channels_pseudo', default=51, type=int)
    parser.add_argument('--masked_modality', default='', type=str, choices=['tcia', 'btcv'])
    parser.add_argument('--domain', default='lobes', type=str, choices=['lobes', 'abdomen', 'all'])  # Present domains. Used e.g. for domain-wise prototypes
    parser.add_argument('--num_annotated', default=-1, type=int)  # Determines amount of annotated subjects available during training. 3 (x2) ~ 10%, 6 (x2) ~ 20%, 15 (x2) ~ 50%.
    parser.add_argument('--additive_alpha', default=[0., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], type=float)  # Additive alpha value based on foreground / background ratio. 0 for background.
    parser.add_argument('--additive_alpha_factor', default=0.001, type=float)  # Factor to compress the range of minimal (0.) and maximal additive alpha value
    parser.add_argument('--normalize_data', default=True, type=bool)
    return parser


def list_of_tupels(args):
    lists_ = [x_.split(',') for x_ in args.replace(' ','').split(';')]
    tuples_ = list()
    for list_ in lists_:
        tuples_.append(tuple([int(x_) for x_ in list_]))
    return tuples_
