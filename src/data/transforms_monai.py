from typing import Optional, Tuple, List, Type, Sequence
import torch
import monai.transforms as mtransforms
import itertools
import pathlib as plb
import warnings


# See https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unetr_btcv_segmentation_3d_lightning.ipynb for a recent example
def generate_transforms(patch_size_students: List[Tuple[int, int, int]],
                        patch_size_teacher: Optional[Tuple[int, int, int]] = None,
                        content_keys: Optional[List[str]] = None,
                        aux_keys: Optional[List[str]] = None,
                        num_samples: int = 2,  # Different slice crops per volume
                        n_transforms: int = 1,  # Different transforms of cropped volumes
                        a_min: float = -1000,
                        a_max: float = 1000,
                        b_min: float = 0,
                        b_max: float = 1,
                        variant: int = 0,
                        orientation: str = 'xy',
                        intensity_inversion: bool = False,
                        augmentation_mode: str = '2d',
                        shape_z: int = 1,
                        orientation_augmentation: bool = False,
                        crop_foreground: bool = False,
                        crop_label: bool = False,
                        crop_label_fine: bool = False,
                        label_indices: Sequence[int] = (),
                        normalize: bool = False):
    patch_size_teacher = patch_size_teacher if patch_size_teacher is not None else patch_size_students[0]
    if content_keys is None:
        content_keys = ['image', 'label']

    # Original keys are considered student keys
    content_keys_trafos = [[key_ + f'_trafo{str(idx_trafo)}' for key_ in content_keys] for idx_trafo in range(1, n_transforms)]
    aux_keys_trafos = [[key_ + f'_trafo{str(idx_trafo)}' for key_ in aux_keys] for idx_trafo in range(1, n_transforms)]
    content_keys_all_student_first = [content_keys] + content_keys_trafos  # list of lists (so every sample can be transformed independently)
    aux_keys_all_student_first = [aux_keys] + aux_keys_trafos

    # Students extended keys (e.g. smaller variant)
    # Following scheme [idx_student][idx_trafo][key]
    n_students = len(patch_size_students)
    content_keys_students = [content_keys]
    content_keys_all_students = [content_keys_all_student_first]
    aux_keys_all_students = [aux_keys_all_student_first]
    for idx_student in range(0, n_students - 1):
        content_keys_students.append([key_ + f'_var{str(idx_student)}' for key_ in content_keys])  # Student content keys without trafo
        content_keys_all_students.append([[key_ + f'_var{str(idx_student)}' for key_ in content_keys_] for content_keys_ in content_keys_all_student_first])  # Student content keys with trafo
        aux_keys_all_students.append([[key_ + f'_var{str(idx_student)}' for key_ in aux_keys_] for aux_keys_ in aux_keys_all_student_first])

    # Teacher keys
    content_keys_teacher = [key_ + '_teacher' for key_ in content_keys]  # [key]
    content_keys_teacher_clean = [key_ + '_clean' for key_ in content_keys_teacher]
    content_keys_all_teacher = [[key_ + '_teacher' for key_ in content_keys_] for content_keys_ in content_keys_all_student_first]  # [idx_trafo][key]
    content_keys_all_teacher_clean = [[key_ + '_clean' for key_ in content_keys_] for content_keys_ in content_keys_all_teacher]
    aux_keys_all_teacher = [[key_ + '_teacher' for key_ in aux_keys_] for aux_keys_ in aux_keys_all_student_first]  # [idx_trafo][key]

    # All keys
    content_keys_all = content_keys_all_students + [content_keys_all_teacher]  # [idx_student / teacher][idx_trafo][key]
    aux_keys_all = aux_keys_all_students + [aux_keys_all_teacher]  # [idx_student / teacher][idx_trafo][key]

    # Different augmentation variants
    if variant == 0:
        crop_weights = (0.33, 0.66)
        drop_weights = (0.6, 0.15, 0.15, 0.1)
        prob_rand_adjustments = 0.05
        prob_affine = 0.8
        additional_augs = True
        rotate_value_xy_student = 0.3 if patch_size_students[-1][2] > 1 else 0.
        shear_value_xy_student = 0.015 if patch_size_students[-1][2] > 1 else 0.
        scale_range_xy_student = (-0.15, 0.15) if patch_size_students[-1][2] > 1 else (0., 0.)
        rotate_value_z_student = 0.2 if patch_size_students[-1][2] > 1 else 0.
        shear_value_z_student = 0.015 if patch_size_students[-1][2] > 1 else 0.
        scale_range_z_student = (-0.1, 0.1) if patch_size_students[-1][2] > 1 else (0., 0.)
    else:
        raise ValueError

    # Pre-processing
    transform_train = mtransforms.Compose([
        # Image loading, normalization and (sub-)selection
        mtransforms.LoadImaged(keys=[x_ for x_ in content_keys if 'grid' not in str(x_)]),  # Exclude tensors from loading
        mtransforms.EnsureChannelFirstd(keys=[x_ for x_ in content_keys if 'grid' not in str(x_)]),   # Grid already has a channel (and does not have metadata for a check)
        # mtransforms.Orientation(axcodes='RAS')  # Images are already pre-processed with orientation RAS
        mtransforms.NormalizeIntensityd(keys=[x_ for x_ in content_keys if 'image' in str(x_)]) if normalize else \
        mtransforms.ScaleIntensityRanged(keys=[x_ for x_ in content_keys if 'image' in str(x_)], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),  # CT only. -1000 / 1000 / 0 / 1 according to monai example code. Other code snippets used -175 / 250
        # mtransforms.EnsureTyped(keys=[x_ for x_ in content_keys if 'grid' not in str(x_)], track_meta=False),  # Disable monai's new meta tensors
    ])

    if crop_foreground:
        transform_train = mtransforms.Compose([
            transform_train,
            mtransforms.OneOf([
                mtransforms.Compose([]),  # Dummy for applying nothing
                mtransforms.CropForegroundd(keys=[x_ for x_ in content_keys], source_key='image', select_fn=lambda x: x > 0, margin=(32, 32, 16)),  # Clip any (lower) clipped content (if any).
            ], weights=crop_weights),
        ])
    if crop_label:  # Could be extended to label indices downstream active
        if crop_label_fine and len(label_indices) > 0:
            def _select_fn(x):
                mask = x == label_indices[0]
                for idx_ in label_indices[1:]:
                    mask = torch.logical_or(mask, x == idx_)
                if not torch.any(mask == True).item():
                    mask = x > 0
                    warnings.warn(f'Training subject {x._meta["filename_or_obj"]} does not contain the target class.')
                return mask
        else:
            def _select_fn(x):
                mask = x > 0
                assert torch.any(mask == True).item()
                return mask
        transform_train = mtransforms.Compose([
            transform_train,
            mtransforms.OneOf([
                mtransforms.Compose([]),  # Dummy for applying nothing
                mtransforms.CropForegroundd(keys=[x_ for x_ in content_keys], source_key='label', select_fn=_select_fn, margin=(32, 32, 16)),  # Clip any (lower) clipped content (if any).
            ], weights=crop_weights),
        ])

    # Rotate orientation
    if orientation == 'xy':
        pass
    elif orientation == 'zy':
        transform_train = mtransforms.Compose([
            transform_train,
            mtransforms.Rotate90d(keys=content_keys, k=1, spatial_axes=(0, 2)),  # Rotate xz (as orientation augmentation) so resulting slices contain zy
        ])
    elif orientation == 'xz':
        transform_train = mtransforms.Compose([
            transform_train,
            mtransforms.Rotate90d(keys=content_keys, k=1, spatial_axes=(1, 2)),  # Rotate yz (as orientation augmentation) so resulting slices contain xz
        ])
    else:
        raise ValueError(f'Orientation {orientation} is not a valid choice.')

    # Generate samples
    transform_train = mtransforms.Compose([
        transform_train,
        mtransforms.RandSpatialCropSamplesd(keys=content_keys, roi_size=(patch_size_teacher[0], patch_size_teacher[1], patch_size_teacher[2]), random_center=True, random_size=False, num_samples=num_samples),  # Generates num_samples different slices
        mtransforms.CopyItemsd(keys=content_keys + aux_keys, times=n_transforms - 1, names=list(itertools.chain(*(content_keys_trafos + aux_keys_trafos)))) if n_transforms > 1 else mtransforms.Compose([]),  # Copies selected slices (and auxiliary info) for different augmentations
        mtransforms.CopyItemsd(keys=list(itertools.chain(*(content_keys_all_student_first + aux_keys_all_student_first))), times=1,
                               names=list(itertools.chain(*(content_keys_all_teacher + aux_keys_all_teacher)))),
    ])

    # Add further student samples
    for idx_student in range(1, n_students):
        transform_train = mtransforms.Compose([
            transform_train,
            mtransforms.CopyItemsd(keys=list(itertools.chain(*(content_keys_all_student_first + aux_keys_all_student_first))), times=1,
                                   names=list(itertools.chain(*(content_keys_all_students[idx_student] + aux_keys_all_students[idx_student])))),
        ])

    # Masking - see https://github.com/Project-MONAI/tutorials/tree/master/self_supervised_pretraining for more
    # Only applied to student.
    # dropout_holes = True -> replaces values inside region. dropout_holes = False -> replaces values outside region
    # Students (large and small) specific augmentations
    for idx_student in range(n_students):
        for idx_trafo in range(n_transforms):
            transform_train = mtransforms.Compose([
                transform_train,
                mtransforms.OneOf([
                    mtransforms.Compose([]),  # Dummy for applying nothing
                    mtransforms.RandCoarseDropoutd(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=1.0,
                                                   dropout_holes=True, holes=1, max_holes=5,
                                                   spatial_size=(5, 5, min(3, patch_size_students[idx_student][2])), max_spatial_size=(20, 20, min(10, patch_size_students[idx_student][2]))),  # spatial_size=20, max_spatial_size=40 used for a visualization example
                    mtransforms.RandCoarseShuffled(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=1.0,
                                                   holes=1, max_holes=3,
                                                   spatial_size=(5, 5, min(3, patch_size_students[idx_student][2])), max_spatial_size=(20, 20, min(10, patch_size_students[idx_student][2]))),
                    mtransforms.RandCoarseDropoutd(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=1.0,
                                                   dropout_holes=False, holes=1, max_holes=5,
                                                   spatial_size=(48, 48, min(16, patch_size_students[idx_student][2])), max_spatial_size=(96, 96, min(24, patch_size_students[idx_student][2]))),  # Holes are aggregated prior to outer fill.
                ], weights=drop_weights)  # (0.0, 1.0, 0.0, 0.0) used for a visualization example
            ])

    # Augmentations on (pre-)crop
    # Students (large and small) are augmented differently
    for idx_student in range(n_students):
        for idx_trafo in range(n_transforms):
            if additional_augs:
                transform_train = mtransforms.Compose([
                    transform_train,
                    mtransforms.RandStdShiftIntensityd(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=prob_rand_adjustments, factors=(-0.1, 0.1)),
                ])
            transform_train = mtransforms.Compose([
                transform_train,
                mtransforms.RandAdjustContrastd(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=prob_rand_adjustments, gamma=(0.9, 1.1)),
                mtransforms.RandScaleIntensityd(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=prob_rand_adjustments, factors=-2.) if intensity_inversion else mtransforms.Compose([]),  # Invert image (v = v * (1 + factor))
                mtransforms.RandScaleIntensityd(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=prob_rand_adjustments, factors=(-0.1, 0.1)),
                mtransforms.RandHistogramShiftd(keys=[x_ for x_ in content_keys_all_students[idx_student][idx_trafo] if 'image' in str(x_)], prob=prob_rand_adjustments, num_control_points=(8, 12)),  # Shifts around image histogram
                mtransforms.RandAffined(keys=content_keys_all_students[idx_student][idx_trafo],
                                        prob=prob_affine,
                                        rotate_range=(rotate_value_z_student, rotate_value_z_student, rotate_value_xy_student),  # 3D rotate in radians
                                        shear_range=(shear_value_xy_student, shear_value_z_student, shear_value_xy_student, shear_value_z_student, shear_value_z_student, shear_value_z_student),  # 3D shear
                                        scale_range=(scale_range_xy_student, scale_range_xy_student, scale_range_z_student),  # 3D scaling. For some reason they add 1.0 internally ...
                                        mode=['nearest' if 'label' in key_ or 'pseudo' in key_ else 'bilinear' for key_ in content_keys_all_students[idx_student][idx_trafo]],
                                        padding_mode='reflection'),
            ])
    # Teacher
    rotate_value_z = 0.15 if patch_size_students[-1][2] > 1 else 0.
    shear_value_z = 0.01 if patch_size_students[-1][2] > 1 else 0.
    scale_range_z = (-0.075, 0.075) if patch_size_students[-1][2] > 1 else (0., 0.)
    for idx_trafo in range(n_transforms):
        transform_train = mtransforms.Compose([
            transform_train,
            mtransforms.CopyItemsd(keys=content_keys_all_teacher[idx_trafo], times=1, names=content_keys_all_teacher_clean[idx_trafo]),
        ])
        if additional_augs:
            transform_train = mtransforms.Compose([
                transform_train,
                mtransforms.RandStdShiftIntensityd(keys=[x_ for x_ in content_keys_all_teacher[idx_trafo] if 'image' in str(x_)], prob=prob_rand_adjustments, factors=(-0.075, 0.075)),
            ])
        transform_train = mtransforms.Compose([
            transform_train,
            mtransforms.RandAdjustContrastd(keys=[x_ for x_ in content_keys_all_teacher[idx_trafo] if 'image' in str(x_)], prob=prob_rand_adjustments, gamma=(0.95, 1.05)),
            mtransforms.RandScaleIntensityd(keys=[x_ for x_ in content_keys_all_teacher[idx_trafo] if 'image' in str(x_)], prob=prob_rand_adjustments, factors=(-0.05, 0.05)),
            mtransforms.RandHistogramShiftd(keys=[x_ for x_ in content_keys_all_teacher[idx_trafo] if 'image' in str(x_)], prob=prob_rand_adjustments, num_control_points=(10, 14)),  # Fixed number of control points
            mtransforms.RandAffined(keys=content_keys_all_teacher[idx_trafo],
                                    prob=prob_affine,
                                    rotate_range=(rotate_value_z, rotate_value_z, 0.15),
                                    shear_range=(0.01, shear_value_z, 0.01, shear_value_z, shear_value_z, shear_value_z),
                                    scale_range=((-0.1, 0.1), (-0.1, 0.1), scale_range_z),
                                    mode=['nearest' if 'label' in key_ or 'pseudo' in key_ else 'bilinear' for key_ in content_keys_all_teacher[idx_trafo]],
                                    padding_mode='reflection'),
        ])

    # Final conversion
    # Students
    for idx_student in range(n_students):
        for idx_trafo in range(n_transforms):
            transform_train = mtransforms.Compose([
                transform_train,
                mtransforms.RandSpatialCropd(keys=content_keys_all_students[idx_student][idx_trafo], roi_size=patch_size_students[idx_student], random_center=True, random_size=False),  # Crop center of augmented patch
                mtransforms.SpatialPadd(keys=content_keys_all_students[idx_student][idx_trafo], spatial_size=patch_size_students[idx_student], mode='reflect'),
                mtransforms.ToTensord(keys=content_keys_all_students[idx_student][idx_trafo], track_meta=False)
            ])
    # Teacher
    for idx_trafo in range(n_transforms):
        transform_train = mtransforms.Compose([
            transform_train,
            mtransforms.RandSpatialCropd(keys=content_keys_all_teacher[idx_trafo], roi_size=patch_size_teacher, random_center=True, random_size=False),  # Crop center of augmented patch
            mtransforms.RandSpatialCropd(keys=content_keys_all_teacher_clean[idx_trafo], roi_size=patch_size_teacher, random_center=True, random_size=False),
            mtransforms.SpatialPadd(keys=content_keys_all_teacher[idx_trafo], spatial_size=patch_size_teacher, mode='reflect'),
            mtransforms.SpatialPadd(keys=content_keys_all_teacher_clean[idx_trafo], spatial_size=patch_size_teacher, mode='reflect'),
            mtransforms.ToTensord(keys=content_keys_all_teacher[idx_trafo], track_meta=False),
            mtransforms.ToTensord(keys=content_keys_all_teacher_clean[idx_trafo], track_meta=False),
        ])

    # Join transformed elements (along channel dim) - so it doesn't need to perform within the training / validation routine
    # Students
    for idx_student in range(n_students):
        for key_, keys_student_ in zip(content_keys, content_keys_students[idx_student]):
            applied_student_keys_ = [x_ for x_ in list(itertools.chain(*content_keys_all_students[idx_student])) if key_ in str(x_)]
            if len(applied_student_keys_) > 0:
                transform_train = mtransforms.Compose([transform_train, mtransforms.ConcatItemsd(keys=applied_student_keys_, name=keys_student_)])
    # Teacher
    for key_, key_teacher_ in zip(content_keys, content_keys_teacher):
        applied_teacher_keys = [x_ for x_ in list(itertools.chain(*content_keys_all_teacher)) if key_ in str(x_)]
        if len(applied_teacher_keys) > 0:
            transform_train = mtransforms.Compose([transform_train, mtransforms.ConcatItemsd(keys=applied_teacher_keys, name=key_teacher_)])
    for key_, key_teacher_ in zip(content_keys, content_keys_teacher_clean):
        applied_teacher_keys_clean_ = [x_ for x_ in list(itertools.chain(*content_keys_all_teacher_clean)) if key_ in str(x_)]
        if len(applied_teacher_keys_clean_) > 0:
            transform_train = mtransforms.Compose([transform_train, mtransforms.ConcatItemsd(keys=applied_teacher_keys_clean_, name=key_teacher_)])

    # Discard obsolete additional keys and meta data (of additional trafos and variants)
    transform_train = mtransforms.Compose([
        transform_train,
        mtransforms.SelectItemsd(keys=list(itertools.chain(*content_keys_students)) + content_keys_teacher + content_keys_teacher_clean + aux_keys)
    ])

    # Validation
    transform_val = mtransforms.Compose([
        mtransforms.LoadImaged(keys=[x_ for x_ in content_keys if 'grid' not in str(x_)]),
        mtransforms.EnsureChannelFirstd(keys=[x_ for x_ in content_keys if 'grid' not in str(x_)]),
        mtransforms.NormalizeIntensityd(keys=[x_ for x_ in content_keys if 'image' in str(x_)]) if normalize else \
        mtransforms.ScaleIntensityRanged(keys=[x_ for x_ in content_keys if 'image' in str(x_)], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),  # CT only
        mtransforms.ToTensord(keys=content_keys, track_meta=False),
        mtransforms.SelectItemsd(keys=content_keys + aux_keys)  # Discard (currently) unused meta data
    ])

    return transform_train, transform_val


def generate_test_transforms(content_keys: Optional[List[str]] = None,
                             aux_keys: Optional[List[str]] = None,
                             a_min: float = -1000,
                             a_max: float = 1000,
                             b_min: float = 0,
                             b_max: float = 1,
                             normalize: bool = False):
    if content_keys is None:
        content_keys = ['image', 'label']

    # Validation
    transform_test = mtransforms.Compose([
        mtransforms.LoadImaged(keys=[x_ for x_ in content_keys if 'grid' not in str(x_)]),
        mtransforms.EnsureChannelFirstd(keys=[x_ for x_ in content_keys if 'grid' not in str(x_)]),
        mtransforms.NormalizeIntensityd(keys=[x_ for x_ in content_keys if 'image' in str(x_)]) if normalize else \
        mtransforms.ScaleIntensityRanged(keys=[x_ for x_ in content_keys if 'image' in str(x_)], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),  # CT only
        # mtransforms.EnsureTyped(keys=[x_ for x_ in content_keys if 'grid' not in str(x_)], track_meta=False),  # Disable monai's new meta tensors
        mtransforms.ToTensord(keys=content_keys, track_meta=True),
    ])

    return transform_test


def generate_test_post_transforms(output_dir: str,
                                  output_postfix: str,
                                  transform_test: mtransforms.InvertibleTransform,
                                  n_classes: Optional[int] = None):

    # Create output directory (if it doesn't exist)
    plb.Path(output_dir).mkdir(parents=True, exist_ok=True)

    transform_test_post = mtransforms.Compose([
        mtransforms.Invertd(
            keys="pred",
            transform=transform_test,
            orig_keys="label",
            meta_keys="pred_meta_dict",
            orig_meta_keys="label_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        mtransforms.AsDiscreted(keys="pred", argmax=True, to_onehot=n_classes),
        mtransforms.EnsureTyped(keys="pred", track_meta=False),
        mtransforms.SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir=output_dir, output_postfix=output_postfix, resample=False),  # Use image_meta_dict since label_meta_dict filename may not be unique.
    ])

    return transform_test_post
