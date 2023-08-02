import dicom2nifti
import nibabel as nib
from monai import transforms as mtransforms
from nibabel.orientations import ornt_transform, axcodes2ornt, inv_ornt_aff, apply_orientation, io_orientation, aff2axcodes
from typing import Tuple, Optional, List
import numpy as np
import pathlib as plb

from skimage import transform as stransform


def reorient_nii(nib_image: nib.Nifti1Image,
                 affine: Optional[np.array] = None,
                 axes_code: Tuple[str, ...] = ('L', 'A', 'S'),
                 verbose: bool = True):

    volume_old = nib_image.get_fdata()
    affine_old = affine if affine is not None else nib_image.affine  # Use passed affine (e.g. to overwrite disparate affine in mask)
    target_orientation = axcodes2ornt(axes_code)
    ornt_trans = ornt_transform(io_orientation(affine_old), target_orientation)
    affine_trans = inv_ornt_aff(ornt_trans, volume_old.shape)
    affine_new = np.dot(affine_old, affine_trans)
    volume_new = apply_orientation(volume_old, ornt_trans)
    if verbose:
        print(f'Orientation change: {aff2axcodes(affine_old)} -> {aff2axcodes(affine_new)}')
        print(f'Affine change:\n{affine_old}\n->\n{affine_new}')
    nib_image_new = nib.Nifti1Image(volume_new, affine_new, nib_image.header)

    return nib_image_new


def reorient_acdc(dir_images: str,
                  dir_masks: str,
                  dir_scribbles: str,
                  dir_images_out: str,
                  dir_masks_out: str,
                  dir_scribbles_out: str):

    path_images = sorted(plb.Path(dir_images).rglob('*[0-9].nii.gz'))
    path_masks = sorted(plb.Path(dir_masks).rglob('*_gt.nii.gz'))
    path_scribbles = sorted(plb.Path(dir_scribbles).rglob('*_scribble.nii.gz'))

    assert len(path_images) == len(path_masks) and len(path_images) == len(path_scribbles)
    [plb.Path(x_).mkdir(parents=True, exist_ok=True) for x_ in [dir_images_out, dir_masks_out, dir_scribbles_out]]

    for path_img_, path_mask_, path_scribbles_ in zip(path_images, path_masks, path_scribbles):
        print(f'Fixing subject {path_img_.name} with mask {path_mask_.name} and scribbles {path_scribbles_.name}.')
        nib_image_ = nib.load(path_img_)
        nib_mask_ = nib.load(path_mask_)
        nib_scribbles = nib.load(path_scribbles_)
        common_affine = nib_image_.affine
        nib_image_new = reorient_nii(nib_image=nib_image_, affine=common_affine)
        nib_mask_new = reorient_nii(nib_image=nib_mask_, affine=common_affine)
        nib_scribbles_new = reorient_nii(nib_image=nib_scribbles, affine=common_affine)

        path_out_image_ = plb.Path(dir_images_out) / path_img_.relative_to(plb.Path(dir_images))
        path_out_mask_ = plb.Path(dir_masks_out) / path_mask_.relative_to(plb.Path(dir_masks))
        path_out_scribbles_ = plb.Path(dir_scribbles_out) / path_scribbles_.relative_to(plb.Path(dir_scribbles))

        nib.save(nib_image_new, path_out_image_)
        nib.save(nib_mask_new, path_out_mask_)
        nib.save(nib_scribbles_new, path_out_scribbles_)


def _convert_dcm_to_nifti(dir_in: str,
                          path_out: str):

    print(f'Converting {dir_in} to {path_out}.')
    dicom2nifti.dicom_series_to_nifti(dir_in, path_out, reorient_nifti=False)


def _rescale(path_images: List[str],
             path_mask: str = '',
             path_mask_foreground: str = '',
             path_scribbles: str = '',
             target_resolution: Tuple[float, ...] = (1.0, 1.0, 1.0),
             target_size: Tuple[int, ...] | None = (256, 256, -1),
             minimum_size: Tuple[int, ...] = (160, 160, 64),
             rescale_z: bool = True,
             view: bool = False):

    crop_foreground = len(path_mask_foreground) > 0
    nii_images = [nib.load(x_) for x_ in path_images]

    np_images = [x_.get_fdata() for x_ in nii_images]
    np_mask = None
    if path_mask:
        nii_mask = nib.load(path_mask)
        np_mask = nii_mask.get_fdata(dtype=np.float32).round().astype(np.int32)
    np_scribbles = None
    if path_scribbles:
        nii_scribbles = nib.load(path_scribbles)
        np_scribbles = nii_scribbles.get_fdata()

    if view:
        viewer = nib.viewers.OrthoSlicer3D(np.stack([np_images[0] / 255. * 10, np_mask], axis=-1))
        viewer.show()

    pix_dim = nii_images[0].header['pixdim'][1:4]
    scale = [x_ / y_ for x_, y_ in zip(pix_dim, target_resolution)]
    scale[2] = 1. if not rescale_z else scale[2]
    new_affine = nii_images[0].affine
    new_affine[0, 0] = new_affine[0, 0] / scale[0]  # Note: all (existing) processed files have probably a wrong spacing info in the header (03.09.22)
    new_affine[1, 1] = new_affine[1, 1] / scale[1]
    new_affine[2, 2] = new_affine[2, 2] / scale[2]

    if crop_foreground:
        lbl_foreground, meta_lbl_ = mtransforms.LoadImage()(path_mask_foreground)
        lbl_foreground = mtransforms.EnsureChannelFirst()(lbl_foreground, meta_dict=meta_lbl_)
        lbl_foreground = mtransforms.Orientation(axcodes='RAS')(lbl_foreground)
        lbl_foreground = mtransforms.Spacing(pixdim=target_resolution, mode='nearest', padding_mode='reflection', recompute_affine=True)(lbl_foreground)

        def _mask_selector(x):
            return lbl_foreground > 0

    for idx_img in range(len(path_images)):
        img_, meta_ = mtransforms.LoadImage()(path_images[idx_img])
        original_image_affine = np.array(img_.affine, copy=True)
        img_ = mtransforms.EnsureChannelFirst()(img_, meta_dict=meta_)
        img_ = mtransforms.Orientation(axcodes='RAS')(img_)
        img_ = mtransforms.Spacing(pixdim=target_resolution, mode='bilinear', padding_mode='reflection', recompute_affine=True)(img_)
        img_ = mtransforms.CropForeground(select_fn=_mask_selector, margin=(32, 32, 16))(img_) if crop_foreground else img_
        img_ = mtransforms.SpatialPad(spatial_size=minimum_size, mode='constant', constant_values=-1000)(img_)  # Pad content if there is a very small subject
        # img_ = mtransforms.ResizeWithPadOrCrop(spatial_size=target_size, mode='constant', constant_values=-1000)(img_)[0]
        new_image = img_[0]
        new_image_affine = new_image.affine

        if view:
            viewer = nib.viewers.OrthoSlicer3D(new_image)
            viewer.show()

        print(f'Writing scaled image with target resolution: {target_resolution}, original: {pix_dim}, resulting: {new_image.pixdim} \n'
              f'and size {np_images[0].shape} -> {new_image.shape} \n'
              f'for {path_images[idx_img]} \n'
              f'with affine \n {new_image_affine} \n')
        nii_image_converted = nib.Nifti1Image(np.array(new_image).astype(np.float32), np.array(new_image_affine), nii_images[0].header)
        nib.save(nii_image_converted, path_images[idx_img])

    if np_mask is not None:
        img_, meta_ = mtransforms.LoadImage()(path_mask)
        img_.affine = original_image_affine  # Use image affine since the one in the mask is often broken.
        img_ = mtransforms.EnsureChannelFirst()(img_, meta_dict=meta_)
        img_ = mtransforms.Orientation(axcodes='RAS')(img_)
        img_ = mtransforms.Spacing(pixdim=target_resolution, mode='nearest', padding_mode='reflection', recompute_affine=True)(img_)
        img_ = mtransforms.CropForeground(select_fn=_mask_selector, margin=(32, 32, 16))(img_) if crop_foreground else img_
        img_ = mtransforms.SpatialPad(spatial_size=minimum_size, mode='constant', constant_values=0)(img_)  # Pad content if there is a very small subject
        # img_ = mtransforms.ResizeWithPadOrCrop(spatial_size=target_size, mode='constant', constant_values=-1000)(img_)[0]
        new_mask = img_[0]
        new_mask_affine = new_mask.affine
        if not np.allclose(new_mask_affine, new_image_affine):
            print('Encountered different affines.')
            print(new_mask.affine)
            print(new_image_affine)
            raise ValueError

        nii_mask_converted = nib.Nifti1Image(np.array(new_mask).astype(np.int32), np.array(new_mask_affine), nii_images[0].header)
        nib.save(nii_mask_converted, path_mask)

    if np_scribbles is not None:
        np_scribbles_rescaled = stransform.rescale(np_scribbles,
                                                   scale,
                                                   order=0,
                                                   preserve_range=True,
                                                   anti_aliasing=False,
                                                   mode='constant',
                                                   cval=0)
        np_scribbles_resized = mtransforms.ResizeWithPadOrCrop(spatial_size=target_size, mode='constant', constant_values=0)(np.expand_dims(np_scribbles_rescaled, 0))[0]
        nii_scribbles_converted = nib.Nifti1Image(np_scribbles_resized.astype(np.int32), new_affine, nii_images[0].header)
        nib.save(nii_scribbles_converted, path_scribbles)

    if view:
        viewer = nib.viewers.OrthoSlicer3D(np.stack([new_image / 255. * 10, new_mask], axis=-1))
        viewer.show()
