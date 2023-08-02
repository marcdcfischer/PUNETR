import pathlib as plb
import csv
import nibabel as nib
import numpy as np
from src.data.datasets.pre_processing import _rescale
from p_tqdm import p_map


def _process(path_img,
             paths_lbl,
             dirs_out,
             view: bool = False):
    nii_image, niis_lbl = nib.load(path_img), [nib.load(x_) for x_ in paths_lbl]
    np_image = nii_image.get_fdata(dtype=np.float32)
    np_labels = [x_.get_fdata(dtype=np.float32).round().astype(np.int32) for x_ in niis_lbl]
    print(f'Encountering shapes {np_image.shape, [x_.shape for x_ in np_labels]}.')

    if view:
        viewer = nib.viewers.OrthoSlicer3D(np.stack([np_image / 255. * 1., *np_labels], axis=-1))
        viewer.show()

    # Label encoding
    bg_label = np.full_like(np_labels[0], fill_value=0.5)
    mask_1 = np.argmax(np.stack([bg_label, *np_labels[:5]], axis=0), axis=0)
    mask_2 = np.argmax(np.stack([bg_label, np.sum(np.stack(np_labels[:5], axis=0), axis=0), *np_labels[5:]], axis=0), axis=0)
    mask_3 = np.argmax(np.stack([bg_label, *np_labels], axis=0), axis=0)

    if view:
        viewer = nib.viewers.OrthoSlicer3D(np.stack([np_image / 255. * 5., mask_1, mask_2], axis=-1))
        viewer.show()

    if all([np.count_nonzero(x_) for x_ in np_labels]):
        nii_image_converted = nib.Nifti1Image(np_image, nii_image.affine, nii_image.header)
        nii_lbl_1_converted = nib.Nifti1Image(mask_1,  nii_image.affine, nii_image.header)
        nii_lbl_2_converted = nib.Nifti1Image(mask_2,  nii_image.affine, nii_image.header)
        nii_lbl_3_converted = nib.Nifti1Image(mask_3, nii_image.affine, nii_image.header)
        nib.save(nii_image_converted, plb.Path(dirs_out[0]) / (path_img.parent.name + '_img.nii.gz'))
        nib.save(nii_image_converted, plb.Path(dirs_out[1]) / (path_img.parent.name + '_img.nii.gz'))
        nib.save(nii_lbl_1_converted, plb.Path(dirs_out[0]) / (path_img.parent.name + '_lbl.nii.gz'))
        nib.save(nii_lbl_2_converted, plb.Path(dirs_out[1]) / (path_img.parent.name + '_lbl.nii.gz'))
        nib.save(nii_lbl_3_converted, plb.Path(dirs_out[2]) / (path_img.parent.name + '_lbl_all.nii.gz'))
        print(f'Saved valid subject {path_img.parent.name}.')
        return True
    else:
        print(f'Skipped invalid subject {path_img.parent.name}.')
        return False


if __name__ == '__main__':
    # data from https://zenodo.org/record/1169361#.YqhgFXhBxhH
    dir_data = '/my/data/Totalsegmentator_dataset'
    dirs_out = ['/my/data/processed_tseg_lobes/', '/my/data/processed_tseg_abdomen/', '/my/data/processed_tseg_foreground/']
    [plb.Path(x_).mkdir(parents=True, exist_ok=True) for x_ in dirs_out]
    target_resolution = (1.25, 1.25, 2.5)
    target_size = None  # (280, 280, -1)
    minimum_size = (160, 160, 32)
    view = False
    overwrite = True

    # Preprocess data - cropping via csv
    blacklist = ['s0002']  # for unfitting pictures
    eligible_subjects = dict()
    with open(plb.Path(dir_data) / 'meta.csv', mode='r', encoding='utf-8-sig') as file_:
        reader = csv.DictReader(file_, delimiter=';')
        for row in reader:
            if (('thorax' in row['study_type'] and 'abdomen' in row['study_type']) or 'whole body' in row['study_type']) and not 'angiography' in row['study_type'] and row['image_id'] not in blacklist:
                eligible_subjects[row['image_id']] = row

    seg_lobes = ['lung_lower_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right']  # 5
    seg_abdomen = ['liver', 'spleen', 'pancreas', 'kidney_left', 'kidney_right']  # 5 (+1 for whole lung)
    paths_imgs, paths_lbls = [], []
    def _process_single(subj_):
        print(f'--- Processing {subj_} with image region {eligible_subjects[subj_]["study_type"]} ---')
        path_img = plb.Path(dir_data) / subj_ / 'ct.nii.gz'
        paths_lbl = [plb.Path(dir_data) / subj_ / 'segmentations' / f'{x_}.nii.gz' for x_ in seg_lobes + seg_abdomen]

        if overwrite:
            valid = _process(path_img=path_img,
                             paths_lbl=paths_lbl,
                             dirs_out=dirs_out,
                             view=view)

            if valid:
                for dir_out_ in dirs_out[:2]:
                    _rescale(path_images=[str(plb.Path(dir_out_) / (subj_ + '_img.nii.gz'))],
                             path_mask=str(plb.Path(dir_out_) / (subj_ + '_lbl.nii.gz')),
                             path_mask_foreground=str(plb.Path(dirs_out[2]) / (subj_ + '_lbl_all.nii.gz')),
                             target_resolution=target_resolution,
                             target_size=target_size,
                             minimum_size=minimum_size,
                             view=view)

    p_map(_process_single,
          list(eligible_subjects.keys()),
          num_cpus=0.5)
