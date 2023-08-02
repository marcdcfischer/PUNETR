from argparse import ArgumentParser, Namespace
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler
from typing import Optional, Any, Dict, Union, List, Tuple
import pathlib as plb
from src.modules.architectures.momentum_model_simple import MomentumModelSimple
from src.modules.losses.contrastive_protos_teacher import ContrastiveProtosTeacherLoss
from src.utils.plotting import image_grid
from src.modules.losses import focal
import wandb
import monai
import monai.inferers as minferers
import monai.losses as mlosses
from monai.data import decollate_batch
import einops
import torch.nn.functional as F
import itertools
from functools import partial
import numpy as np
from src.data.transforms_monai import generate_test_post_transforms
import nibabel as nib
import warnings


# This module is used for the baseline architectures (available through monai)
class InstructionModelSimple(pl.LightningModule):
    def __init__(self, conf: Union[Dict, Namespace], **kwargs):
        super().__init__()
        self.save_hyperparameters(conf)

        print(f'Establishing architecture with parameters: \n {self.hparams}')

        # Architecture
        self.architecture = MomentumModelSimple(conf=self.hparams)

        # Losses
        # Segmentation losses
        # Monai focal loss uses weird BCE-formulation without softmax
        self.loss_seg_dice = mlosses.DiceLoss(include_background=self.hparams.background_dice,
                                              to_onehot_y=False,
                                              softmax=True,
                                              squared_pred=True,
                                              jaccard=False,
                                              reduction="mean",
                                              smooth_nr=1e-5,
                                              smooth_dr=1e-5,
                                              batch=False)
        self.loss_seg_focal = focal.FocalLoss(self.hparams.out_channels,
                                              loss_weight=self.hparams.loss_weight_segmentation * self.hparams.loss_weight_segmentation_focal,
                                              gamma=self.hparams.loss_weight_segmentation_gamma,
                                              alpha_background=self.hparams.loss_weight_segmentation_alpha_background if not self.hparams.downstream else self.hparams.loss_weight_segmentation_alpha_background_downstream,
                                              alpha_foreground=self.hparams.loss_weight_segmentation_alpha_foreground,
                                              additive_alpha=self.hparams.additive_alpha,
                                              normalized=self.hparams.loss_segmentation_normalized)

        # Contrastive losses
        self.loss_cluster_pairs = ContrastiveProtosTeacherLoss(
            reduction_factor=self.hparams.reduction_factor,
            reduction_factor_protos=self.hparams.reduction_factor_protos,
            loss_weight=self.hparams.loss_weight_sim_protos if not self.hparams.downstream else self.hparams.loss_weight_sim_protos_downstream,
            k_means_iterations=self.hparams.k_means_iterations,
            use_weighting_protos=self.hparams.use_weighting_protos,
            use_weighting_teacher=self.hparams.use_weighting_teacher,
            fwhm_student_protos=self.hparams.fwhm_student_protos,
            fwhm_teacher_protos=self.hparams.fwhm_teacher_protos,
            temp_proto_teacher=self.hparams.temp_proto_teacher,
            temp_proto_student=self.hparams.temp_proto_student,
            max_dist_target=self.hparams.max_dist_target,
            scale_z=self.hparams.scale_z,
            normalized=self.hparams.normalize_protos,
        )

        # Metrics
        self.score_seg_train = monai.metrics.DiceMetric(include_background=True, reduction='mean_batch')
        self.score_seg_train_annotated = monai.metrics.DiceMetric(include_background=True, reduction='mean_batch')
        self.score_seg_train_non_annotated = monai.metrics.DiceMetric(include_background=True, reduction='mean_batch')
        self.score_seg_val = monai.metrics.DiceMetric(include_background=True, reduction='mean_batch')
        self.score_seg_test = monai.metrics.DiceMetric(include_background=True, reduction='mean_batch')
        self.score_loss_train = monai.metrics.LossMetric(lambda x: x, reduction="mean_batch")
        self.score_loss_val = monai.metrics.LossMetric(lambda x: x, reduction="mean_batch")
        self.score_loss_test = monai.metrics.LossMetric(lambda x: x, reduction="mean_batch")

    def forward(self,
                x: List[torch.Tensor],
                x_teacher: Optional[torch.Tensor] = None,
                teacher_prediction: bool = True,
                second_student_prediction: bool = True):

        x_teacher = x_teacher if teacher_prediction else None
        x = x if second_student_prediction else x[:1]
        dict_out_students, dict_out_teacher = self.architecture(x, x_teacher)

        return dict_out_students, dict_out_teacher

    def forward_prediction(self, x: torch.Tensor):  # takes an image and predicts one (as required for monai sliding window inference)

        dict_out_students, _ = self.architecture([x], None)

        return dict_out_students[0]['dense']['embedded_latents']

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        if self.hparams.plot_only:
            self.architecture.eval()

        if batch_idx == 0:
            torch.cuda.empty_cache()  # helps with fragmentation
            if self.current_epoch == 0:
                print(f'Training with optimizer(s) {self.optimizers()}')
            if self.hparams.downstream and self.lr_schedulers() is not None:
                print(f'Training epoch {self.current_epoch} with step size(s) {self.lr_schedulers().get_last_lr()}')
                for idx_step_size, step_size_ in enumerate(self.lr_schedulers().get_last_lr()):
                    self.log(f'train_step_size_{idx_step_size}', step_size_, sync_dist=True)

        # Decide if this batch content gets plotted
        plot_content = self.hparams.plot and (self.current_epoch + 1) % self.hparams.plot_interval_train == 0 and batch_idx % 10 == 0 and (batch_idx <= 20 or self.hparams.plot_only)
        batch = batch[0]

        # Batch preparations
        # Aux content for one of the categories (student large) only
        aux_names = list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'name' in str(x_)]))))  # custom collate on auxiliary keys, since monai can only concat arrays / tensors.
        aux_frames = list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'frame' in str(x_)]))))
        aux_annotated = torch.stack(list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'annotated' in str(x_)])))))  # custom collate on auxiliary keys, since monai can only concat arrays / tensors.

        # Image content
        x_students = [einops.rearrange(batch['image'], 'b (t c) h w d -> (b t) c h w d', t=self.hparams.num_transforms)]  # Concatenate differently transformed elements in batch dim
        y_students = [einops.rearrange(batch['label'], 'b (t c) h w d -> (b t) c h w d', t=self.hparams.num_transforms).float().round().long()[:, 0, ...]]
        for idx_student in range(self.hparams.n_students - 1):
            x_students.append(einops.rearrange(batch[f'image_var{idx_student}'], 'b (t c) h w d -> (b t) c h w d', t=self.hparams.num_transforms))
            y_students.append(einops.rearrange(batch[f'label_var{idx_student}'], 'b (t c) h w d -> (b t) c h w d', t=self.hparams.num_transforms).float().round().long()[:, 0, ...])
        x_teacher = einops.rearrange(batch['image_teacher'], 'b (t c) h w d -> (b t) c h w d', t=self.hparams.num_transforms)
        y_teacher = einops.rearrange(batch['label_teacher'], 'b (t c) h w d -> (b t) c h w d', t=self.hparams.num_transforms).float().round().long()[:, 0, ...]

        y_students_one_hot = [einops.rearrange(F.one_hot(x_, num_classes=self.hparams.out_channels), 'b h w d c -> b c h w d') for x_ in y_students]
        y_teacher_one_hot = einops.rearrange(F.one_hot(y_teacher, num_classes=self.hparams.out_channels), 'b h w d c -> b c h w d')

        coord_grids_students = None
        coord_grids_teacher = None
        if 'coord_grid' in batch.keys():
            coord_grids_students = [einops.rearrange(batch['coord_grid'], 'b (t c) h w d -> (b t) c h w d', t=self.hparams.num_transforms)]
            for idx_student in range(self.hparams.n_students - 1):
                coord_grids_students.append(einops.rearrange(batch[f'coord_grid_var{idx_student}'], 'b (t c) h w d -> (b t) c h w d', t=self.hparams.num_transforms))
            coord_grids_teacher = einops.rearrange(batch['coord_grid_teacher'], 'b (t c) h w d -> (b t) c h w d', t=self.hparams.num_transforms)

        # Momentum update
        self.architecture.update_teacher()

        # prediction
        dict_out_students, dict_out_teacher = self(x_students,
                                                   x_teacher,
                                                   teacher_prediction=self.loss_cluster_pairs.loss_weight > 0.,
                                                   second_student_prediction=True)

        # Loss calculation
        loss_dict, plots_dict = dict(), dict()
        # Segmentation
        for idx_student in range(len(dict_out_students)):
            if self.hparams.loss_weight_segmentation > 0.:
                loss_dict.update(
                    self.loss_seg_focal(dict_out_students[idx_student]['dense']['embedded_latents'][aux_annotated, ...],
                                        y_students[idx_student][aux_annotated, ...],
                                        tag=f'seg_focal_s{idx_student}'))
                loss_dict.update({f'seg_dice_s{idx_student}':
                                      self.hparams.loss_weight_segmentation_dice * self.hparams.loss_weight_segmentation * self.loss_seg_dice(
                                          dict_out_students[idx_student]['dense']['embedded_latents'][
                                              aux_annotated, ...],
                                          y_students_one_hot[idx_student][aux_annotated, ...])
                                  })
            else:
                loss_dict[f'seg_focal_s{idx_student}'] = torch.tensor(0., device=self.device, requires_grad=False)
                loss_dict[f'seg_dice_s{idx_student}'] = torch.tensor(0., device=self.device, requires_grad=False)

        # Self-supervised contrastive
        if self.hparams.contrastive_pairs:
            if self.loss_cluster_pairs.loss_weight > 0.:
                loss_tmp_, plots_tmp_ = self.loss_cluster_pairs(embeddings_students=[x_['patched']['embedded_latents'] for x_ in dict_out_students],
                                                                embeddings_teacher=dict_out_teacher['patched']['embedded_latents'],
                                                                frames=aux_frames,
                                                                coord_grids_students=coord_grids_students,
                                                                coord_grids_teacher=coord_grids_teacher,
                                                                generate_plots=plot_content)
                loss_dict.update(loss_tmp_), plots_dict.update(plots_tmp_)
            else:
                for idx_student in range(len(dict_out_students)):
                    loss_dict[f'contrastive_proxy_sim_clustered_s{idx_student}'] = torch.tensor(0., device=self.device, requires_grad=False)
        loss_dict['all'] = sum(loss_dict.values())

        with torch.no_grad():
            # metrics
            self.score_loss_train(loss_dict['all'])
            for idx_student in range(len(dict_out_students)):
                predictions_one_hot_ = F.one_hot(dict_out_students[idx_student]['dense']['embedded_latents'].argmax(dim=1).long(), num_classes=self.hparams.out_channels).permute(0, 4, 1, 2, 3)
                self.score_seg_train(y_pred=predictions_one_hot_,
                                     y=y_students_one_hot[idx_student])
                if any(aux_annotated):
                    self.score_seg_train_annotated(y_pred=predictions_one_hot_[aux_annotated, ...],
                                                   y=y_students_one_hot[idx_student][aux_annotated, ...])
                if any(~aux_annotated):
                    self.score_seg_train_non_annotated(y_pred=predictions_one_hot_[~aux_annotated, ...],
                                                       y=y_students_one_hot[idx_student][~aux_annotated, ...])

            # logging
            for key_, value_ in loss_dict.items():
                self.log(f'train_loss_{key_}', value_.detach().cpu(), prog_bar='all' in key_)

            # plotting
            if plot_content:
                if not dist.is_initialized() or dist.get_rank() < 1:
                    # Fetch valid elements (if sim of plots has only been calculated for valid elements)
                    valid_entries = np.array(aux_frames) == np.unique(aux_frames)[0]
                    png_paths = list()
                    for idx_student in range(len(dict_out_students)):
                        y_one_hot_list_ = [y_students_one_hot[idx_student], y_teacher_one_hot]
                        for identifier_, dict_, x_, y_one_hot_, in zip(['student', 'teacher'], [dict_out_students[idx_student], dict_out_teacher], [x_students[idx_student], x_teacher], y_one_hot_list_):
                            if dict_ is not None:
                                png_paths.extend(image_grid.plot_grid_middle(x_.detach().cpu(),
                                                                             y_one_hot_.detach().cpu(),
                                                                             torch.softmax(dict_['dense']['embedded_latents'].detach().cpu().float(), dim=1),
                                                                             None,
                                                                             indices_elements=[idx_ * 2 for idx_ in range(max(min(x_students[0].shape[0] // 2, 2), 1))],
                                                                             prefix=f'{self.hparams.run_name}_train_b{batch_idx}_s{idx_student}_ep{str(self.current_epoch).zfill(3)}_{identifier_}',
                                                                             path_plots=self.hparams.default_root_dir))
                    if not self.hparams.online_off:
                        png_paths_grid_teacher = [png_ for png_ in png_paths if "grid" in png_ and "teacher" in png_]
                        png_paths_grid_student = [png_ for png_ in png_paths if "grid" in png_ and "student" in png_]
                        if len(png_paths_grid_teacher) > 0:
                            self.loggers[1].log_image(key="grid_teacher", images=[wandb.Image(png_) for png_ in png_paths_grid_teacher], caption=[str(plb.Path(png_).stem) for png_ in png_paths_grid_teacher])
                        if len(png_paths_grid_student) > 0:
                            self.loggers[1].log_image(key="grid_student", images=[wandb.Image(png_) for png_ in png_paths_grid_student], caption=[str(plb.Path(png_).stem) for png_ in png_paths_grid_student])

        return {'loss': loss_dict['all']}

    def on_train_epoch_end(self) -> None:
        score_loss = self.score_loss_train.aggregate()
        self.score_loss_train.reset()
        self.log(f'train_loss_mean', score_loss, sync_dist=True)
        for scorer_, tag_ in zip([self.score_seg_train], ['all']):  # self.score_seg_train_annotated, self.score_seg_train_non_annotated], ['all', 'annotated', 'non_annotated']):
            try:
                score_seg = scorer_.aggregate()
                scorer_.reset()
                self.log_dict({f'train_dice_{tag_}_c{str(idx_c).zfill(2)}': score_seg[idx_c] for idx_c in range(len(score_seg))}, sync_dist=True)
                self.log(f'train_dice_{tag_}_mean', torch.mean(score_seg[1:]), sync_dist=True)
            except:
                pass

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        if batch_idx == 0:
            torch.cuda.empty_cache()  # helps with fragmentation

        # batch preparations
        aux_annotated = torch.stack(list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'annotated' in x_])))))  # custom collate on auxiliary keys, since monai can only concat arrays / tensors.
        x = batch['image']
        y = batch['label'].float().round().long()[:, 0, ...]
        y_one_hot = einops.rearrange(F.one_hot(y, num_classes=self.hparams.out_channels), 'b h w d c -> b c h w d')
        scribbles, scribbles_masked = None, None
        if 'scribbles' in batch.keys():
            scribbles = batch['scribbles'][:, 0, ...]
            scribbles_masked = scribbles.clone()
            scribbles_masked[~aux_annotated, ...] = self.hparams.out_channels  # Mask non-annotated

        # Activate respective instructions
        loss_dict_all = list()

        # Sliding Window Inference - by teacher
        volume_prediction = minferers.sliding_window_inference(inputs=x,
                                                               roi_size=self.hparams.patch_size_students[0],
                                                               sw_batch_size=self.hparams.batch_size,
                                                               predictor=partial(self.forward_prediction),
                                                               overlap=self.hparams.sliding_window_overlap,
                                                               mode='gaussian')

        # Loss calculation
        loss_dict, plots_dict = dict(), dict()
        # Segmentation
        if any(aux_annotated) and self.hparams.loss_weight_segmentation > 0.:
            loss_dict.update(self.loss_seg_focal(volume_prediction[aux_annotated, ...],
                                                 y[aux_annotated, ...]))
            loss_dict.update({f'seg_dice':
                                  self.hparams.loss_weight_segmentation_dice * self.hparams.loss_weight_segmentation * self.loss_seg_dice(
                                      volume_prediction[aux_annotated, ...],
                                      y_one_hot[aux_annotated, ...])
                              })
        else:
            loss_dict[f'seg_focal'] = torch.tensor(0., device=self.device, requires_grad=False)
            loss_dict[f'seg_dice'] = torch.tensor(0., device=self.device, requires_grad=False)
        # Self-supervised contrastive not available in the current setup, since patched embeddings would be required
        loss_dict['all'] = sum(loss_dict.values())
        loss_dict_all.append(loss_dict)

        with torch.no_grad():
            # metrics
            self.score_loss_val(loss_dict['all'])
            predictions_one_hot = F.one_hot(volume_prediction.argmax(dim=1).long(), num_classes=self.hparams.out_channels).permute(0, 4, 1, 2, 3)
            self.score_seg_val(y_pred=predictions_one_hot,
                               y=y_one_hot)

            # plotting
            if self.hparams.plot and (self.current_epoch + 1) % self.hparams.plot_interval_val == 0 and batch_idx % 10 == 0 and batch_idx <= 20:
                if not dist.is_initialized() or dist.get_rank() < 1:
                    png_paths = list()
                    png_paths.extend(image_grid.plot_grid_middle(x.detach().cpu(),
                                                                 y_one_hot.detach().cpu(),
                                                                 torch.softmax(volume_prediction.detach().cpu(), dim=1),
                                                                 scribbles.detach().cpu() if scribbles is not None else None,
                                                                 indices_elements=[idx_ * 2 for idx_ in range(
                                                                     max(min(x.shape[0] // 2, 2), 1))],
                                                                 prefix=f'{self.hparams.run_name}_val_b{batch_idx}_s0_c-all_ep{str(self.current_epoch).zfill(3)}',
                                                                 path_plots=self.hparams.default_root_dir))
                    if not self.hparams.online_off:
                        png_paths_grid_teacher = [png_ for png_ in png_paths if "grid" in png_ and "teacher" in png_]
                        png_paths_grid_student = [png_ for png_ in png_paths if "grid" in png_ and "student" in png_]
                        if len(png_paths_grid_teacher) > 0:
                            self.loggers[1].log_image(key="grid_teacher_val",
                                                      images=[wandb.Image(png_) for png_ in png_paths_grid_teacher],
                                                      caption=[str(plb.Path(png_).stem) for png_ in png_paths_grid_teacher])
                        if len(png_paths_grid_student) > 0:
                            self.loggers[1].log_image(key="grid_student_val",
                                                      images=[wandb.Image(png_) for png_ in png_paths_grid_student],
                                                      caption=[str(plb.Path(png_).stem) for png_ in png_paths_grid_student])

        if len(loss_dict_all) > 0:
            loss_dict_all = {k_: sum(dict_[k_] for dict_ in loss_dict_all) / len(loss_dict_all) for k_ in loss_dict_all[0].keys()}

            with torch.no_grad():
                # logging
                self.log(f'val_epoch', self.current_epoch, sync_dist=True)  # log for ckpt selection
                for key_, value_ in loss_dict_all.items():
                    self.log(f'val_loss_{key_}', value_, sync_dist=True, prog_bar='all' in key_)  # logs per epoch
        else:
            loss_dict_all = {'all': torch.tensor(0., device=self.device, requires_grad=False)}

        return {'loss': loss_dict_all['all']}

    def on_validation_epoch_end(self) -> None:
        score_loss = self.score_loss_val.aggregate()
        self.score_loss_val.reset()
        self.log(f'val_loss_mean', score_loss, sync_dist=True)
        score_seg = self.score_seg_val.aggregate()
        self.score_seg_val.reset()
        self.log_dict({f'val_dice_c{str(idx_c).zfill(2)}': score_seg[idx_c] for idx_c in range(len(score_seg))}, sync_dist=True)
        self.log(f'val_dice_mean', torch.mean(score_seg[1:]), sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        if batch_idx == 0:
            torch.cuda.empty_cache()  # helps with fragmentation

        # batch preparations
        aux_annotated = torch.stack(list(itertools.chain(*list(zip(*[batch[x_] for x_ in batch.keys() if 'annotated' in x_])))))  # custom collate on auxiliary keys, since monai can only concat arrays / tensors.
        x = batch['image']
        y = batch['label'].float().round().long()[:, 0, ...]
        y_one_hot = einops.rearrange(F.one_hot(y, num_classes=self.hparams.out_channels), 'b h w d c -> b c h w d')
        scribbles, scribbles_masked = None, None
        if 'scribbles' in batch.keys():
            scribbles = batch['scribbles'][:, 0, ...]
            scribbles_masked = scribbles.clone()
            scribbles_masked[~aux_annotated, ...] = self.hparams.out_channels  # Mask non-annotated

        # Activate respective instructions
        loss_dict_all = list()

        # Sliding Window Inference - by teacher
        volume_prediction = minferers.sliding_window_inference(inputs=x,
                                                               roi_size=self.hparams.patch_size_students[0],
                                                               sw_batch_size=self.hparams.batch_size,
                                                               predictor=partial(self.forward_prediction),
                                                               overlap=self.hparams.sliding_window_overlap,
                                                               mode='gaussian')

        # Loss calculation
        loss_dict, plots_dict = dict(), dict()
        # Segmentation
        test_loss = False
        if test_loss and any(aux_annotated):
            if self.hparams.loss_weight_segmentation > 0.:
                loss_dict.update(self.loss_seg_focal(volume_prediction[aux_annotated, ...],
                                                     y[aux_annotated, ...]))
                loss_dict.update({f'seg_dice':
                                      self.hparams.loss_weight_segmentation_dice * self.hparams.loss_weight_segmentation * self.loss_seg_dice(
                                          volume_prediction[aux_annotated, ...],
                                          y_one_hot[aux_annotated, ...])
                                  })
        else:
            loss_dict[f'seg_focal'] = torch.tensor(0., device=self.device, requires_grad=False)
            loss_dict[f'seg_dice'] = torch.tensor(0., device=self.device, requires_grad=False)
        # Self-supervised contrastive not available in the current setup, since patched embeddings would be required
        loss_dict['all'] = sum(loss_dict.values())
        loss_dict_all.append(loss_dict)

        with torch.no_grad():
            # metrics
            self.score_loss_test(loss_dict['all'])
            predictions_one_hot = F.one_hot(volume_prediction.argmax(dim=1).long(), num_classes=self.hparams.out_channels).permute(0, 4, 1, 2, 3)
            self.score_seg_test(y_pred=predictions_one_hot,
                                y=y_one_hot)

            # save predictions
            test_viz = False
            if test_viz:
                import matplotlib
                matplotlib.use('tkagg')  # Care: can lead to buffer overflow atm ...
                argmaxed = torch.argmax(volume_prediction[0, ...].detach().cpu(), dim=0)
                viewer = nib.viewers.OrthoSlicer3D(np.array((argmaxed / argmaxed.max())))
                viewer.clim = [0.0, 1.0]
                viewer.show()
            post_transform = generate_test_post_transforms(output_dir=self.hparams.export_dir,
                                                           output_postfix=f'pred_cat-all',
                                                           transform_test=self.trainer.datamodule.transform_test,
                                                           n_classes=None)
            batch['pred'] = volume_prediction
            [post_transform(x_) for x_ in decollate_batch(batch)]

            # plotting
            if self.hparams.plot and self.hparams.plot_interval_test > 0 and batch_idx % 10 == 0 and batch_idx <= 20:
                if not dist.is_initialized() or dist.get_rank() < 1:
                    png_paths = list()
                    png_paths.extend(image_grid.plot_grid_middle(x.detach().cpu(),
                                                                 y_one_hot.detach().cpu(),
                                                                 torch.softmax(volume_prediction.detach().cpu(), dim=1),
                                                                 scribbles.detach().cpu() if scribbles is not None else None,
                                                                 indices_elements=[idx_ * 2 for idx_ in range(
                                                                     max(min(x.shape[0] // 2, 2), 1))],
                                                                 prefix=f'{self.hparams.run_name}_test_b{batch_idx}_s0_c-all_ep{str(self.current_epoch).zfill(3)}', path_plots=self.hparams.default_root_dir))
                    if not self.hparams.online_off:
                        png_paths_grid_teacher = [png_ for png_ in png_paths if "grid" in png_ and "teacher" in png_]
                        png_paths_grid_student = [png_ for png_ in png_paths if "grid" in png_ and "student" in png_]
                        if len(png_paths_grid_teacher) > 0:
                            self.loggers[1].log_image(key="grid_teacher_test",
                                                      images=[wandb.Image(png_) for png_ in png_paths_grid_teacher],
                                                      caption=[str(plb.Path(png_).stem) for png_ in png_paths_grid_teacher])
                        if len(png_paths_grid_student) > 0:
                            self.loggers[1].log_image(key="grid_student_test",
                                                      images=[wandb.Image(png_) for png_ in png_paths_grid_student],
                                                      caption=[str(plb.Path(png_).stem) for png_ in png_paths_grid_student])

        if len(loss_dict_all) > 0:
            loss_dict_all = {k_: sum(dict_[k_] for dict_ in loss_dict_all) / len(loss_dict_all) for k_ in loss_dict_all[0].keys()}

            with torch.no_grad():
                # logging
                self.log(f'test_epoch', self.current_epoch, sync_dist=True)  # log for ckpt selection
                for key_, value_ in loss_dict_all.items():
                    self.log(f'test_loss_{key_}', value_, sync_dist=True)  # logs per epoch
        else:
            loss_dict_all = {'all': torch.tensor(0., device=self.device, requires_grad=False)}

        return {'loss': loss_dict_all['all']}

    def on_test_epoch_end(self) -> None:
        score_loss = self.score_loss_test.aggregate()
        self.score_loss_test.reset()
        self.log(f'test_loss_mean', score_loss, sync_dist=True)
        test_score = False
        if test_score:
            score_seg = self.score_seg_test.aggregate()
            self.score_seg_test.reset()
            self.log_dict({f'test_dice_c{str(idx_c).zfill(2)}': score_seg[idx_c] for idx_c in range(len(score_seg))}, sync_dist=True)
            self.log(f'test_dice_mean', torch.mean(score_seg[1:]), sync_dist=True)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        pass

    def configure_optimizers(self):
        if self.hparams.architecture == "wip_simple":
            optimizer = AdamW([{'params': [param_ for name_, param_ in [*self.architecture.network_student.get_named_body_parameters(), *self.architecture.network_student.get_named_adapter_parameters()]]},
                               {'params': [param_ for name_, param_ in self.architecture.network_student.get_named_instruction_parameters()],
                                'lr': self.hparams.learning_rate_instructions if not self.hparams.downstream else self.hparams.learning_rate_instructions_downstream}],
                              lr=self.hparams.learning_rate if not self.hparams.downstream else self.hparams.learning_rate_downstream,
                              weight_decay=self.hparams.weight_decay if not self.hparams.downstream else self.hparams.weight_decay_downstream)
        else:
            optimizer = AdamW(self.parameters(),
                              lr=self.hparams.learning_rate if not self.hparams.downstream else self.hparams.learning_rate_downstream,
                              weight_decay=self.hparams.weight_decay if not self.hparams.downstream else self.hparams.weight_decay_downstream)

        if self.hparams.downstream and self.hparams.with_scheduler_downstream:
            assert self.hparams.max_epochs is not None
            print(f'Using one cycle lr scheduler for {self.hparams.max_epochs} epochs and {1} steps per epoch.')
            scheduler = lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=[self.hparams.learning_rate_downstream, self.hparams.learning_rate_instructions_downstream] if self.hparams.architecture == "wip_simple" else self.hparams.learning_rate_downstream,
                                                total_steps=None,
                                                epochs=self.hparams.max_epochs,
                                                steps_per_epoch=1,  # The amount of scheduler.step() performed in an epoch. Probably defaults to 1 for lightning.
                                                pct_start=0.25,
                                                anneal_strategy='cos',
                                                cycle_momentum=True,
                                                base_momentum=0.85,
                                                max_momentum=0.95,
                                                div_factor=20,  # 1e-2 / 1e2 = 1e-4
                                                final_div_factor=0.25,  # 1e-4 / 1 = 1e-4
                                                three_phase=False,
                                                last_epoch=- 1,
                                                verbose=False)
            return [optimizer], [scheduler]
        return optimizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict = checkpoint["state_dict"].copy()
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                # Adjust parameters with size mismatch
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                          f"required shape: {model_state_dict[k].shape}, "
                          f"loaded shape: {state_dict[k].shape}")
                    checkpoint["state_dict"][k] = model_state_dict[k]
                    is_changed = True
            else:
                # Remove parameters not in the actual model
                warnings.warn(f"Dropping parameter: {k}")
                del checkpoint["state_dict"][k]
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.s3_bucket and not self.hparams.online_off:
            print(f'\rUploading checkpoint to {self.hparams.ckpt_dir} ...')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--learning_rate_downstream', default=5e-4, type=float)
        parser.add_argument('--learning_rate_instructions', default=1e-3, type=float)
        parser.add_argument('--learning_rate_instructions_downstream', default=5e-3, type=float)
        parser.add_argument('--weight_decay', default=1e-2, type=float)
        parser.add_argument('--weight_decay_downstream', default=0, type=float)
        parser.add_argument('--with_scheduler_downstream', default=True, type=bool)
        parser.add_argument('--sliding_window_overlap', default=0.7, type=float)

        # Segmentation loss
        parser.add_argument('--pseudo_segmentation', default=False, type=bool)
        parser.add_argument('--loss_segmentation_normalized', default=True, type=bool)
        parser.add_argument('--loss_weight_segmentation', default=2e-2, type=float)  # Note: atm an additional normalization factor of at [1e-3, 1e-1] is applied within the focal loss at all times
        parser.add_argument('--loss_weight_segmentation_dice', default=0.33, type=float)  # Lambda for dice loss
        parser.add_argument('--loss_weight_segmentation_focal', default=1.0, type=float)  # Lambda for focal loss
        parser.add_argument('--loss_weight_segmentation_gamma', default=2.0, type=float)
        parser.add_argument('--loss_weight_segmentation_alpha_background', default=1.0, type=float)
        parser.add_argument('--loss_weight_segmentation_alpha_background_downstream', default=1.0, type=float)  # May use higher value than during training, to avoid fast collapse of background into foreground.
        parser.add_argument('--loss_weight_segmentation_alpha_foreground', default=1.0, type=float)
        parser.add_argument('--background_dice', action='store_true')

        # Contrastive loss
        parser.add_argument('--contrastive_pairs', default=True, type=bool)
        parser.add_argument('--loss_weight_sim_paired', default=0., type=float)
        parser.add_argument('--loss_weight_sim_protos', default=1e-2, type=float)
        parser.add_argument('--loss_weight_sim_protos_downstream', default=0., type=float)
        parser.add_argument('--loss_weight_sim_closest', default=0., type=float)
        parser.add_argument('--loss_weight_dissim_closest', default=0., type=float)
        parser.add_argument('--k_means_iterations', default=3, type=int)
        parser.add_argument('--reduction_factor', default=[4., 4., 4.], type=float)
        parser.add_argument('--reduction_factor_protos', default=[12., 12., 8.], type=float)
        parser.add_argument('--fwhm_student_protos', default=96., type=float)
        parser.add_argument('--fwhm_teacher_protos', default=96., type=float)
        parser.add_argument('--temp_proto_teacher', default=0.025, type=float)
        parser.add_argument('--temp_proto_student', default=0.05, type=float)
        parser.add_argument('--use_weighting_protos', default=True, type=bool)
        parser.add_argument('--use_weighting_teacher', default=False, type=bool)
        parser.add_argument('--max_dist_target', default=7.0, type=float)  # ~ norm([4, 4, 2 * 2])
        parser.add_argument('--scale_z', default=2.0, type=float)
        parser.add_argument('--normalize_protos', default=True, type=bool)

        # Misc
        parser.add_argument('--label_indices_max_active', default=-1, type=int)  # Should stay negative for simple case.
        parser.add_argument('--label_indices_base', default=[], nargs='*', type=int)  # Does not have any effect atm.
        parser.add_argument('--label_indices_downstream_active', default=[], nargs='*', type=int)  # Does not have any effect atm.
        parser.add_argument('--separate_background', default=False, type=bool)  # Should stay False for simple case.
        parser.add_argument('--downstream', action='store_true')

        return parser
