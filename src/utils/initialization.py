from typing import Type
import pytorch_lightning as pl
import pathlib as plb
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary
from src.utils.old_lightning_argparse import from_argparse_args

from src.utils import callbacks, logging_custom
import warnings


def setup_training(hparams, cls_dm: Type[pl.LightningDataModule], cls_model: Type[pl.LightningModule], path_root):

    # Load and adjust hparams
    if hparams.no_overwrite:
        if hparams.ckpt is None:
            warnings.warn(f'No checkpoint is available, despite flag --no_overwrite being active.')
        else:
            hparams_ckpt = TorchCheckpointIO().load_checkpoint(path=hparams.ckpt,
                                                               map_location=lambda storage, loc: storage)['hyper_parameters']

            # Overwrite some important (local) values
            hparams_ckpt['no_overwrite'] = hparams.no_overwrite
            hparams_ckpt['cold_start'] = hparams.cold_start
            hparams_ckpt['devices'] = hparams.devices
            hparams_ckpt['accelerator'] = hparams.accelerator
            hparams_ckpt['plugins'] = hparams.plugins
            hparams_ckpt['online_off'] = hparams.online_off
            hparams_ckpt['terminate_on_nan'] = False  # Lightning has become picky ...
            hparams_ckpt['ckpt'] = hparams.ckpt
            hparams_ckpt['tags'] = hparams.tags if hasattr(hparams, 'tags') else hparams_ckpt['tags']
            hparams_ckpt['mode'] = hparams.mode
            hparams_ckpt['architecture'] = hparams.architecture
            hparams_ckpt['downstream'] = hparams.downstream
            hparams_ckpt['separate_background'] = hparams.separate_background
            hparams_ckpt['num_annotated'] = hparams.num_annotated
            hparams_ckpt['loss_weight_segmentation'] = hparams.loss_weight_segmentation
            hparams_ckpt['loss_weight_segmentation_dice'] = hparams.loss_weight_segmentation_dice
            hparams_ckpt['loss_weight_segmentation_focal'] = hparams.loss_weight_segmentation_focal
            hparams_ckpt['loss_weight_segmentation_gamma'] = hparams.loss_weight_segmentation_gamma
            hparams_ckpt['loss_weight_segmentation_alpha_background'] = hparams.loss_weight_segmentation_alpha_background
            hparams_ckpt['loss_weight_segmentation_alpha_background_downstream'] = hparams.loss_weight_segmentation_alpha_background_downstream
            hparams_ckpt['loss_weight_segmentation_alpha_foreground'] = hparams.loss_weight_segmentation_alpha_foreground
            hparams_ckpt['additive_alpha'] = hparams.additive_alpha
            hparams_ckpt['loss_weight_sim_protos'] = hparams.loss_weight_sim_protos
            hparams_ckpt['loss_weight_sim_protos_downstream'] = hparams.loss_weight_sim_protos_downstream
            hparams_ckpt['learning_rate'] = hparams.learning_rate
            hparams_ckpt['learning_rate_downstream'] = hparams.learning_rate_downstream
            hparams_ckpt['learning_rate_instructions'] = hparams.learning_rate_instructions
            hparams_ckpt['learning_rate_instructions_downstream'] = hparams.learning_rate_instructions_downstream
            hparams_ckpt['weight_decay'] = hparams.weight_decay
            hparams_ckpt['weight_decay_downstream'] = hparams.weight_decay_downstream
            hparams_ckpt['with_scheduler_downstream'] = hparams.with_scheduler_downstream
            hparams_ckpt['batch_size'] = hparams.batch_size
            hparams_ckpt['accumulate_grad_batches'] = hparams.accumulate_grad_batches
            hparams_ckpt['max_steps'] = hparams.max_steps
            hparams_ckpt['max_epochs'] = hparams.max_epochs
            hparams_ckpt['num_samples_epoch'] = hparams.num_samples_epoch

            # Replace keys not available in all cases
            for key_ in ['pseudo_segmentation', 'adaptation_variant', 'prompting_variant', 'selective_freezing',
                         'freeze_body', 'freeze_encoder', 'freeze_norm', 'freeze_bias_scores', 'freeze_inactive', 'fixed_output',
                         'tokens_per_instruction', 'tokens_per_instruction_seg', 'mean_aggregation', 'top_k_selection', 'instruction_dropout',
                         'layer_decay', 'layer_decay_downstream', 'layer_decay_factor', 'augmentation_variant', 'domain', 'out_channels',
                         'soft_selection_sigma', 'noninstructed_attention', 'noninstructed_attention_downstream',
                         'no_bias_instructions', 'no_bias_content', 'instructions_use_norm', 'instructions_use_norm_final', 'instructions_elementwise_affine', 'unique_instruction_bias', 'unique_token_bias',
                         'label_indices_base', 'label_indices_downstream_active', 'label_indices_min_active', 'label_indices_max_active',
                         'monai_swin_vit_pretrained', 'monai_swin_vit_ckpt', 'monai_swin_unetr_pretrained', 'monai_swin_unetr_ckpt']:
                if hasattr(hparams, key_):
                    hparams_ckpt[key_] = getattr(hparams, key_)

            # Misc
            hparams_ckpt['tmp_dir'] = hparams.tmp_dir
            hparams_ckpt['export_dir'] = str(plb.Path(hparams.export_dir) / plb.Path(hparams.ckpt).parent.name) if hparams.export_dir else str(plb.Path(hparams.ckpt).parent / 'predictions')
            hparams_ckpt['dir_images'] = hparams.dir_images
            hparams_ckpt['dir_masks'] = hparams.dir_masks
            hparams_ckpt['default_root_dir'] = str(path_root / 'logs' / 'lightning')
            hparams_ckpt['devices'] = hparams.devices
            hparams_ckpt['accelerator'] = hparams.accelerator
            hparams_ckpt['plugins'] = hparams.plugins

            # Logging
            hparams_ckpt['online_off'] = hparams.online_off
            hparams_ckpt['plot_interval_train'] = hparams.plot_interval_train
            hparams_ckpt['plot_interval_val'] = hparams.plot_interval_val
            hparams_ckpt['plot_interval_test'] = hparams.plot_interval_test
            hparams_ckpt['plot_only'] = hparams.plot_only

            # Remove unwanted keys
            del hparams_ckpt['run_name']

            for k_, v_ in hparams_ckpt.items():
                if k_ in hparams.__dict__.keys() and v_ != hparams.__dict__[k_]:
                    print(f'Overwriting {k_} with {v_} from passed args / ckpt instead of {hparams.__dict__[k_]}')
                else:
                    print(f'Setting {k_} to {v_} from passed args / ckpt.')
            hparams.__dict__.update(hparams_ckpt)

    # Logging
    loggers = logging_custom.setup_loggers(hparams, path_root=path_root)
    ckpt_callback = callbacks.setup_checkpointing(hparams)

    # Setup
    dm = cls_dm(hparams)
    dm.prepare_data()
    dm.setup()  # Currently raises a deprecation warning.

    if hparams.ckpt is not None and hparams.cold_start:
        model = cls_model.load_from_checkpoint(hparams.ckpt,
                                               **hparams.__dict__,
                                               strict=False if hparams.downstream else True)  # overwrite params
        resume_from_checkpoint = None
    else:
        model = cls_model(hparams)
        resume_from_checkpoint = hparams.ckpt
    print(ModelSummary(model, max_depth=-1))  # -1 to see the whole model

    # Train model
    check_val_every_n_epoch_ = hparams.check_val_every_n_epoch_ if hasattr(hparams, 'check_val_every_n_epoch_') and hparams.check_val_every_n_epoch_ is not None else hparams.check_val_every_n_epoch  # set via custom argparse parameter
    check_val_every_n_epoch_ = check_val_every_n_epoch_ if not hparams.downstream else hparams.check_val_every_n_epoch_downstream
    trainer = from_argparse_args(Trainer,
                                 hparams,
                                 devices=hparams.devices,  # always set gpus via run config / cmd line arguments
                                 gradient_clip_val=0.1,
                                 gradient_clip_algorithm='value',
                                 accumulate_grad_batches=hparams.accumulate_grad_batches,
                                 check_val_every_n_epoch=check_val_every_n_epoch_,
                                 callbacks=[ckpt_callback],
                                 logger=loggers,
                                 log_every_n_steps=49,  # 50 didn't log properly for acc grad batches of 2.
                                 precision="16-mixed")  # limit_train_batches=10, num_sanity_val_steps=0

    return dm, model, trainer, resume_from_checkpoint


def setup_testing(hparams, cls_dm: Type[pl.LightningDataModule], cls_model: Type[pl.LightningModule], path_root):

    # Adjust hparams
    assert hparams.ckpt is not None
    hparams_ckpt = TorchCheckpointIO().load_checkpoint(path=hparams.ckpt,
                                                       map_location=lambda storage, loc: storage)['hyper_parameters']

    # Overwrite some important (local) values
    hparams_ckpt['no_overwrite'] = hparams.no_overwrite
    hparams_ckpt['cold_start'] = hparams.cold_start
    hparams_ckpt['devices'] = hparams.devices
    hparams_ckpt['accelerator'] = hparams.accelerator
    hparams_ckpt['plugins'] = hparams.plugins
    hparams_ckpt['online_off'] = True
    hparams_ckpt['terminate_on_nan'] = False  # Lightning has become picky ...
    hparams_ckpt['ckpt'] = hparams.ckpt
    hparams_ckpt['monai_swin_vit_pretrained'] = False
    hparams_ckpt['monai_swin_unetr_pretrained'] = False
    hparams_ckpt['mode'] = hparams.mode
    hparams_ckpt['batch_size'] = hparams.batch_size
    hparams_ckpt['separate_background'] = hparams.separate_background  # To evaluate outdated ckpts (with wrongfully selected flag)

    # Misc
    hparams_ckpt['tmp_dir'] = hparams.tmp_dir
    hparams_ckpt['export_dir'] = str(plb.Path(hparams.export_dir) / plb.Path(hparams.ckpt).parent.name) if hparams.export_dir else str(plb.Path(hparams.ckpt).parent / 'predictions')
    hparams_ckpt['dir_images'] = hparams.dir_images
    hparams_ckpt['dir_masks'] = hparams.dir_masks
    hparams_ckpt['default_root_dir'] = str(path_root / 'logs' / 'lightning')
    hparams_ckpt['devices'] = hparams.devices
    hparams_ckpt['accelerator'] = hparams.accelerator
    hparams_ckpt['plugins'] = hparams.plugins

    # Logging
    hparams_ckpt['online_off'] = hparams.online_off
    hparams_ckpt['plot_interval_train'] = hparams.plot_interval_train
    hparams_ckpt['plot_interval_val'] = hparams.plot_interval_val
    hparams_ckpt['plot_interval_test'] = hparams.plot_interval_test
    hparams_ckpt['plot_only'] = hparams.plot_only

    hparams.__dict__.update(hparams_ckpt)

    # Logging
    loggers = logging_custom.setup_loggers(hparams, path_root=path_root)

    # Setup
    dm = cls_dm(hparams)
    dm.prepare_data()
    dm.setup()  # Currently raises a deprecation warning.

    model = cls_model.load_from_checkpoint(hparams.ckpt,
                                           **hparams.__dict__)  # overwrite parameters
    print(ModelSummary(model, max_depth=-1))  # -1 to see the whole model
    trainer = from_argparse_args(Trainer,
                                 hparams,
                                 devices=hparams.devices,
                                 logger=loggers,
                                 precision="16-mixed")

    return dm, model, trainer
