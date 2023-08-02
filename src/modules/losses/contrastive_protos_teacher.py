import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Dict, Tuple, List, Optional, Sequence
import math
import einops
import numpy as np
from src.utils.stable_softmax import stable_softmax


class ContrastiveProtosTeacherLoss(nn.Module):
    def __init__(self,
                 reduction_factor: Sequence[float] = (8., 8., 2.),  # Grid sampling to make loss calc feasible
                 reduction_factor_protos: Sequence[float] = (16., 16., 8.),  # Undersampling factor for prototype seeds
                 loss_weight: float = 1.0,
                 k_means_iterations: int = 3,
                 use_weighting_protos: bool = True,
                 use_weighting_teacher: bool = False,
                 fwhm_student_protos: float = 128.,
                 fwhm_teacher_protos: float = 128.,
                 temp_proto_teacher: float = 0.015,
                 temp_proto_student: float = 0.030,
                 max_dist_target: float = 3.0,
                 scale_z: float = 2.0,
                 normalized: bool = True):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.reduction_factor_protos = reduction_factor_protos
        self.loss_weight = loss_weight
        self.k_means_iterations = k_means_iterations
        self.use_weighting_protos = use_weighting_protos
        self.use_weighting_teacher = use_weighting_teacher
        self.fwhm_student_protos = fwhm_student_protos
        self.fwhm_teacher_protos = fwhm_teacher_protos
        self.temp_proto_teacher = temp_proto_teacher
        self.temp_proto_student = temp_proto_student
        self.max_dist_target = max_dist_target
        self.scale_z = scale_z
        self.normalized = normalized

    def forward(self,
                embeddings_students: List[torch.Tensor],
                embeddings_teacher: torch.Tensor,
                frames: torch.Tensor,
                coord_grids_students: List[torch.Tensor],
                coord_grids_teacher: torch.Tensor,
                dropout_rate: float = 0.2,
                generate_plots: bool = True):
        """

        :param embeddings_students: [B, C, H, W, D]
        :param embeddings_teacher: [B, C, H, W, D]
        :param frames:
        :param coord_grids_students:
        :param coord_grids_teacher:
        :param dropout_rate:
        :return:
        """
        losses, plots = dict(), dict()
        device_ = embeddings_students[0].device
        n_batch, n_channels = embeddings_students[0].shape[0], embeddings_students[0].shape[1]
        n_students = len(embeddings_students)

        # Work with D, H, W format for grid sample
        embeddings_students = [einops.rearrange(x_, 'b c h w d -> b c d h w') for x_ in embeddings_students]
        with torch.no_grad():
            embeddings_teacher = einops.rearrange(embeddings_teacher, 'b c h w d -> b c d h w')
            coord_grids_students = [einops.rearrange(x_, 'b c h w d -> b c d h w') for x_ in coord_grids_students]
            coord_grids_teacher = einops.rearrange(coord_grids_teacher, 'b c h w d -> b c d h w')
            reduction_factor_ = (self.reduction_factor[2], self.reduction_factor[0], self.reduction_factor[1])  # h w d -> d h w
            reduction_factor_protos_ = (self.reduction_factor_protos[2], self.reduction_factor_protos[0], self.reduction_factor_protos[1])  # h w d -> d h w

        # Sample seeds for prototype clustering (atm on a grid)
        with torch.no_grad():
            scaled_reduction_factor_ = torch.tensor([reduction_factor_protos_[idx_] / embeddings_teacher.shape[2:][idx_] for idx_ in range(3)], dtype=torch.float, device=device_).unsqueeze(0)
            spatial_jitter_ = (torch.rand(size=(n_batch, 3), dtype=torch.float, device=device_) - 0.5) * 2  # Jitter from -1 to 1
            spatial_jitter_ = spatial_jitter_ * scaled_reduction_factor_  # Jitter adjusted so it goes from -red.factor/img.size to +red.factor/img.size
            spatial_jitter_ = torch.stack([spatial_jitter_[..., 2], spatial_jitter_[..., 1], spatial_jitter_[..., 0]], dim=-1)  # Reorder spatial jitter from d h w -> x y z
            theta = torch.concat([torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], device=device_).unsqueeze(0).expand(n_batch, -1, -1), spatial_jitter_.unsqueeze(-1)], dim=-1)  # theta incorporating jitter
            reduced_size = [max(int(embeddings_teacher.shape[2:][idx_] // reduction_factor_protos_[idx_]), 1) for idx_ in range(3)]  # [3]
            affine_grids = F.affine_grid(theta=theta, size=[n_batch, 1, *reduced_size], align_corners=False)  # [B, D', H', W', (X Y Z)]
            embeddings_teacher_sampled = F.grid_sample(embeddings_teacher, grid=affine_grids, mode='bilinear', padding_mode='reflection', align_corners=False)  # [B, C, D', H', W']
            coord_grids_teacher_sampled = F.grid_sample(coord_grids_teacher, grid=affine_grids, mode='bilinear', padding_mode='reflection', align_corners=False)  # [B, 3, D', H', W']
            n_patch_sampled = math.prod(embeddings_teacher_sampled.shape[2:])
            assert all([x_ > 1 or y_ == 1 for x_, y_ in zip(reduced_size, embeddings_teacher.shape[2:])])  # Make sure there is more than one prototype in every direction

        # (Down-)sample student and teacher embeddings
        embeddings_students_reduced, coord_grids_students_reduced = [None for _ in range(len(embeddings_students))], [None for _ in range(len(embeddings_students))]
        embeddings_teacher_reduced, coord_grids_teacher_reduced = None, None
        for idx_emb, tuple_emb_grid_ in enumerate(zip(embeddings_students + [embeddings_teacher], coord_grids_students + [coord_grids_teacher])):
            with torch.no_grad():
                scaled_reduction_factor_ = torch.tensor([reduction_factor_[idx_] / tuple_emb_grid_[0].shape[2:][idx_] for idx_ in range(3)], dtype=torch.float, device=device_).unsqueeze(0)
                spatial_jitter_ = (torch.rand(size=(n_batch, 3), dtype=torch.float, device=device_) - 0.5) * 2  # Jitter from -1 to 1
                spatial_jitter_ = spatial_jitter_ * scaled_reduction_factor_  # Jitter adjusted so it goes from -red.factor/img.size to +red.factor/img.size
                spatial_jitter_ = torch.stack([spatial_jitter_[..., 2], spatial_jitter_[..., 1], spatial_jitter_[..., 0]], dim=-1)  # Reorder spatial jitter from d h w -> x y z
                theta = torch.concat([torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], device=device_).unsqueeze(0).expand(n_batch, -1, -1), spatial_jitter_.unsqueeze(-1)], dim=-1)  # theta incorporating jitter
                reduced_size = [max(int(tuple_emb_grid_[0].shape[2:][idx_] // reduction_factor_[idx_]), 1) for idx_ in range(3)]  # [3]
                affine_grids = F.affine_grid(theta=theta, size=[n_batch, 1, *reduced_size], align_corners=False)  # [B, D', H', W', 3]
            if idx_emb < n_students:
                embeddings_students_reduced[idx_emb] = F.grid_sample(tuple_emb_grid_[0], grid=affine_grids, mode='bilinear', padding_mode='reflection', align_corners=False)  # [B, C, D', H', W']
                with torch.no_grad():
                    coord_grids_students_reduced[idx_emb] = F.grid_sample(tuple_emb_grid_[1], grid=affine_grids, mode='bilinear', padding_mode='reflection', align_corners=False)
            else:
                with torch.no_grad():
                    embeddings_teacher_reduced = F.grid_sample(tuple_emb_grid_[0], grid=affine_grids, mode='bilinear', padding_mode='reflection', align_corners=False)
                    coord_grids_teacher_reduced = F.grid_sample(tuple_emb_grid_[1], grid=affine_grids, mode='bilinear', padding_mode='reflection', align_corners=False)

        # Contrastive losses
        loss_sim_clustered = [torch.zeros((0,), device=device_) for _ in range(n_students)]
        unique_frames = np.unique(frames)
        for idx_frame, frame_ in enumerate(unique_frames):
            valid_entries = np.array(frames) == frame_
            n_valid = np.count_nonzero(valid_entries)
            # if n_valid > 1:

            # Std contrastive learning to prototype targets
            # Retrieve proxy samples (that serve as proxy targets)
            # Note: we use a soft-assignment of prototypes chosen on a grid (as seeds / surrogates)
            with torch.no_grad():
                embeddings_teacher_valid = einops.rearrange(embeddings_teacher_reduced[valid_entries, ...], 'v c d h w -> v (d h w) c')  # [B, N, C]
                embeddings_teacher_valid_normed = F.normalize(embeddings_teacher_valid, p=2, dim=-1, eps=1e-8) if self.normalized else embeddings_teacher_valid
                embeddings_teacher_sampled_valid = einops.rearrange(embeddings_teacher_sampled[valid_entries, ...], 'v c d h w -> v (d h w) c')  # [V, P, C]
                embeddings_teacher_sampled_valid_normed = F.normalize(embeddings_teacher_sampled_valid, p=2, dim=-1, eps=1e-8) if self.normalized else embeddings_teacher_sampled_valid

                # Calc protos by soft k-means
                embeddings_protos_valid_normed = embeddings_teacher_sampled_valid_normed
                coords_protos = einops.rearrange(coord_grids_teacher_sampled[valid_entries, ...], 'v c d h w -> v (d h w) c')  # [V, P, 3]
                for idx_itr in range(self.k_means_iterations):
                    # Calc alignment
                    sim_emb_emb_teacher_protos = torch.einsum('v n c, v p c -> v n p', embeddings_teacher_valid_normed, embeddings_protos_valid_normed)  # [V, N, P]. Similarities between all teacher elements and sampled ones
                    sim_emb_emb_teacher_protos_soft = stable_softmax(sim_emb_emb_teacher_protos, self.temp_proto_teacher, dim=-1)  # Cluster alignment

                    # Calc position weights
                    pos_weights_teacher_protos = generate_masks_teacher_protos(
                        coord_grids_teacher=coord_grids_teacher_reduced[valid_entries, ...],
                        coord_grids_protos=coords_protos,
                        fwhm=self.fwhm_teacher_protos,
                        scale_z=self.scale_z,
                    )
                    sim_emb_emb_teacher_teacher_sampled_soft_weighted = sim_emb_emb_teacher_protos_soft * pos_weights_teacher_protos if self.use_weighting_protos else sim_emb_emb_teacher_protos_soft

                    # Aggregate new protos and coords
                    embeddings_protos_valid = torch.einsum('v n p, v n c -> v p c', sim_emb_emb_teacher_teacher_sampled_soft_weighted, embeddings_teacher_valid)\
                                              / torch.sum(sim_emb_emb_teacher_teacher_sampled_soft_weighted, dim=1).unsqueeze(-1)  # [V, P, C] / [V, P, 1]. Denominator is not rly needed (if it is renormalized directly afterwards)
                    embeddings_protos_valid_normed = F.normalize(embeddings_protos_valid, p=2, dim=-1, eps=1e-8) if self.normalized else embeddings_protos_valid  # [V, P, C]
                    with torch.autocast(device_type="cuda", dtype=torch.float32):
                        coords_protos = torch.einsum('v n p, v n c -> v p c', sim_emb_emb_teacher_teacher_sampled_soft_weighted.to(torch.float32), einops.rearrange(coord_grids_teacher_reduced[valid_entries, ...], 'v c d h w -> v (d h w) c'))\
                                        / torch.sum(sim_emb_emb_teacher_teacher_sampled_soft_weighted, dim=1).unsqueeze(-1)  # [V, P, C] / [V, P, 1]

                # Recalc teacher proxy alignment
                sim_emb_emb_teacher_proxy = torch.einsum('v n c, v p c -> v n p', embeddings_teacher_valid_normed, embeddings_protos_valid_normed)
                sim_emb_emb_teacher_proxy_soft = stable_softmax(sim_emb_emb_teacher_proxy, self.temp_proto_teacher, dim=-1)  # [V, N, P]
                pos_weights_teacher_protos = generate_masks_teacher_protos(
                    coord_grids_teacher=coord_grids_teacher_reduced[valid_entries, ...],
                    coord_grids_protos=coords_protos,
                    fwhm=self.fwhm_teacher_protos,
                    scale_z=self.scale_z,
                )
                sim_emb_emb_teacher_proxy_soft_final = sim_emb_emb_teacher_proxy_soft * pos_weights_teacher_protos if self.use_weighting_teacher else sim_emb_emb_teacher_proxy_soft
                if generate_plots and idx_frame == 0:
                    assignments = einops.rearrange(sim_emb_emb_teacher_proxy_soft_final, 'v (d h w) p -> v p h w d', d=embeddings_teacher_reduced.shape[2], h=embeddings_teacher_reduced.shape[3], w=embeddings_teacher_reduced.shape[4])
                    assignments_nonweighted = einops.rearrange(sim_emb_emb_teacher_proxy_soft, 'v (d h w) p -> v p h w d', d=embeddings_teacher_reduced.shape[2], h=embeddings_teacher_reduced.shape[3], w=embeddings_teacher_reduced.shape[4])
                    plots['sim_teacher_proxy_weighted'] = F.interpolate(assignments, scale_factor=(self.reduction_factor[0], self.reduction_factor[1], self.reduction_factor[2]))
                    plots['sim_teacher_proxy_nonweighted'] = F.interpolate(assignments_nonweighted, scale_factor=(self.reduction_factor[0], self.reduction_factor[1], self.reduction_factor[2]))

                # Generate the closest indices
                indices_closest, mask_max_sim_dist, pos_weights_student_protos = generate_masks_student_protos(
                    coord_grids_student=[x_[valid_entries, ...] for x_ in coord_grids_students_reduced],
                    coord_grids_teacher=coord_grids_teacher_reduced[valid_entries, ...],
                    coord_grids_protos=coords_protos,
                    fwhm=self.fwhm_student_protos,
                    max_dist_target=self.max_dist_target,
                    scale_z=self.scale_z,
                    generate_masks=False,
                )
                # plots['pos_masks_student_proto'] = pos_weights_teacher_protos.reshape(n_valid, *embeddings_teacher.shape[2:], *embeddings_teacher_sampled.shape[2:])

            for idx_student in range(n_students):
                embeddings_student_valid_ = einops.rearrange(embeddings_students_reduced[idx_student][valid_entries, ...], 'v c d h w -> v (d h w) c')  # [B, N, C]
                embeddings_student_valid_normed_ = F.normalize(embeddings_student_valid_, p=2, dim=-1, eps=1e-8) if self.normalized else embeddings_student_valid_  # [B, N, C]. Normalize (for cosine sim)
                sim_emb_emb_student_proxy_ = torch.einsum('v n c, v p c -> v n p', embeddings_student_valid_normed_, embeddings_protos_valid_normed)  # [V, N, P]
                sim_emb_emb_student_proxy_soft_log = torch.log_softmax(sim_emb_emb_student_proxy_ / self.temp_proto_student, dim=-1)
                # sim_emb_emb_student_proxy_soft_weighted = sim_emb_emb_student_proxy_soft_ * pos_weights_student_protos if self.use_weighting else sim_emb_emb_student_proxy_soft_

                # Loss calc
                for idx_valid in range(n_valid):
                    if any(mask_max_sim_dist[idx_student][idx_valid, ...]):
                        sim_emb_emb_soft_selected_log = sim_emb_emb_student_proxy_soft_log[idx_valid, ...][mask_max_sim_dist[idx_student][idx_valid, :], :]  # [N_sel, P]
                        cluster_assignments_selected_ = sim_emb_emb_teacher_proxy_soft_final[idx_valid, ...][indices_closest[idx_student][idx_valid, :], :][mask_max_sim_dist[idx_student][idx_valid, :], :]  # [N_sel, P]. Take the closest teacher->proxy assignment for each student position.
                        ce_clustered = - (cluster_assignments_selected_ * torch.clamp(sim_emb_emb_soft_selected_log, min=-1e3, max=-0.)).sum(dim=1).mean(dim=0)
                        entropy_all = ce_clustered
                        loss_sim_clustered[idx_student] = torch.concat([loss_sim_clustered[idx_student], entropy_all.reshape(-1)])

        for idx_student in range(n_students):
            losses[f'contrastive_proxy_sim_clustered_s{idx_student}'] = self.loss_weight * loss_sim_clustered[idx_student].mean() if loss_sim_clustered[idx_student].shape[0] > 0 else torch.tensor(0., device=device_)

            if generate_plots:
                excessive_ = True
                excessive_high_res_ = False
                with torch.no_grad():
                    # Calculate plotted similarities - across non-pairs (for illustration)
                    # Position is atm hardcoded
                    extension_ = 1
                    shape_student_ = embeddings_students[idx_student].shape
                    embeddings_plot_candidates = embeddings_students[idx_student][...,
                                                 shape_student_[2] // 2 - min(shape_student_[2] // 2, extension_): shape_student_[2] // 2 + 1 + min(shape_student_[2] // 2, extension_),
                                                 shape_student_[3] // 2 - extension_: shape_student_[3] // 2 + 1 + extension_,
                                                 shape_student_[4] // 2 - extension_: shape_student_[4] // 2 + 1 + extension_]
                                                 # shape_student_[2] // 4 - 5: shape_student_[2] // 4 + 5,
                                                 # int(shape_student_[3] / 1.5 - 5): int(shape_student_[3] / 1.5 + 5),
                    embeddings_plot_candidates_shape = embeddings_plot_candidates.shape[2:]
                    embeddings_plot_candidates = einops.rearrange(embeddings_plot_candidates, 'b c d h w -> b (d h w) c')
                    embeddings_plot_candidates_normed = F.normalize(embeddings_plot_candidates, p=2, dim=-1, eps=1e-8) if self.normalized else embeddings_plot_candidates
                    embeddings_teacher_vec = einops.rearrange(embeddings_teacher_reduced, 'b c d h w -> b (d h w) c')  # [B, N, C]
                    embeddings_teacher_normed = F.normalize(embeddings_teacher_vec, p=2, dim=-1, eps=1e-8) if self.normalized else embeddings_teacher_vec

                    # Save exemplary similarities of student to teachers
                    sim_emb_emb_student_proxy_plot = torch.einsum('b n c, b m c -> b n m', embeddings_plot_candidates_normed, embeddings_teacher_normed)
                    sim_emb_emb_student_proxy_plot_softmaxed = stable_softmax(sim_emb_emb_student_proxy_plot, self.temp_proto_student, dim=-1)
                    plots[f'sim_student_teacher_s{idx_student}'] = einops.rearrange(sim_emb_emb_student_proxy_plot, 'b (d h w) (k i j) -> b h w d i j k',
                        d=embeddings_plot_candidates_shape[0], h=embeddings_plot_candidates_shape[1], w=embeddings_plot_candidates_shape[2],
                        k=embeddings_teacher_reduced.shape[2], i=embeddings_teacher_reduced.shape[3], j=embeddings_teacher_reduced.shape[4])  # Atm passes non-softmaxed similarities
                    plots[f'sim_student_teacher_softmaxed_s{idx_student}'] = einops.rearrange(sim_emb_emb_student_proxy_plot_softmaxed, 'b (d h w) (k i j) -> b h w d i j k',
                        d=embeddings_plot_candidates_shape[0], h=embeddings_plot_candidates_shape[1], w=embeddings_plot_candidates_shape[2],
                        k=embeddings_teacher_reduced.shape[2], i=embeddings_teacher_reduced.shape[3], j=embeddings_teacher_reduced.shape[4])

                    if excessive_:
                        # Calculate self similarities
                        embeddings_students_self_vec = einops.rearrange(embeddings_students_reduced[idx_student], 'b c d h w -> b (d h w) c')  # [B, N, C]
                        embeddings_students_self_normed = F.normalize(embeddings_students_self_vec, p=2, dim=-1, eps=1e-8) if self.normalized else embeddings_students_self_vec

                        # Save exemplary similarities of student to teachers
                        sim_emb_emb_student_proxy_plot = torch.einsum('b n c, b m c -> b n m', embeddings_plot_candidates_normed, embeddings_students_self_normed)
                        sim_emb_emb_student_proxy_plot_softmaxed = stable_softmax(sim_emb_emb_student_proxy_plot, self.temp_proto_student, dim=-1)
                        plots[f'sim_student_student_self_s{idx_student}'] = einops.rearrange(sim_emb_emb_student_proxy_plot, 'b (d h w) (k i j) -> b h w d i j k',
                            d=embeddings_plot_candidates_shape[0], h=embeddings_plot_candidates_shape[1], w=embeddings_plot_candidates_shape[2],
                            k=embeddings_students_reduced[idx_student].shape[2], i=embeddings_students_reduced[idx_student].shape[3], j=embeddings_students_reduced[idx_student].shape[4])  # Atm passes non-softmaxed similarities
                        plots[f'sim_student_student_self_softmaxed_s{idx_student}'] = einops.rearrange(sim_emb_emb_student_proxy_plot_softmaxed, 'b (d h w) (k i j) -> b h w d i j k',
                            d=embeddings_plot_candidates_shape[0], h=embeddings_plot_candidates_shape[1], w=embeddings_plot_candidates_shape[2],
                            k=embeddings_students_reduced[idx_student].shape[2], i=embeddings_students_reduced[idx_student].shape[3], j=embeddings_students_reduced[idx_student].shape[4])

                        # Calculate plotted similarities - across non-pairs (for illustration)
                        embeddings_students_other_vec = einops.rearrange(embeddings_students_reduced[(idx_student + 1) % 2], 'b c d h w -> b (d h w) c')  # [B, N, C]
                        embeddings_students_other_normed = F.normalize(embeddings_students_other_vec, p=2, dim=-1, eps=1e-8) if self.normalized else embeddings_students_other_vec

                        # Save exemplary similarities of student to teachers
                        # Note: name is kept the same (even if it is inaccurate, so that plots are correctly handled outside of this function)
                        sim_emb_emb_student_proxy_plot = torch.einsum('b n c, b m c -> b n m', embeddings_plot_candidates_normed, embeddings_students_other_normed)
                        sim_emb_emb_student_proxy_plot_softmaxed = stable_softmax(sim_emb_emb_student_proxy_plot, self.temp_proto_student, dim=-1)
                        plots[f'sim_student_student_other_s{idx_student}'] = einops.rearrange(sim_emb_emb_student_proxy_plot, 'b (d h w) (k i j) -> b h w d i j k',
                            d=embeddings_plot_candidates_shape[0], h=embeddings_plot_candidates_shape[1], w=embeddings_plot_candidates_shape[2],
                            k=embeddings_students_reduced[(idx_student + 1) % 2].shape[2], i=embeddings_students_reduced[(idx_student + 1) % 2].shape[3], j=embeddings_students_reduced[(idx_student + 1) % 2].shape[4])  # Atm passes non-softmaxed similarities
                        plots[f'sim_student_student_other_softmaxed_s{idx_student}'] = einops.rearrange(sim_emb_emb_student_proxy_plot_softmaxed, 'b (d h w) (k i j) -> b h w d i j k',
                            d=embeddings_plot_candidates_shape[0], h=embeddings_plot_candidates_shape[1], w=embeddings_plot_candidates_shape[2],
                            k=embeddings_students_reduced[(idx_student + 1) % 2].shape[2], i=embeddings_students_reduced[(idx_student + 1) % 2].shape[3], j=embeddings_students_reduced[(idx_student + 1) % 2].shape[4])

                    if excessive_high_res_:
                        embeddings_teacher_vec = einops.rearrange(embeddings_teacher, 'b c d h w -> b (d h w) c')  # [B, N, C]
                        embeddings_teacher_normed = F.normalize(embeddings_teacher_vec, p=2, dim=-1, eps=1e-8) if self.normalized else embeddings_teacher_vec

                        # Save exemplary similarities of student to teachers
                        sim_emb_emb_student_proxy_plot = torch.einsum('b n c, b m c -> b n m', embeddings_plot_candidates_normed, embeddings_teacher_normed)
                        sim_emb_emb_student_proxy_plot_softmaxed = stable_softmax(sim_emb_emb_student_proxy_plot, self.temp_proto_student, dim=-1)
                        plots[f'sim_student_teacher_high_res_s{idx_student}'] = einops.rearrange(sim_emb_emb_student_proxy_plot, 'b (d h w) (k i j) -> b h w d i j k',
                            d=embeddings_plot_candidates_shape[0], h=embeddings_plot_candidates_shape[1], w=embeddings_plot_candidates_shape[2],
                            k=embeddings_teacher.shape[2], i=embeddings_teacher.shape[3], j=embeddings_teacher.shape[4])  # Atm passes non-softmaxed similarities
                        plots[f'sim_student_teacher_softmaxed_high_res_s{idx_student}'] = einops.rearrange(sim_emb_emb_student_proxy_plot_softmaxed, 'b (d h w) (k i j) -> b h w d i j k',
                            d=embeddings_plot_candidates_shape[0], h=embeddings_plot_candidates_shape[1], w=embeddings_plot_candidates_shape[2],
                            k=embeddings_teacher.shape[2], i=embeddings_teacher.shape[3], j=embeddings_teacher.shape[4])

                        if excessive_:
                            # Calculate self similarities
                            embeddings_students_self_vec = einops.rearrange(embeddings_students[idx_student], 'b c d h w -> b (d h w) c')  # [B, N, C]
                            embeddings_students_self_normed = F.normalize(embeddings_students_self_vec, p=2, dim=-1, eps=1e-8) if self.normalized else embeddings_students_self_vec

                            # Save exemplary similarities of student to teachers
                            sim_emb_emb_student_proxy_plot = torch.einsum('b n c, b m c -> b n m', embeddings_plot_candidates_normed, embeddings_students_self_normed)
                            sim_emb_emb_student_proxy_plot_softmaxed = stable_softmax(sim_emb_emb_student_proxy_plot, self.temp_proto_student, dim=-1)
                            plots[f'sim_student_student_self_high_res_s{idx_student}'] = einops.rearrange(sim_emb_emb_student_proxy_plot, 'b (d h w) (k i j) -> b h w d i j k',
                                d=embeddings_plot_candidates_shape[0], h=embeddings_plot_candidates_shape[1], w=embeddings_plot_candidates_shape[2],
                                k=embeddings_students[idx_student].shape[2], i=embeddings_students[idx_student].shape[3], j=embeddings_students[idx_student].shape[4])  # Atm passes non-softmaxed similarities
                            plots[f'sim_student_student_self_softmaxed_high_res_s{idx_student}'] = einops.rearrange(sim_emb_emb_student_proxy_plot_softmaxed, 'b (d h w) (k i j) -> b h w d i j k',
                                d=embeddings_plot_candidates_shape[0], h=embeddings_plot_candidates_shape[1], w=embeddings_plot_candidates_shape[2],
                                k=embeddings_students[idx_student].shape[2], i=embeddings_students[idx_student].shape[3], j=embeddings_students[idx_student].shape[4])

                            # Calculate plotted similarities - across non-pairs (for illustration)
                            embeddings_students_other_vec = einops.rearrange(embeddings_students[(idx_student + 1) % 2], 'b c d h w -> b (d h w) c')  # [B, N, C]
                            embeddings_students_other_normed = F.normalize(embeddings_students_other_vec, p=2, dim=-1, eps=1e-8) if self.normalized else embeddings_students_other_vec

                            # Save exemplary similarities of student to teachers
                            # Note: name is kept the same (even if it is inaccurate, so that plots are correctly handled outside of this function)
                            sim_emb_emb_student_proxy_plot = torch.einsum('b n c, b m c -> b n m', embeddings_plot_candidates_normed, embeddings_students_other_normed)
                            sim_emb_emb_student_proxy_plot_softmaxed = stable_softmax(sim_emb_emb_student_proxy_plot, self.temp_proto_student, dim=-1)
                            plots[f'sim_student_student_other_high_res_s{idx_student}'] = einops.rearrange(sim_emb_emb_student_proxy_plot, 'b (d h w) (k i j) -> b h w d i j k',
                                d=embeddings_plot_candidates_shape[0], h=embeddings_plot_candidates_shape[1], w=embeddings_plot_candidates_shape[2],
                                k=embeddings_students[(idx_student + 1) % 2].shape[2], i=embeddings_students[(idx_student + 1) % 2].shape[3], j=embeddings_students[(idx_student + 1) % 2].shape[4])  # Atm passes non-softmaxed similarities
                            plots[f'sim_student_student_other_softmaxed_high_res_s{idx_student}'] = einops.rearrange(sim_emb_emb_student_proxy_plot_softmaxed, 'b (d h w) (k i j) -> b h w d i j k',
                                d=embeddings_plot_candidates_shape[0], h=embeddings_plot_candidates_shape[1], w=embeddings_plot_candidates_shape[2],
                                k=embeddings_students[(idx_student + 1) % 2].shape[2], i=embeddings_students[(idx_student + 1) % 2].shape[3], j=embeddings_students[(idx_student + 1) % 2].shape[4])
        return losses, plots


def generate_masks_student_protos(coord_grids_student: List[torch.Tensor],
                                  coord_grids_teacher: torch.Tensor,
                                  coord_grids_protos: torch.Tensor,
                                  fwhm: float = 256.,
                                  max_dist_target: float = 3.0,
                                  scale_z: float = 2.0,
                                  thresh: float = 0.5,
                                  generate_masks: bool = False):

    # i,j,k position differences - student to teacher
    pos_masks_student_protos = list()
    indices_closest = list()
    mask_max_sim_dist = list()
    for idx_student in range(len(coord_grids_student)):
        diff_ijk = (einops.rearrange(coord_grids_student[idx_student], 'b c d h w -> c b (d h w) ()') - einops.rearrange(coord_grids_teacher, 'b c d h w -> c b () (d h w)'))  # [3, B, N1, N2].
        diff_ijk[2, ...] *= scale_z  # penalize coord diff in z-direction more strongly.
        diff_all = torch.linalg.norm(diff_ijk, ord=2, dim=0)  # [B, N1, N2]

        diff_minimum, indices_closest_ = torch.min(diff_all, dim=-1)  # [B, N1], [B, N1].
        indices_closest.append(indices_closest_)
        mask_max_sim_dist.append(diff_minimum <= max_dist_target)  # [B, N1].

        if generate_masks:
            diff_ijk = (einops.rearrange(coord_grids_student[idx_student], 'b c d h w -> c b (d h w) ()') - einops.rearrange(coord_grids_protos, 'b n c -> c b () n'))  # [3, B, N1, N2].
            diff_ijk[2, ...] *= scale_z  # penalize coord diff in z-direction more strongly.
            diff_all = torch.linalg.norm(diff_ijk, ord=2, dim=0)  # [B, N1, N2]
            sigma_squared = (fwhm / 2.355) ** 2  # FWHM ~= 2.355*sigma
            pos_masks_student_protos.append(torch.exp(- diff_all ** 2 / (2 * sigma_squared)) >= thresh)  # [B, N1, N2]. Weights are compared to threshold to produce binary mask.

    return indices_closest, mask_max_sim_dist, pos_masks_student_protos


def generate_masks_teacher_protos(coord_grids_teacher: torch.Tensor,
                                  coord_grids_protos: torch.Tensor,
                                  fwhm: float = 256.,
                                  scale_z: float = 2.0):
    # i,j,k position differences - teacher to prototype surrogates
    diff_ijk = (einops.rearrange(coord_grids_teacher, 'b c d h w -> c b (d h w) ()') - einops.rearrange(coord_grids_protos, 'b n c -> c b () n'))  # [3, B, N1, N2]. Protos are already in node shape.
    diff_ijk[2, ...] *= scale_z  # penalize coord diff in z-direction more strongly.
    diff_all = torch.linalg.norm(diff_ijk, ord=2, dim=0)  # [B, N1, N2]

    sigma_squared = (fwhm / 2.355)**2  # FWHM ~= 2.355*sigma
    pos_weights_teacher_protos = torch.exp(- diff_all ** 2 / (2 * sigma_squared))  # [B, N1, N2]. True weights (not masks).

    return pos_weights_teacher_protos
