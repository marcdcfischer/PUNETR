import torch
import torch.nn as nn
import einops
from typing import Optional, Sequence
import warnings
from monai.networks.layers import trunc_normal_


class LearnedNormedQuery(nn.Module):
    def __init__(self,
                 n_queries: int,
                 query_channels: int = 512,
                 requires_grad: bool = True):
        super().__init__()
        self.query = nn.Parameter(nn.init.xavier_uniform_(torch.empty((n_queries, query_channels)),
                                                          gain=nn.init.calculate_gain('linear')), requires_grad=requires_grad)  # [N, C]. Perceiver used truncated normal
        self.query_norm = nn.LayerNorm(query_channels, elementwise_affine=True)

    def forward(self, batch_size: int):
        return einops.repeat(self.query_norm(self.query), 'n c -> b n c', b=batch_size)


class LearnedNormedInstruction(nn.Module):
    """ List of learned embeddings vector """
    def __init__(self,
                 instruction_pool_size: int,
                 tokens_per_instruction: int = 10,
                 instruction_channels: int = 512,
                 requires_grad: bool = True,
                 use_norm: bool = True,
                 elementwise_affine: bool = True):
        super().__init__()
        self.instruction_pool_size = instruction_pool_size
        self.instructions = nn.ParameterList()
        for idx_i in range(instruction_pool_size):
            self.instructions.append(nn.Parameter(nn.init.xavier_uniform_(torch.empty((tokens_per_instruction, instruction_channels)),
                                                                          gain=nn.init.calculate_gain('linear')), requires_grad=requires_grad))  # [N, C]
        self.instructions_norm = nn.LayerNorm(instruction_channels, elementwise_affine=elementwise_affine) if use_norm else None  # Atm joint norm for all instruction sets and tokens.

    def forward(self):

        instructions = torch.stack(list(self.instructions), dim=0)
        instructions = self.instructions_norm(instructions) if self.instructions_norm is not None else instructions

        return instructions  # [I, N, C]


class LearnedNormedPseudoInstruction(nn.Module):
    """ List of learned embeddings vector """
    def __init__(self,
                 instruction_pool_size_subjects: int,
                 instruction_pool_size_labels: int,
                 tokens_per_instruction: int = 10,
                 instruction_channels: int = 512,
                 requires_grad: bool = True,
                 elementwise_affine: bool = True):
        super().__init__()
        self.instruction_pool_size_subjects = instruction_pool_size_subjects
        self.instruction_pool_size_labels = instruction_pool_size_labels
        self.instructions = nn.ParameterList()
        for idx_i in range(instruction_pool_size_subjects):
            self.instructions.append(nn.Parameter(nn.init.xavier_uniform_(torch.empty((instruction_pool_size_labels, tokens_per_instruction, instruction_channels)),
                                                                          gain=nn.init.calculate_gain('linear')), requires_grad=requires_grad))  # [N, C]
        self.instructions_norm = nn.LayerNorm(instruction_channels, elementwise_affine=elementwise_affine)  # Atm joint norm for all instructions.

    def forward(self, idx_subject, idx_label):

        # Gather and update only params of available subjects (to prevent potential excessive grad calc)
        instructions = self.instructions_norm(self.instructions[idx_subject][idx_label, ...])

        return instructions  # [N, C]


class DeepInstructedAttentionPositionScores(nn.Module):
    """
     Only inst -> cont and relative positions are needed for this case (others are 0 since instructions are not further used).
    """
    def __init__(self,
                 embedding_dim: int = 32,
                 heads: int = 4,
                 instruction_pool_size: int = 2,
                 tokens_per_instruction: int = 10,
                 separate_background: bool = True,
                 max_absolute_positions: Sequence[int] = (7, 7, 7),  # Max absolute positions index
                 max_capped_distances: Sequence[int] = (7, 7, 7),  # Max capped relative distances
                 unique_instruction_bias: bool = True,
                 unique_token_bias: bool = True,
                 no_bias_instructions: bool = False,  # Disables weights for instructions and cross biases.
                 no_bias_content: bool = False,  # Disables content weights
                 pre_scale: bool = True):
        super().__init__()
        self.heads = heads
        self.tokens_per_instruction = tokens_per_instruction
        self.separate_background = separate_background
        self.unique_instruction_bias = unique_instruction_bias
        self.unique_token_bias = unique_token_bias
        self.max_instructions = instruction_pool_size if unique_instruction_bias else 1
        self.max_token_positions = tokens_per_instruction if unique_token_bias else 1
        self.max_absolute_positions = max_absolute_positions
        self.max_capped_distances = max_capped_distances
        self.no_bias_instructions = no_bias_instructions
        self.no_bias_content = no_bias_content
        self.embedding_dim = embedding_dim
        self.inv_temperature = embedding_dim ** -0.5
        self.pre_scale = pre_scale

        # Learned encoding
        self.encoding_cross_inst_content = nn.ParameterList()
        for _ in range(self.max_instructions):
            self.encoding_cross_inst_content.append(nn.Parameter(nn.init.xavier_uniform_(torch.empty((self.max_token_positions, embedding_dim)),
                                                                 gain=nn.init.calculate_gain('linear')), requires_grad=True))
        self.encoding_content_h = nn.Parameter(nn.init.xavier_uniform_(torch.empty((2 * self.max_capped_distances[0] - 1, embedding_dim)),
                                                                       gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.encoding_content_w = nn.Parameter(nn.init.xavier_uniform_(torch.empty((2 * self.max_capped_distances[1] - 1, embedding_dim)),
                                                                       gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.encoding_content_d = nn.Parameter(nn.init.xavier_uniform_(torch.empty((2 * self.max_capped_distances[2] - 1, embedding_dim)),
                                                                       gain=nn.init.calculate_gain('linear')), requires_grad=True)

        # To be encountered relative distances
        relative_distances_h = torch.arange(self.max_absolute_positions[0], dtype=torch.long).reshape(1, -1) - torch.arange(self.max_absolute_positions[0], dtype=torch.long).reshape(-1, 1)
        relative_distances_h = torch.clamp(relative_distances_h + self.max_capped_distances[0] - 1, min=0, max=(self.max_capped_distances[0] - 1) * 2)
        self.register_buffer('relative_distances_h', relative_distances_h)
        relative_distances_w = torch.arange(self.max_absolute_positions[1], dtype=torch.long).reshape(1, -1) - torch.arange(self.max_absolute_positions[1], dtype=torch.long).reshape(-1, 1)
        relative_distances_w = torch.clamp(relative_distances_w + self.max_capped_distances[1] - 1, min=0, max=(self.max_capped_distances[1] - 1) * 2)
        self.register_buffer('relative_distances_w', relative_distances_w)
        relative_distances_d = torch.arange(self.max_absolute_positions[2], dtype=torch.long).reshape(1, -1) - torch.arange(self.max_absolute_positions[2], dtype=torch.long).reshape(-1, 1)
        relative_distances_d = torch.clamp(relative_distances_d + self.max_capped_distances[2] - 1, min=0, max=(self.max_capped_distances[2] - 1) * 2)
        self.register_buffer('relative_distances_d', relative_distances_d)

        # Learned weights (per head) to calculate score - similar to neural interpreter
        # Note: this variant is query independent (this replaces q in q^T * emb[diff]).
        self.weights_cross_inst_content = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                               gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.weights_content_h = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                      gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.weights_content_w = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                      gain=nn.init.calculate_gain('linear')), requires_grad=True)
        self.weights_content_d = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                      gain=nn.init.calculate_gain('linear')), requires_grad=True)

    def forward(self, dim_q, dim_k, dim_i, dim_h, dim_w, dim_d, label_indices: Optional[torch.Tensor] = None, device: Optional[torch.device] = None):
        """
        :param dim_q: queries dim
        :param dim_k: keys dim
        :param dim_i: actual instructions dim
        :param dim_h: actual height dim
        :param dim_w: actual width dim
        :param dim_d: actual depth dim
        :return: additive attention scores
        """

        # Retrieve embeddings according to relative / absolute / categorical position
        n_instruction_categories = dim_i // self.tokens_per_instruction
        if dim_i > 0:
            assert n_instruction_categories > 0

            if self.unique_instruction_bias:
                # Unique learnable positional embedding for all tokens and all instructions
                if label_indices is not None:
                    encodings_ = list()
                    for idx_batch in range(label_indices.shape[0]):
                        if self.separate_background:
                            instruction_indices_true = torch.nonzero(label_indices[idx_batch, 1:], as_tuple=False).squeeze(dim=1) + 1  # All indices with designated background instruction ignored
                        else:
                            instruction_indices_true = torch.nonzero(label_indices[idx_batch, :], as_tuple=False).squeeze(dim=1)  # All indices including shared background
                        encoding_ = torch.concat([self.encoding_cross_inst_content[idx_] for idx_ in instruction_indices_true], dim=0) if instruction_indices_true.numel() > 1 else self.encoding_cross_inst_content[instruction_indices_true]
                        encodings_.append(encoding_)  # [I_active * T,  C]
                    cross_inst_content_embeddings = torch.stack(encodings_, dim=0)  # [B, I_active * T, C]
                else:
                    cross_inst_content_embeddings = torch.concat(list(self.encoding_cross_inst_content[:n_instruction_categories]), dim=0).unsqueeze(dim=0) if n_instruction_categories > 1 else self.encoding_cross_inst_content[:n_instruction_categories] # [1, I_active * T, C]
                if not self.unique_token_bias:
                    cross_inst_content_embeddings = cross_inst_content_embeddings.repeat_interleave(1, self.tokens_per_instruction, 1)
            elif not self.unique_instruction_bias and self.unique_token_bias:
                cross_inst_content_embeddings = self.encoding_cross_inst_content[0].unsqueeze(dim=0).repeat(1, n_instruction_categories, 1)
            else:
                # Same learnable positional embedding for all tokens (across all instructions)
                cross_inst_content_embeddings = self.encoding_cross_inst_content[0].unsqueeze(dim=0).expand(-1, dim_i, -1)
        else:
            warnings.warn(f'Using empty instruction bias score.')
            cross_inst_content_embeddings = torch.zeros((1, dim_i, self.embedding_dim), dtype=torch.float, device=device)

        row_embeddings = self.encoding_content_h[self.relative_distances_h[:dim_h, :dim_h], :]  # [H, H, C]. Relative row positions
        col_embeddings = self.encoding_content_w[self.relative_distances_w[:dim_w, :dim_w], :]  # [W, W, C]. Relative column positions
        depth_embeddings = self.encoding_content_d[self.relative_distances_d[:dim_d, :dim_d], :]  # [D, D, C]. Relative depth positions

        cross_inst_content_scores = torch.einsum('h c, b n c -> b h n', self.weights_cross_inst_content, cross_inst_content_embeddings)  # [B, Heads, I]
        row_scores = torch.einsum('h c, n m c -> h n m', self.weights_content_h, row_embeddings).unsqueeze(dim=0)  # [1, Heads, H, H]
        col_scores = torch.einsum('h c, n m c -> h n m', self.weights_content_w, col_embeddings).unsqueeze(dim=0)  # [1, Heads, W, W]
        depth_scores = torch.einsum('h c, n m c -> h n m', self.weights_content_d, depth_embeddings).unsqueeze(dim=0)  # [1, Heads, D, D]
        content_scores = einops.rearrange(row_scores, 'b h n m -> b h n () () m () ()') + einops.rearrange(col_scores, 'b h n m -> b h () n () () m ()') + einops.rearrange(depth_scores, 'b h n m -> b h () () n () () m')  # [1, Heads, H, W, D, H, W, D]
        content_scores /= 3
        content_scores = einops.rearrange(content_scores, 'b h i j k l m n -> b h (i j k) (l m n)')  # [1, Heads, #Content, #Content]

        # Attention score matrix
        # A (0) | B (0)
        # - - -
        # C | D
        scores = torch.zeros((cross_inst_content_scores.shape[0], self.heads, dim_q, dim_k), dtype=torch.float, device=device)
        if not self.no_bias_instructions:
            scores[:, :, dim_i:, :dim_i] = cross_inst_content_scores.unsqueeze(dim=-2).expand(-1, -1, dim_k - dim_i, -1)  # [B, Heads, #Content, I]. Matrix B
        if not self.no_bias_content:
            scores[:, :, dim_i:, dim_i:] = content_scores  # [B, Heads, #Content, #Content]. Matrix D

        # (Pre-)scale scores
        scores = scores * self.inv_temperature if self.pre_scale else scores

        return scores

    def named_parameters_bias_content(self):

        params_ = [(name_, param_) for name_, param_ in self.named_parameters() if any([str_ in name_ for str_ in ['encoding_content', 'weights_content']])]

        return params_
        # return [self.encoding_content_h.named_parameters(), self.encoding_content_w, self.encoding_content_d,
        #         self.weights_content_h, self.weights_content_w, self.weights_content_d]

    def named_parameters_bias_instructions(self):

        params_ = [(name_, param_) for name_, param_ in self.named_parameters() if any([str_ in name_ for str_ in ['encoding_cross_inst_content', 'weights_cross_inst_content']])]

        return params_
        # return [self.encoding_cross_inst_content,
        #         self.weights_cross_inst_content]


class DeepInstructedAttentionPositionScoresLegacy(nn.Module):
    """
     Only inst -> cont and relative positions are needed for this case (others are 0 since instructions are not further used).
    """
    def __init__(self,
                 embedding_dim: int = 32,
                 heads: int = 4,
                 instruction_pool_size: int = 2,
                 tokens_per_instruction: int = 10,
                 separate_background: bool = True,
                 unique_instruction_bias: bool = True,
                 unique_token_bias: bool = True,
                 no_bias_instructions: bool = False,  # Disables weights for instructions and cross biases.
                 no_bias_content: bool = False,  # Disables content weights
                 pre_scale: bool = False):
        super().__init__()
        self.heads = heads
        self.tokens_per_instruction = tokens_per_instruction
        self.separate_background = separate_background
        self.unique_instruction_bias = unique_instruction_bias
        self.unique_token_bias = unique_token_bias
        self.max_instructions = instruction_pool_size if unique_instruction_bias else 1
        self.max_token_positions = tokens_per_instruction if unique_token_bias else 1
        self.no_bias_instructions = no_bias_instructions
        self.no_bias_content = no_bias_content
        self.embedding_dim = embedding_dim
        self.inv_temperature = embedding_dim ** -0.5
        self.pre_scale = pre_scale
        self.window_size_bias = (7, 7, 7)  # Use original window size to load biases of pre-trained model
        mesh_args = torch.meshgrid.__kwdefaults__

        # Learned encoding
        self.encoding_cross_inst_content = nn.ParameterList()
        for _ in range(self.max_instructions):
            self.encoding_cross_inst_content.append(nn.Parameter(nn.init.xavier_uniform_(torch.empty((self.max_token_positions, embedding_dim)),
                                                                 gain=nn.init.calculate_gain('linear')), requires_grad=True))

        # Learned weights (per head) to calculate score - similar to neural interpreter
        # Note: this variant is query independent (this replaces q in q^T * emb[diff]).
        self.weights_cross_inst_content = nn.Parameter(nn.init.xavier_uniform_(torch.empty((heads, embedding_dim)),
                                                                               gain=nn.init.calculate_gain('linear')), requires_grad=True)

        # MONAI relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size_bias[0] - 1) * (2 * self.window_size_bias[1] - 1) * (2 * self.window_size_bias[2] - 1),
                self.heads,
            )
        )
        coords_d = torch.arange(self.window_size_bias[0])
        coords_h = torch.arange(self.window_size_bias[1])
        coords_w = torch.arange(self.window_size_bias[2])
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size_bias[0] - 1
        relative_coords[:, :, 1] += self.window_size_bias[1] - 1
        relative_coords[:, :, 2] += self.window_size_bias[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size_bias[1] - 1) * (2 * self.window_size_bias[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size_bias[2] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, dim_q, dim_k, dim_i, dim_h, dim_w, dim_d, label_indices: Optional[torch.Tensor] = None, device: Optional[torch.device] = None):
        """
        :param dim_q: queries dim
        :param dim_k: keys dim
        :param dim_i: actual instructions dim
        :param dim_h: actual height dim
        :param dim_w: actual width dim
        :param dim_d: actual depth dim
        :return: additive attention scores
        """

        # Retrieve embeddings according to relative / absolute / categorical position
        n_instruction_categories = dim_i // self.tokens_per_instruction
        if dim_i > 0:
            assert n_instruction_categories > 0

            if self.unique_instruction_bias:
                # Unique learnable positional embedding for all tokens and all instructions
                if label_indices is not None:
                    encodings_ = list()
                    for idx_batch in range(label_indices.shape[0]):
                        if self.separate_background:
                            instruction_indices_true = torch.nonzero(label_indices[idx_batch, 1:], as_tuple=False).squeeze(dim=1) + 1  # All indices with designated background instruction ignored
                        else:
                            instruction_indices_true = torch.nonzero(label_indices[idx_batch, :], as_tuple=False).squeeze(dim=1)  # All indices including shared background
                        encoding_ = torch.concat([self.encoding_cross_inst_content[idx_] for idx_ in instruction_indices_true], dim=0) if instruction_indices_true.numel() > 1 else self.encoding_cross_inst_content[instruction_indices_true]
                        encodings_.append(encoding_)  # [I_active * T,  C]
                    cross_inst_content_embeddings = torch.stack(encodings_, dim=0)  # [B, I_active * T, C]
                else:
                    cross_inst_content_embeddings = torch.concat(list(self.encoding_cross_inst_content[:n_instruction_categories]), dim=0).unsqueeze(dim=0) if n_instruction_categories > 1 else self.encoding_cross_inst_content[:n_instruction_categories] # [1, I_active * T, C]
                if not self.unique_token_bias:
                    cross_inst_content_embeddings = cross_inst_content_embeddings.repeat_interleave(1, self.tokens_per_instruction, 1)
            elif not self.unique_instruction_bias and self.unique_token_bias:
                cross_inst_content_embeddings = self.encoding_cross_inst_content[0].unsqueeze(dim=0).repeat(1, n_instruction_categories, 1)
            else:
                # Same learnable positional embedding for all tokens (across all instructions)
                cross_inst_content_embeddings = self.encoding_cross_inst_content[0].unsqueeze(dim=0).expand(-1, dim_i, -1)
        else:
            warnings.warn(f'Using empty instruction bias score.')
            cross_inst_content_embeddings = torch.zeros((1, dim_i, self.embedding_dim), dtype=torch.float, device=device)
        cross_inst_content_scores = torch.einsum('h c, b n c -> b h n', self.weights_cross_inst_content, cross_inst_content_embeddings)  # [B, Heads, I]
        n_ = dim_h * dim_w * dim_d
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n_, :n_].reshape(-1)  # type: ignore
        ].reshape(n_, n_, -1)  # [Q, K, Heads]. Direct relative position bias (overall)
        content_scores = einops.rearrange(relative_position_bias, 'n m h -> () h n m')  # [1, Heads, #Content, #Content]

        # Attention score matrix
        # A (0) | B (0)
        # - - -
        # C | D
        scores = torch.zeros((cross_inst_content_scores.shape[0], self.heads, dim_q, dim_k), dtype=torch.float, device=device)
        if not self.no_bias_instructions:
            scores[:, :, dim_i:, :dim_i] = cross_inst_content_scores.unsqueeze(dim=-2).expand(-1, -1, dim_k - dim_i, -1)  # [B, Heads, #Content, I]. Matrix B
        if not self.no_bias_content:
            scores[:, :, dim_i:, dim_i:] = content_scores  # [B, Heads, #Content, #Content]. Matrix D

        # (Pre-)scale scores - Not pre-scaled in (legacy) SwinUNETR
        scores = scores * self.inv_temperature if self.pre_scale else scores

        return scores

    def named_parameters_bias_content(self):

        params_ = [(name_, param_) for name_, param_ in self.named_parameters() if any([str_ in name_ for str_ in ['relative_position_bias_table']])]
        assert len(params_) > 0

        return params_
        # return [self.encoding_content_h.named_parameters(), self.encoding_content_w, self.encoding_content_d,
        #         self.weights_content_h, self.weights_content_w, self.weights_content_d]

    def named_parameters_bias_instructions(self):

        params_ = [(name_, param_) for name_, param_ in self.named_parameters() if any([str_ in name_ for str_ in ['encoding_cross_inst_content', 'weights_cross_inst_content']])]
        assert len(params_) > 0

        return params_
        # return [self.encoding_cross_inst_content,
        #         self.weights_cross_inst_content]
