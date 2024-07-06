import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from mpformer.cmodules.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
from mmpose.models.utils.ops import resize


def get_ref_feat(values_flatten, ref_points, spatial_shapes):
    n_heads, n_points = 1, 1
    N, Len_q, _ = ref_points.shape
    N, Len_in, d_model = values_flatten.shape
    n_levels = spatial_shapes.size(0)

    # N, Len_q, n_heads, n_levels, n_points, 2
    # A = A.view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
    # A = F.softmax(A, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
    sampling_locations = ref_points.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2).expand(
        -1, -1, n_heads, n_levels, n_points, -1).contiguous()
    attention_weights = (torch.ones(N, Len_q, n_heads, n_levels, n_points).to(values_flatten) /
                         (n_levels * n_points)).contiguous()
    values_split = values_flatten.view(N, Len_in, n_heads, d_model // n_heads)
    # feat_from = MSDeformAttnFunction.apply(values_split, spatial_shapes, level_start_index,
    #                                        sampling_locations, attention_weights, 128)
    feat_from = ms_deform_attn_core_pytorch(values_split, spatial_shapes, sampling_locations, attention_weights)
    return feat_from
