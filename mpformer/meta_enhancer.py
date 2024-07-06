import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy
from typing import Optional, List
from capeformer.models.utils.builder import TRANSFORMER
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from capeformer.models.utils.two_stage_support_refine_transformer import (inverse_sigmoid, MLP,
                                                                          TransformerDecoderLayer,
                                                                          TransformerEncoderLayer,
                                                                          TransformerEncoder, TransformerDecoder)


class MetaEnhancer(nn.Module):
    def __init__(self,
                 dim_in=256,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 positional_encoding=None):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.positional_encoding = positional_encoding

        self.input_proj = nn.Conv2d(256, d_model, kernel_size=1)
        if dim_in != d_model:
            self.query_proj = nn.Linear(dim_in, d_model)
        else:
            self.query_proj = nn.Identity()
        return

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')

        nn.init.xavier_uniform_(self.input_proj.weight, gain=1)
        nn.init.constant_(self.input_proj.bias, 0)

        if hasattr(self.query_proj, 'weight') and self.query_proj.weight.dim() > 1:
            nn.init.xavier_uniform_(self.query_proj.weight, gain=1)
            nn.init.constant_(self.query_proj.bias, 0)

    def forward(self, features_q, query_embed, masks_query):
        features_q = self.input_proj(features_q)
        bs, dim, h, w = features_q.shape
        masks = features_q.new_zeros((bs, 1, query_embed.size(1))).to(torch.bool)
        support_order_embedding = self.positional_encoding(masks)
        masks = features_q.new_zeros((features_q.shape[0], features_q.shape[2], features_q.shape[3])).to(torch.bool)
        pos_embed = self.positional_encoding(masks)
        query_embed = self.query_proj(query_embed)

        src = features_q.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        support_order_embed = support_order_embedding.flatten(2).permute(2, 0, 1)
        pos_embed = torch.cat((pos_embed, support_order_embed))
        query_embed = query_embed.transpose(0, 1)
        mask = masks.flatten(1)
        memory, refined_query_embed = self.encoder(src, query_embed,
                                                   src_key_padding_mask=mask,
                                                   query_key_padding_mask=masks_query,
                                                   pos=pos_embed)

        return refined_query_embed.transpose(0, 1)
