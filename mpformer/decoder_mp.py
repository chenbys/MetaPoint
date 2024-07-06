import torch
import torch.nn as nn
from .cmodules.ops.modules import MSDeformAttn
import torch.nn.functional as F
from capeformer.models.keypoint_heads.two_stage_head import inverse_sigmoid, TokenDecodeMLP
from mmcv.cnn.bricks.transformer import build_positional_encoding
from .cmodules.basic_func import get_ref_feat


class MPEmbeddingDecoder(nn.Module):
    def __init__(self, decodercfg):
        super().__init__()
        self.decodercfg = decodercfg
        self.global_dim = 256
        self.out_layer_num = decodercfg.out_layer_num
        self.enc_layers = nn.ModuleList()
        self.point_heads = nn.ModuleList()
        for l in range(self.out_layer_num):
            self.enc_layers.append(EncoderLayer(n_levels=decodercfg.in_layer_num,
                                                n_heads=decodercfg.n_heads,
                                                d_ffn=decodercfg.d_ffn,
                                                n_points=decodercfg.n_points))
            self.point_heads.append(TokenDecodeMLP(in_channels=self.global_dim, hidden_channels=self.global_dim))

        if decodercfg.self_att:
            self.self_att_layers = nn.ModuleList()
            self.self_att_norms = nn.ModuleList()
            for _ in range(self.out_layer_num):
                self.self_att_layers.append(nn.MultiheadAttention(self.global_dim,
                                                                  num_heads=decodercfg.n_heads,
                                                                  dropout=0.1))
                self.self_att_norms.append(nn.LayerNorm(self.global_dim))

        self.pe_layer = build_positional_encoding(dict(type='SinePositionalEncoding', num_feats=128, normalize=True))
        self.ref_feat = decodercfg.ref_feat
        return

    def forward(self, query_embed, featmaps, init_points=None):
        '''
            All layers share the same ref points!
            query_embed: [B,K,C]
            featmaps: list of [B,C,H,W]
            init_points: [B,K,2]
        '''
        valid_mask = (query_embed.abs().max(-1, keepdims=True)[0] != 0).float()

        if init_points is None:
            init_points = self.get_default_init_point(query_embed)
        else:
            init_points = init_points.detach()

        value_feat_flatten = []
        spatial_shapes = []
        for l, featmap in enumerate(featmaps):
            bs, c, h, w = featmap.shape
            spatial_shapes.append((h, w))
            featmap = featmap.flatten(2).transpose(1, 2)
            value_feat_flatten.append(featmap)

        value_feat_flatten = torch.cat(value_feat_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=query_embed.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        refined_points_ml = []
        for l, enc_layer in enumerate(self.enc_layers):
            query_pos = self.get_query_pos(init_points)
            query_embed = self.forward_self_att(query_embed, query_pos, l) * valid_mask

            if self.ref_feat:
                ref_feat = get_ref_feat(value_feat_flatten, init_points, spatial_shapes)
                query_embed = query_embed + ref_feat* valid_mask

            refined_embed = enc_layer(query_embed, query_pos, value_feat_flatten, init_points,
                                      spatial_shapes, level_start_index) * valid_mask
            refined_points = (inverse_sigmoid(init_points) + self.point_heads[l](refined_embed)).sigmoid()
            refined_points_ml.append(refined_points)
            init_points = refined_points.detach()
            query_embed = refined_embed

        return refined_embed, torch.stack(refined_points_ml, 1)

    def get_default_init_point(self, query_embed):
        device = query_embed.device
        B, K = query_embed.shape[:2]
        K_sqr = int(K ** 0.5)
        ref_y, ref_x = torch.meshgrid(torch.linspace(0., 1, K_sqr, dtype=torch.float32, device=device),
                                      torch.linspace(0., 1, K_sqr, dtype=torch.float32, device=device))
        ref = torch.stack((ref_x, ref_y), -1)
        return ref.reshape(1, -1, 2).expand(B, -1, -1)

    def get_query_pos(self, init_points):
        if self.decodercfg.iden_pos:
            masks = init_points.new_zeros((init_points.size(0), 1, init_points.size(1))).to(torch.bool)
            support_order_embedding = self.pe_layer(masks)
            iden_pos = support_order_embedding.flatten(2).permute(0, 2, 1)
        else:
            iden_pos = 0

        if self.decodercfg.coor_pos:
            coor_pos = self.pe_layer.forward_coordinates(init_points)
        else:
            coor_pos = 0

        return iden_pos + coor_pos

    def forward_self_att(self, query_embed, query_pos, layer_idx):
        if not self.decodercfg.self_att:
            return query_embed

        q = k = self.with_pos_embed(query_embed, query_pos).transpose(0, 1)
        # [L,B,C]
        tgt2 = self.self_att_layers[layer_idx](q, k, value=query_embed.transpose(0, 1))[0]
        tgt = query_embed + F.dropout(tgt2.transpose(0, 1), 0.1)
        tgt = self.self_att_norms[layer_idx](tgt)
        return tgt

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos


class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, query_embed, query_pos, values_flatten, reference_points, spatial_shapes, level_start_index):
        L = spatial_shapes.size(0)
        ref_points_ex = reference_points.unsqueeze(-2).expand(-1, -1, L, -1)

        query_embed2 = self.self_attn(self.with_pos_embed(query_embed, query_pos),
                                      ref_points_ex, values_flatten,
                                      spatial_shapes, level_start_index)
        query_embed = query_embed + self.dropout1(query_embed2)
        query_embed = self.norm1(query_embed)
        query_embed = self.forward_ffn(query_embed)
        return query_embed


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
