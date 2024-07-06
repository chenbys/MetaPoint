import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy
from typing import Optional, List
from capeformer.models.utils.builder import TRANSFORMER
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ProposalGenerator(nn.Module):

    def __init__(self, hidden_dim, proj_dim, dynamic_proj_dim):
        super().__init__()
        self.support_proj = nn.Linear(hidden_dim, proj_dim)
        self.query_proj = nn.Linear(hidden_dim, proj_dim)
        self.dynamic_proj = nn.Sequential(
            nn.Linear(hidden_dim, dynamic_proj_dim), nn.ReLU(),
            nn.Linear(dynamic_proj_dim, hidden_dim))
        self.dynamic_act = nn.Tanh()

    def forward(self, query_feat, support_feat, spatial_shape):
        """
        Args:
            support_feat: [query, bs, c]
            query_feat: [hw, bs, c]
            spatial_shape: h, w
        """
        device = query_feat.device
        _, bs, c = query_feat.shape
        h, w = spatial_shape
        # [bs, query, 2], Normalize the coord to [0,1]
        side_normalizer = torch.tensor([w, h]).to(query_feat.device)[None, None, :]

        query_feat = query_feat.transpose(0, 1)
        support_feat = support_feat.transpose(0, 1)
        nq = support_feat.shape[1]

        fs_proj = self.support_proj(support_feat)  # [bs, query, c]
        fq_proj = self.query_proj(query_feat)  # [bs, hw, c]
        pattern_attention = self.dynamic_act(self.dynamic_proj(fs_proj))  # [bs, query, c]

        fs_feat = (pattern_attention + 1) * fs_proj  # [bs, query, c]
        similarity = torch.bmm(fq_proj, fs_feat.transpose(1, 2))  # [bs, hw, query]
        similarity = similarity.transpose(1, 2).reshape(bs, nq, h, w)

        # ------------------------------------------------------------------------------------------------
        grid_y, grid_x = torch.meshgrid(torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                                        torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device))

        # compute softmax and sum up
        # [bs, query, 2, h, w]
        coord_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).unsqueeze(0).repeat(bs, nq, 1, 1, 1)
        coord_grid = coord_grid.permute(0, 1, 3, 4, 2)  # [bs, query, h, w, 2]
        similarity_softmax = similarity.flatten(2, 3).softmax(dim=-1)  # [bs, query, hw]
        similarity_coord_grid = similarity_softmax[:, :, :, None] * coord_grid.flatten(2, 3)
        proposal_for_loss = similarity_coord_grid.sum(dim=2, keepdim=False)  # [bs, query, 2]
        proposal_for_loss = proposal_for_loss / side_normalizer

        max_pos = torch.argmax(similarity.reshape(bs, nq, -1), dim=-1, keepdim=True)  # (bs, nq, 1)
        max_mask = F.one_hot(max_pos, num_classes=w * h)  # (bs, nq, 1, w*h)
        max_mask = max_mask.reshape(bs, nq, w, h).type(torch.float)  # (bs, nq, w, h)
        local_max_mask = F.max_pool2d(input=max_mask, kernel_size=3, stride=1, padding=1).reshape(bs, nq, w * h, 1)

        # first, extract the local probability map with the mask
        local_similarity_softmax = similarity_softmax[:, :, :, None] * local_max_mask  # (bs, nq, w*h, 1)

        # then, re-normalize the local probability map
        local_similarity_softmax = local_similarity_softmax / (
                local_similarity_softmax.sum(dim=-2, keepdim=True) + 1e-10)  # [bs, nq, w*h, 1]

        # point-wise mulplication of local probability map and coord grid
        proposals = local_similarity_softmax * coord_grid.flatten(2, 3)  # [bs, nq, w*h, 2]

        # sum the mulplication to obtain the final coord proposals
        proposals = proposals.sum(dim=2) / side_normalizer  # [bs, nq, 2]

        return proposal_for_loss, similarity, proposals


@TRANSFORMER.register_module()
class TwoStageSupportRefineTransformer(nn.Module):

    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 similarity_proj_dim=256,
                 dynamic_proj_dim=128,
                 return_intermediate_dec=True):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead,
                                                dim_feedforward, dropout,
                                                activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead,
                                                dim_feedforward, dropout,
                                                activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            d_model,
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec)

        self.proposal_generator = ProposalGenerator(
            hidden_dim=d_model,
            proj_dim=similarity_proj_dim,
            dynamic_proj_dim=dynamic_proj_dim)

        # self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')

    def forward(self, src, mask, query_embed, pos_embed, support_order_embed,
                query_padding_mask, position_embedding, kpt_branch):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        support_order_embed = support_order_embed.flatten(2).permute(2, 0, 1)
        pos_embed = torch.cat((pos_embed, support_order_embed))
        query_embed = query_embed.transpose(0, 1)  # [query, bs, c ]
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        # NOTE: to refine the support feature, we will concatenate query embed into src
        memory, refined_query_embed = self.encoder(
            src,
            query_embed,
            src_key_padding_mask=mask,
            query_key_padding_mask=query_padding_mask,
            pos=pos_embed)

        # generate initial proposals and corresponding positional embedding.
        initial_proposals_for_loss, similarity_map, initial_proposals = self.proposal_generator(
            memory, refined_query_embed, spatial_shape=[h, w])  # inital_proposals has been normalized

        # ----------------------------------------------------------------------------------------------------------------
        # NOTE: to implement the positional embedding for query, we directly treat the query embed as the tgt for decoder
        # tgt = torch.zeros_like(query_embed)
        initial_position_embedding = position_embedding.forward_coordinates(initial_proposals)
        hs, out_points = self.decoder(refined_query_embed,
                                      memory,
                                      memory_key_padding_mask=mask,
                                      pos=pos_embed,
                                      query_pos=initial_position_embedding,
                                      tgt_key_padding_mask=query_padding_mask,
                                      position_embedding=position_embedding,
                                      initial_proposals=initial_proposals,
                                      kpt_branch=kpt_branch)

        return hs.transpose(1, 2), \
               memory.permute(1, 2, 0).view(bs, c, h, w), \
               initial_proposals_for_loss, out_points, similarity_map,


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src,
                query,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                query_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # src: [hw, bs, c]
        # query: [num_query, bs, c]
        # mask: None by default
        # src_key_padding_mask: [bs, hw]
        # query_key_padding_mask: [bs, nq]
        # pos: [hw, bs, c]

        # organize the input
        # implement the attention mask to mask out the useless points
        n, bs, c = src.shape
        src_cat = torch.cat((src, query), dim=0)  # [hw + nq, bs, c]
        mask_cat = torch.cat((src_key_padding_mask, query_key_padding_mask), dim=1)  # [bs, hw+nq]
        output = src_cat

        for layer in self.layers:
            output = layer(
                output,
                query_length=n,
                src_mask=mask,
                src_key_padding_mask=mask_cat,
                pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        # resplit the output into src and query
        refined_query = output[n:, :, :]  # [nq, bs, c]
        output = output[:n, :, :]  # [n, bs, c]

        return output, refined_query


class TransformerDecoder(nn.Module):

    def __init__(self, d_model, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.ref_point_head = MLP(d_model, d_model, d_model, 2)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None,
                position_embedding=None, initial_proposals=None, kpt_branch=None):
        """
        position_embedding: Class used to compute positional embedding
        inital_proposals: [bs, nq, 2], normalized coordinates of inital proposals
        kpt_branch: MLP used to predict the offsets for each query.
        """
        output = tgt
        intermediate = []
        query_coordinates = initial_proposals.detach()
        query_points = [initial_proposals.detach()]

        tgt_key_padding_mask_remove_all_true = tgt_key_padding_mask.clone().to(
            tgt_key_padding_mask.device)
        tgt_key_padding_mask_remove_all_true[tgt_key_padding_mask.logical_not().sum(dim=-1) == 0, 0] = False

        for lidx, layer in enumerate(self.layers):
            if lidx == 0:  # use positional embedding form inital proposals
                query_pos_embed = query_pos.transpose(0, 1)
            else:
                query_pos_embed = position_embedding.forward_coordinates(query_coordinates)
                query_pos_embed = query_pos_embed.transpose(0, 1)
            query_pos_embed = self.ref_point_head(query_pos_embed)

            output = layer(output, memory,
                           tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask_remove_all_true,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos_embed)

            if self.return_intermediate:
                intermediate.append(self.norm(output))

            # update the query coordinates
            query_coordinates_unsigmoid = inverse_sigmoid(query_coordinates)
            delta_unsig = kpt_branch[lidx](output.transpose(0, 1))
            new_query_coordinates = query_coordinates_unsigmoid + delta_unsig
            new_query_coordinates = new_query_coordinates.sigmoid()

            # TODO: refer to DINO "look up twice" strategy for improvements
            query_coordinates = new_query_coordinates.detach()  # CHECK shape
            query_points.append(new_query_coordinates)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), query_points

        return output.unsqueeze(0), query_points


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,
                query_length,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        src = self.with_pos_embed(src, pos)
        q = k = src
        # NOTE: compared with original implementation, we add positional embedding into the VALUE.
        src2 = self.self_attn(
            q,
            k,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
        self.choker = nn.Linear(in_features=2 * d_model, out_features=d_model)
        # Implementation of Feedforward model

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # q = k = self.with_pos_embed(
        #    tgt, query_pos)  # TODO: modifiy here for two stage.
        q = k = self.with_pos_embed(tgt, query_pos + pos[memory.shape[0]:])
        tgt2 = self.self_attn(q, k, value=tgt,
                              attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # concatenate the positional embedding with the content feature,
        # instead of direct addition
        cross_attn_q = torch.cat((tgt, query_pos + pos[memory.shape[0]:]), dim=-1)
        cross_attn_k = torch.cat((memory, pos[:memory.shape[0]]), dim=-1)

        tgt2 = self.multihead_attn(query=cross_attn_q, key=cross_attn_k, value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(self.choker(tgt2))
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
