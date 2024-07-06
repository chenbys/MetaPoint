from mmpose.models import builder
from mmpose.models.detectors.base import BasePose
from mmpose.models.builder import POSENETS
from mmpose.models.utils.ops import resize
from .meta_enhancer import MetaEnhancer
from .cmodules.mpmatcher import MPMatcher
from mmcv.cnn.bricks.transformer import build_positional_encoding
from .decoder_mp import MPEmbeddingDecoder
from .cmodules.loss_func import *
from .cmodules.out_func import *

eps = 1e-2


@POSENETS.register_module()
class MetaPointFormer(BasePose):
    def __init__(self, encoder_config, archicfg=None,
                 comdecodercfg=None, metadecodercfg=None, refdecodercfg=None,
                 enhancecfg=None, metacfg=None, hypercfg=None,
                 train_cfg=None, test_cfg=None, pretrained=None):
        super().__init__()
        self.archicfg = archicfg
        metadecodercfg.update(comdecodercfg)
        refdecodercfg.update(comdecodercfg)
        self.metadecodercfg = metadecodercfg
        self.refdecodercfg = refdecodercfg
        self.enhancecfg = enhancecfg
        self.metacfg = metacfg
        self.hypercfg = hypercfg
        self.global_dim = 256

        self.meta_point_num = metacfg['point_num']
        self.meta_point_dim = metacfg['point_dim']
        self.meta_point_embed = nn.Embedding(self.meta_point_num, self.meta_point_dim)
        self.meta_embed_proj = nn.Sequential()
        if self.meta_point_dim != self.global_dim:
            self.meta_embed_proj.add_module('l1', nn.Linear(self.meta_point_dim, self.global_dim))
            self.meta_embed_proj.add_module('a1', nn.ReLU())
        assert self.meta_point_num >= 68, 'all training images have at least 68 annotated keypoints.'

        self.backbone = builder.build_backbone(encoder_config)
        self.backbone.init_weights(pretrained)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.target_type = test_cfg.get('target_type', 'GaussianHeatMap')  # GaussianHeatMap

        self.backbone_dims = [256, 512, 1024, 2048]
        self.backbone_shapes = [[64, 64], [32, 32], [16, 16], [8, 8]]
        self.make_backbone_projs()

        self.matcher = MPMatcher(hypercfg, asm_type=metacfg.asm_type)

        if '-' in self.archicfg.enhance_sk_w_meta:
            self.enhance_sk_w_meta_type, self.enhance_sk_w_meta_tgt = self.archicfg.enhance_sk_w_meta.split('-')
            if self.enhance_sk_w_meta_type == 'cat':
                self.enhance_sk_w_meta_proj = nn.Sequential(
                    nn.Linear(self.global_dim * (len(self.enhance_sk_w_meta_tgt) + 1), self.global_dim),
                    nn.ReLU(),
                    nn.Linear(self.global_dim, self.global_dim),
                    nn.ReLU(),
                )
            self.enhance_sk_w_meta = True
        else:
            self.enhance_sk_w_meta = False

        if archicfg.enhance_sk_on_q:
            self.qfeatmap_enhancer = MetaEnhancer(**enhancecfg, positional_encoding=build_positional_encoding(
                dict(type='SinePositionalEncoding', num_feats=128, normalize=True)), dim_in=256)
        else:
            self.qfeatmap_enhancer = None

        self.meta_decoder = MPEmbeddingDecoder(metadecodercfg)
        self.ref_decoder = MPEmbeddingDecoder(refdecodercfg)
        self.major_key = f'refine_{self.refdecodercfg.out_layer_num - 1}'

        assert self.archicfg.skfeat_layer_num <= self.archicfg.res_layer_num
        assert self.metadecodercfg.in_layer_num <= self.archicfg.res_layer_num
        assert self.refdecodercfg.in_layer_num <= self.archicfg.res_layer_num

        if self.archicfg.skfeat_layer_fusion == 'cat':
            self.skfeat_layer_proj = nn.Sequential(
                nn.Linear(self.global_dim * self.archicfg.skfeat_layer_num, self.global_dim),
                nn.ReLU(),
                nn.Linear(self.global_dim, self.global_dim),
                nn.ReLU(),
            )

        self.visible_head = nn.Sequential(nn.ReLU(), nn.Linear(256, 256),
                                          nn.ReLU(), nn.Linear(256, 2))

        return

    def basic_forward(self, imgs_s_list, gtmaps_s_list, gtvisibles_s_list, imgs_q, gtmaps_q, gtvisibles_q, metas):
        num_shot = len(imgs_s_list)
        device = gtmaps_q.device
        batch_size = imgs_q.size(0)
        imsize_s = torch.tensor([imgs_q.shape[-2], imgs_q.shape[-1]]).unsqueeze(0).repeat(imgs_q.shape[0], 1, 1)
        gtpoints_s_list = [s / imsize_s.to(device) for s in self.parse_support_keypoints(metas, device)]
        gtcovisible_s = torch.stack(gtvisibles_s_list, 0).min(0)[0]

        imsize_q = torch.tensor([imgs_q.shape[-2], imgs_q.shape[-1]]).unsqueeze(0).repeat(imgs_q.shape[0], 1, 1)
        gtpoints_q = self.parse_query_keypoints(metas, device) / imsize_q.to(device)
        gtdict_q = dict(gtpoints=gtpoints_q, gtmaps=gtmaps_q, gtvisibles=gtvisibles_q)

        covisibles = gtcovisible_s.bool() & gtvisibles_q.bool() if self.training else gtcovisible_s.bool()
        # -----------------------------------------------------------------------------------------------
        meta_embed = self.meta_embed_proj(self.meta_point_embed.weight)[None].expand(batch_size, -1, -1)
        meta_embed_s_list, meta_point_s_list, meta_visible_s_list = [], [], []
        sk_feats_list, sk_masks_list = [], []
        for s, imgs_s in enumerate(imgs_s_list):
            featmaps_s = self.encode_multilayer_resfeats(imgs_s)
            meta_embed_s, meta_point_s = self.meta_decoder(meta_embed, featmaps_s, init_points=None)
            meta_visible_s = self.visible_head(meta_embed_s)
            meta_visible_s = torch.softmax(meta_visible_s, -1)[..., -1:].clamp(min=eps, max=1 - eps)
            meta_embed_s_list.append(meta_embed_s)
            meta_point_s_list.append(meta_point_s)
            meta_visible_s_list.append(meta_visible_s)
            sk_feats, sk_masks = self.get_skfeats(featmaps_s, gtmaps_s_list[s], gtvisibles_s_list[s])
            sk_feats_list.append(sk_feats)
            sk_masks_list.append(sk_masks)
        sk_feats_list = torch.stack(sk_feats_list)
        sk_valid_list = 1 - torch.stack(sk_masks_list).float().unsqueeze(-1)
        sk_feats = (sk_feats_list * sk_valid_list).sum(0) / sk_valid_list.sum(0).clamp(min=1)
        sk_masks = ~gtcovisible_s.bool().squeeze(-1)
        featmaps_q = self.encode_multilayer_resfeats(imgs_q)
        meta_embed_q, meta_point_q = self.meta_decoder(meta_embed, featmaps_q, init_points=None)
        meta_visible_q = self.visible_head(meta_embed_q)
        meta_visible_q = torch.softmax(meta_visible_q, -1)[..., -1:].clamp(min=eps, max=1 - eps)
        meta_to_kp = self.matcher(meta_point_s_list, meta_visible_s_list,
                                  covisibles, gtpoints_s_list, gtvisibles_s_list)
        sk_feats = self.qfeatmap_enhancer(featmaps_q[0], sk_feats,
                                          sk_masks) if self.qfeatmap_enhancer is not None else sk_feats
        includes = [sk_feats]
        if 'q' in self.enhance_sk_w_meta_tgt:
            enhance_q = self.metaseq2skseq(meta_to_kp, meta_embed_q, covisibles)
            includes.append(enhance_q)
        if 's' in self.enhance_sk_w_meta_tgt:
            enhance_s = self.metaseq2skseq(meta_to_kp, torch.stack(meta_embed_s_list).mean(0), covisibles)
            includes.append(enhance_s)
        if self.enhance_sk_w_meta_type == 'add':
            sk_feats = torch.stack(includes).sum(0)
        elif self.enhance_sk_w_meta_type == 'cat':
            sk_feats = self.enhance_sk_w_meta_proj(torch.cat(includes, -1))
        else:
            raise NotImplementedError
        ref_init_points = self.metaseq2skseq(meta_to_kp, meta_point_q[:, -1], covisibles)
        sk_feats = sk_feats * (~sk_masks).float().unsqueeze(-1)
        sk_feats_q, ref_point_q = self.ref_decoder(sk_feats, featmaps_q, init_points=ref_init_points)
        assert ((sk_feats.abs().max(-1, keepdims=True)[0] != 0).float() == (~sk_masks).float().unsqueeze(-1)).min()

        if self.training:
            gtdict_q['covisibles'] = covisibles
            loss = {}
            loss_meta_q = get_meta_loss_nbipart(self.hypercfg, meta_to_kp, meta_point_q, meta_visible_q, gtdict_q)
            loss_ref_q = get_ref_loss(self.hypercfg, ref_point_q, gtdict_q)
            loss.update(loss_meta_q)
            loss.update(loss_ref_q)
            return loss
        else:
            multi_pred_pose = obtain_inference_result(meta_to_kp, gtcovisible_s, ref_point_q)
            processed = {}
            for k, v in multi_pred_pose.items():
                processed[k] = decode_to_raw(metas, v, img_size=imgs_q.shape[-2:], test_cfg=self.test_cfg,
                                             support_visibles=gtcovisible_s.data.cpu().numpy())

            result = dict(major=processed[self.major_key], sample_image_file=metas[0]['sample_image_file'])
            return result

    def parse_query_keypoints(self, img_meta, device):
        return torch.stack([torch.tensor(info['query_joints_3d']).to(device) for info in img_meta], dim=0)[:, :, :2]

    def parse_support_keypoints(self, img_meta, device):
        shot_num = len(img_meta[0]['sample_joints_3d'])
        return [
            torch.stack([torch.tensor(info['sample_joints_3d'][s]).to(device) for info in img_meta], dim=0)[:, :, :2]
            for s in range(shot_num)]

    def forward_train(self, imgs_s, gtmaps_s, gtvisibles_s, imgs_q, gtmaps_q, gtvisibles_q, metas, **kwargs):
        return

    def forward_test(self, imgs_s, gtmaps_s, gtvisibles_s, imgs_q, gtmaps_q, gtvisibles_q,
                     metas=None, vis_similarity_map=False, vis_offset=False, **kwargs):
        return

    def show_result(self, **kwargs):
        return

    def forward(self, img_s, img_q, target_s=None, target_weight_s=None, target_q=None, target_weight_q=None,
                img_metas=None, return_loss=True, **kwargs):

        return self.basic_forward(img_s, target_s, target_weight_s, img_q,
                                  target_q, target_weight_q, img_metas, **kwargs)

    @property
    def with_keypoint(self):
        return True

    def make_backbone_projs(self):
        input_proj_list = []
        for in_channels in self.backbone_dims[::-1][:self.archicfg.res_layer_num]:
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, self.global_dim, kernel_size=1),
                nn.GroupNorm(32, self.global_dim),
            ))
        self.input_proj = nn.ModuleList(input_proj_list)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        return

    def encode_multilayer_resfeats(self, images):
        raw_resfeats = self.backbone(images)
        srcs = []
        for idx in range(self.archicfg.res_layer_num):
            x = raw_resfeats[-idx - 1]
            src = self.input_proj[idx](x)
            srcs.append(src)
        return srcs

    def get_skfeats(self, featmaps, gtmaps=None, gtvisibles=None):
        gtmaps = gtmaps / (gtmaps.sum(dim=-1).sum(dim=-1)[:, :, None, None] + 1e-8)
        msk_feats = []
        for l, featmap in enumerate(featmaps[:self.archicfg.skfeat_layer_num]):
            resized_feature = resize(input=featmap, size=gtmaps.shape[-2:], mode='bilinear', align_corners=False)
            skfeats = gtmaps.flatten(2) @ resized_feature.flatten(2).permute(0, 2, 1)
            msk_feats.append(skfeats)

        if self.archicfg.skfeat_layer_fusion == 'add':
            skfeats = torch.stack(msk_feats).sum(0)
        elif self.archicfg.skfeat_layer_fusion == 'cat':
            skfeats = self.skfeat_layer_proj(torch.cat(msk_feats, -1))
        else:
            raise NotImplementedError

        skfeats = skfeats * gtvisibles
        masks_skfeats = (~gtvisibles.to(torch.bool)).squeeze(-1)
        return skfeats, masks_skfeats

    def metaseq2skseq(self, meta_to_kp, metaseq, valid_factors):
        assert metaseq.ndim == 3
        mapped = []
        for b, (m2k, mseq) in enumerate(zip(meta_to_kp, metaseq)):
            if self.matcher.asm_type == 'bipart':
                valid_factor = valid_factors[b, :, 0]
                sseq = mseq.new_zeros(len(valid_factor), metaseq.size(-1))
                sseq[valid_factor.bool()] = mseq[m2k]
                mapped.append(sseq)
            elif self.matcher.asm_type == 'nbipart':
                mapped.append(mseq[m2k] * valid_factors[b])
            else:
                raise NotImplementedError

        return torch.stack(mapped, 0)
