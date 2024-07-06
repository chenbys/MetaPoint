import torch
import torch.nn.functional as F
from torch import nn


def get_ref_loss(hypercfg, ref_point, gtdict):
    l1_loss_lists = [[] for _ in range(30)]

    for b in range(len(ref_point)):
        covisible = gtdict['covisibles'][b, :, 0]
        gt_points = gtdict['gtpoints'][b][covisible]

        l1_loss_ml = F.l1_loss(ref_point[b, :, covisible], gt_points, reduction='none').sum(-1).mean(-1)

        delta = len(l1_loss_ml) - torch.linspace(1, len(l1_loss_ml), len(l1_loss_ml)).to(l1_loss_ml)
        l1_loss_ml2 = torch.relu(l1_loss_ml - delta * hypercfg.aux)
        l1_loss_ml = l1_loss_ml2

        for l, l1_loss in enumerate(l1_loss_ml):
            l1_loss_lists[l].append(l1_loss)

    loss_dict = {}
    for l, ref_loss_list in enumerate(l1_loss_lists):
        if ref_loss_list:
            loss_dict[f'loss_r{l}'] = torch.stack(ref_loss_list).nanmean() * hypercfg.l1 * hypercfg.ref

    return loss_dict


def get_meta_loss_nbipart(hypercfg, meta_to_kp, meta_point, meta_visible, gtdict):
    l1_loss_lists = [[] for _ in range(30)]
    vp_loss_list = []
    vn_loss_list = []

    for b, m2k in enumerate(meta_to_kp):
        gt_visible = gtdict['gtvisibles'][b, :, 0]
        gt_point = gtdict['gtpoints'][b]

        pd_point_ml = meta_point[b][:, m2k]
        assert pd_point_ml.shape[1:] == gt_point.shape
        l1_loss_mlp = F.l1_loss(pd_point_ml, gt_point, reduction='none').sum(-1)
        delta = len(l1_loss_mlp) - torch.linspace(1, len(l1_loss_mlp), len(l1_loss_mlp)).to(gt_point)
        l1_loss_mlp2 = torch.relu(l1_loss_mlp - (delta * hypercfg.aux)[:, None])
        l1_loss_mlp = l1_loss_mlp2
        l1_loss_ml = (l1_loss_mlp * gt_visible[None]).sum(-1) / gt_visible[None].sum(-1).clamp(min=1)

        for l, l1_loss in enumerate(l1_loss_ml):
            l1_loss_lists[l].append(l1_loss / l1_loss_mlp.size(0))

        visible_asm = meta_visible[b][m2k].squeeze(-1)
        loss_vp = -gt_visible * torch.log(visible_asm)
        loss_vn = -(1 - gt_visible) * torch.log(1 - visible_asm)
        unassigned_idx = torch.ones_like(meta_visible[b, :, 0]).bool()
        unassigned_idx[m2k] = False
        visible_uasm = meta_visible[b][unassigned_idx].squeeze(-1)
        loss_vu = - torch.log(1 - visible_uasm)
        vp_loss_list.append(loss_vp.sum())
        vn_loss_list.append(loss_vn.sum() + loss_vu.sum())

    loss_dict = {}
    for l, ref_loss_list in enumerate(l1_loss_lists):
        if ref_loss_list:
            loss_dict[f'loss_m{l}'] = torch.stack(ref_loss_list).nanmean() * hypercfg.l1 * hypercfg.meta

    loss_dict[f'loss_vp'] = torch.stack(vp_loss_list).nanmean() * hypercfg.vp * hypercfg.meta
    loss_dict[f'loss_vn'] = torch.stack(vn_loss_list).nanmean() * hypercfg.vn * hypercfg.meta
    return loss_dict