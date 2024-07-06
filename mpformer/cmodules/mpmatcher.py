import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


class MPMatcher(nn.Module):
    def __init__(self, hypercfg, asm_type):
        super().__init__()
        self.hypercfg = hypercfg
        self.asm_type = asm_type

    @torch.no_grad()
    def forward(self, meta_point_s_list, meta_visible_s_list,
                covisibles, gtpoints_s_list, gtvisibles_s_list):
        bs = covisibles.size(0)
        meta_indices = []
        num_shots = len(meta_point_s_list)
        for b in range(bs):
            costs = []
            for s in range(num_shots):
                gt_visibles = gtvisibles_s_list[s][b].transpose(0, 1)
                gt_points = gtpoints_s_list[s][b]
                pd_points_ml = meta_point_s_list[s][b]
                pd_visibles = meta_visible_s_list[s][b]
                cost_p_ml = F.l1_loss(pd_points_ml.unsqueeze(2),
                                      gt_points.unsqueeze(0).unsqueeze(0),
                                      reduction='none').sum(-1)

                delta = len(cost_p_ml) - torch.linspace(1, len(cost_p_ml), len(cost_p_ml)).to(gt_points)
                cost_p_ml2 = torch.relu(cost_p_ml - (delta * self.hypercfg.aux)[:, None, None])
                cost_p_ml = cost_p_ml2
                cost_p = gt_visibles * cost_p_ml.mean(0)
                # -y*log(p)-(1-y)*log(1-p)
                cost_vp = -gt_visibles * torch.log(pd_visibles)
                cost_vn = -(1 - gt_visibles) * torch.log(1 - pd_visibles)
                cost = self.hypercfg.l1 * cost_p + self.hypercfg.vp * cost_vp + self.hypercfg.vn * cost_vn
                costs.append(cost)
            meta_idx, kp_idx = linear_sum_assignment(torch.stack(costs).mean(0).cpu())
            reorder = kp_idx.argsort()
            meta_ridx = meta_idx[reorder]
            kp_ridx = kp_idx[reorder]
            meta_indices.append(meta_ridx)

        return [torch.as_tensor(i, dtype=torch.int64) for i in meta_indices]
