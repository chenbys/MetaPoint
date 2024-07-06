import numpy as np
from mmpose.core.post_processing import transform_preds
import torch


def obtain_inference_result(meta_to_kp, gt_covisibles_s, ref_point_q, invisible_score=0.5):
    all_res = {}
    for b, m2k in enumerate(meta_to_kp):
        visible_s = gt_covisibles_s[b, :, 0].bool()
        for l, pred_point in enumerate(ref_point_q[b]):
            full_res = torch.ones_like(pred_point) * invisible_score
            full_res[visible_s] = pred_point[visible_s]
            key = f'refine_{l}'
            if key not in all_res:
                all_res[key] = []
            all_res[key].append(full_res)

    return {k: torch.stack(v).cpu().numpy() for k, v in all_res.items() if len(v) > 0}


def decode_to_raw(img_metas, output, img_size, test_cfg={}, **kwargs):
    assert output.ndim == 3
    assert output.shape[-1] == 2

    batch_size = len(img_metas)
    W, H = img_size
    output = output * np.array([W, H])[None, None, :]  # [bs, query, 2], coordinates with recovered shapes.

    if 'bbox_id' or 'query_bbox_id' in img_metas[0]:
        bbox_ids = []
    else:
        bbox_ids = None

    c = np.zeros((batch_size, 2), dtype=np.float32)
    s = np.zeros((batch_size, 2), dtype=np.float32)
    image_paths = []
    score = np.ones(batch_size)
    for i in range(batch_size):
        c[i, :] = img_metas[i]['query_center']
        s[i, :] = img_metas[i]['query_scale']
        image_paths.append(img_metas[i]['query_image_file'])

        if 'query_bbox_score' in img_metas[i]:
            score[i] = np.array(
                img_metas[i]['query_bbox_score']).reshape(-1)
        if 'bbox_id' in img_metas[i]:
            bbox_ids.append(img_metas[i]['bbox_id'])
        elif 'query_bbox_id' in img_metas[i]:
            bbox_ids.append(img_metas[i]['query_bbox_id'])

    preds = np.zeros(output.shape)
    for i in range(output.shape[0]):
        preds[i] = transform_preds(output[i], c[i], s[i], [W, H], use_udp=test_cfg.get('use_udp', False))

    all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
    all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
    all_preds[:, :, 0:2] = preds[:, :, 0:2]

    if 'support_visibles' in kwargs:
        all_preds[:, :, 2:3] = kwargs['support_visibles']
    else:
        all_preds[:, :, 2:3] = 1.0

    all_boxes[:, 0:2] = c[:, 0:2]
    all_boxes[:, 2:4] = s[:, 0:2]
    all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
    all_boxes[:, 5] = score
    result = {}
    result['preds'] = all_preds
    result['boxes'] = all_boxes
    result['image_paths'] = image_paths
    result['bbox_ids'] = bbox_ids
    return result
