
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.functional import F
from terminaltables import AsciiTable

def _iou(candidates, gt):
    start, end = candidates[:,0].float(), candidates[:,1].float()
    s, e = gt[0].float(), gt[1].float()
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union
    
def _pretty_print_results(recall_x_iou, recall_metrics, iou_metrics):
    num_recall_metrics = len(recall_metrics)
    num_iou_metrics = len(iou_metrics)
    
    for i, r in enumerate(recall_metrics):
        # Pretty print
        table = [[f'Recall@{r},mIoU@{j:.1f}' for j in iou_metrics]]
        table.append(['{:.02f}'.format(recall_x_iou[i][j]*100) for j in range(num_iou_metrics)])
        table = AsciiTable(table)
        for c in range(num_iou_metrics):
            table.justify_columns[c] = 'center'
        print(table.table)
    
def _compute_proposals_feats(v_feat, windows_idx):
    max_ = len(v_feat)
    proposal_features = []
    for s,e in windows_idx.tolist():
        s, e = max(s, 0), min(e, max_)
        proposal_features.append(torch.mean(v_feat[s:e], dim=0))

    proposal_features = torch.stack(proposal_features)
    return proposal_features.float()

def mask_to_moments(mask, num_clips):
    grids = torch.nonzero(mask, as_tuple=False)   
    grids[:, 1] += 1
    return grids.type(torch.int)

def _compute_VLG_proposals(num_frames, num_input_frames, stride, MASK, movies_durations, FPS):
    proposals = {}
    windows = {}
    for m, d in tqdm(movies_durations.items()):
        tot_frames = math.ceil(d * FPS)
        moments = mask_to_moments(MASK, num_frames)
        starts  = torch.arange(0, tot_frames - num_input_frames, stride, dtype=torch.int) 
        moments = torch.cat([moments * num_input_frames/num_frames  + s for s in starts])
        proposals[m] = moments / FPS
    return proposals

def _create_mask(N,POOLING_COUNTS):    
    # same anchor as in VLG-Net
    mask2d = torch.zeros(N, N, dtype=torch.bool)
    mask2d[range(N), range(N)] = 1

    stride, offset = 1, 0
    maskij = [(i,i) for i in range(N)]
    for c in POOLING_COUNTS:
        for _ in range(c):
            # fill a diagonal line
            offset += stride
            i, j = range(0, N - offset, stride), range(offset, N, stride)
            mask2d[i, j] = 1
            maskij += list(zip(i, j))
        stride *= 2
    return mask2d

def _get_movies_durations(annotations_keys, test_data):
    movies_durations = {}
    movies_list = []
    for k in annotations_keys:
        movie = test_data[k]['movie']
        if movie not in movies_list:
            movies_list.append(movie)
            movies_durations[movie] = test_data[k]['movie_duration']
            
    return movies_durations

def _nms(moments, scores, topk, thresh=0.5):
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]

    suppressed = torch.zeros_like(moments[:,0], dtype=torch.bool) 
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = _iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True
        if i % topk.item() == 0:
            if (~suppressed[:i]).sum() >= topk:
                break

    moments = moments[~suppressed]
    return moments[:topk]