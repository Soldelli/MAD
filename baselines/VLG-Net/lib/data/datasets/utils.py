import h5py
import torch
from os.path import exists
from torch.functional import F

def iou(candidates, gt):
    start, end = candidates[:,0].float(), candidates[:,1].float()
    s, e = gt[0].float(), gt[1].float()
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union

def score2d_to_moments_scores(score2d, num_clips, duration):
    grids = torch.nonzero(score2d, as_tuple=False)   
    scores = score2d[grids[:,0], grids[:,1]]
    grids[:, 1] += 1
    moments = grids * duration / num_clips
    return moments, scores

def moment_to_iou2d(moment, num_clips, duration):
    iou2d = torch.ones(num_clips, num_clips)
    candidates, _ = score2d_to_moments_scores(iou2d, num_clips, duration)
    iou2d = iou(candidates, moment).reshape(num_clips, num_clips)
    return iou2d

def movie2feats(feat_file, movies):
    assert exists(feat_file), '{} not found'.format(feat_file)
    with h5py.File(feat_file, 'r') as f:
        vid_feats = {m: torch.from_numpy(f[m][:]) for m in movies}
    return vid_feats
