import h5py
import json
import torch
from tqdm import tqdm
from torch.functional import F
from utils import _get_movies_durations, _create_mask, _compute_VLG_proposals, _compute_proposals_feats, _nms, _pretty_print_results, _iou

# Load annotations 
SPLIT='test'
root = './datasets/MAD' 
test_data = json.load(open(f'{root}/annotations/MAD_{SPLIT}.json','r'))
annotations_keys = list(test_data.keys())
movies_durations = _get_movies_durations(annotations_keys, test_data)

# Load features
FPS = 5
video_feats = h5py.File(f'{root}/features/CLIP_frames_features_5fps.h5', 'r')
lang_feats  = h5py.File(f'{root}/features/CLIP_language_features_MAD_{SPLIT}.h5', 'r')

# Define proposals
num_frames = 64
num_input_frames = 128
test_stride = 64
MASK = _create_mask(num_frames, [5,8,8,8])
proposals = _compute_VLG_proposals(num_frames, num_input_frames, test_stride, MASK, movies_durations, float(FPS))

# Define metric parameters
iou_metrics = torch.tensor([0.1,0.3,0.5])
num_iou_metrics = len(iou_metrics)

recall_metrics = torch.tensor([1,5,10,50,100])
max_recall = recall_metrics.max()
num_recall_metrics = len(recall_metrics)
recall_x_iou = torch.zeros((num_recall_metrics,len(iou_metrics)))

proposals_features = {}
cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

# Computer performance
for k in tqdm(annotations_keys):
    movie = test_data[k]['movie']
    prop = proposals[movie]
    windows_idx = torch.round(prop*FPS).int()
    gt_grounding = torch.tensor(test_data[k]['ext_timestamps'])
    
    # Get movie features and sentence features
    l_feat = torch.tensor(lang_feats[k], dtype=torch.float)[None, :]
    
    try:
        p_feats = proposals_features[movie]
    except:
        v_feat  = torch.tensor(video_feats[movie], dtype=torch.float)
        p_feats = _compute_proposals_feats(v_feat, windows_idx)
        proposals_features[movie] = p_feats
        
    sim = cosine_similarity(l_feat, p_feats)    
    best_moments = _nms(prop, sim, topk=recall_metrics[-1], thresh=0.3)

    mious = _iou(best_moments[:max_recall], gt_grounding)
    bools = mious[:,None].expand(max_recall, num_iou_metrics) > iou_metrics
    for i, r in enumerate(recall_metrics):
        recall_x_iou[i] += bools[:r].any(dim=0)

recall_x_iou /= len(annotations_keys)
_pretty_print_results(recall_x_iou, recall_metrics, iou_metrics) 
