import os
import json
import torch
import logging
import torch.nn.functional as F

from tqdm import tqdm
from terminaltables import AsciiTable
from lib.data import datasets
from lib.data.datasets.utils import iou

def pretty_print_results(recall_x_iou, recall_metrics, iou_metrics, logger):
    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    for i, r in enumerate(recall_metrics):
        # Pretty print
        table = [[f'Recall@{r},mIoU@{j:.1f}' for j in iou_metrics]]
        table.append(['{:.02f}'.format(recall_x_iou[i][j]*100) for j in range(num_iou_metrics)])
        table = AsciiTable(table)
        for c in range(num_iou_metrics):
            table.justify_columns[c] = 'center'
        logger.info('\n' + table.table)

def save_results_evaluation(recall_x_iou, recall_metrics, iou_metrics, cfg):
    directory = f'{cfg.OUTPUT_DIR}/evaluations/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    dump_folder = f'{cfg.OUTPUT_DIR}/evaluations/{cfg.SPLIT}'
    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder)

    dump_dict = dict(
        results=recall_x_iou.tolist(),
        recall_metrics=recall_metrics.tolist(),
        iou_metrics=iou_metrics.tolist(),
    )
    dump_filename = f'{dump_folder}/{cfg.CHECKPOINT.split("/")[-1].split(".")[0]}_nms_{cfg.TEST.NMS_THRESH}_stride_{cfg.TEST.STRIDE}.json'
    json.dump(dump_dict,open(dump_filename,'w'))

def nms(moments, scores, topk, thresh, relative_fps):
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    moments = moments / relative_fps

    suppressed = torch.zeros_like(moments[:,0], dtype=torch.bool) 
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True
        if i % topk.item() == 0:
            if (~suppressed[:i]).sum() >= topk:
                break

    moments = moments[~suppressed]
    return moments[:topk]

def evaluate(dataset, predictions, nms_thresh, cfg, moments_indexes,
            recall_metrics=(1,5,10,50,100), 
            iou_metrics=(0.1,0.3,0.5,0.7), 
            summary_writer=None):

    """evaluate dataset using different methods based on dataset type.
    Args:
    Returns:
    """
    dataset_name = dataset.__class__.__name__
    logger = logging.getLogger("vlg.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))
    
    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    recall_metrics = torch.tensor(recall_metrics)
    iou_metrics    = torch.tensor(iou_metrics)

    def _eval(dataset, idx, scores, moments_indexes):
        # Compute moment candidates and their scores
        stride = dataset.get_relative_stride()
        num_windows = dataset.get_number_of_windows(idx)
        candidates  = torch.cat([moments_indexes + i * stride for i in range(num_windows)])

        # Sort and apply nms
        relative_fps = dataset.get_relative_fps()
        moments = nms(candidates, scores, topk=recall_metrics[-1], 
                        relative_fps=relative_fps, thresh=nms_thresh)
        
        # Compute performance
        recall_x_iou_idx = torch.zeros(num_recall_metrics, num_iou_metrics)
        gt_moment = dataset.get_moment(idx)
        mious = iou(moments, gt_moment)
        if len(mious)< recall_metrics[-1]:
            mious = F.pad(mious, (0,recall_metrics[-1] - len(mious) ), "constant", 0.0)
        bools = mious[:,None].expand(recall_metrics[-1], num_iou_metrics) >= iou_metrics
        for i, r in enumerate(recall_metrics):
            recall_x_iou_idx[i] += bools[:r].any(dim=0)

        return recall_x_iou_idx

    recall_x_iou_dict = {}
    num_predictions = len(predictions)
    for idx, pred_scores in tqdm(enumerate(predictions)):      
        recall_x_iou_dict[idx] = _eval(dataset, idx, pred_scores, moments_indexes)
    recall_x_iou = torch.stack(list(recall_x_iou_dict.values()),dim=0).sum(dim=0)
    
    logger.info('{} is recall shape, should be [num_recall_metrics, num_iou_metrics]'.format(recall_x_iou.shape))
    recall_x_iou /= num_predictions

    for i in range(num_recall_metrics):
        for j in range(num_iou_metrics):
            name = f'R@{recall_metrics[i]:.0f}-IoU={iou_metrics[j]:.1f}'
            summary_writer.add_scalar(name,recall_x_iou[i,j]*100)

    pretty_print_results(recall_x_iou, recall_metrics, iou_metrics, logger)
    save_results_evaluation(recall_x_iou, recall_metrics, iou_metrics, cfg)
    return torch.tensor(recall_x_iou)

