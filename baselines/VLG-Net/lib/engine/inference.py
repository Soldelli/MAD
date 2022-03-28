import logging
import time
import os
from tqdm import tqdm

import torch
import math
import time

from lib.engine.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str

def compute_on_dataset(model, data_loader, device, cfg, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")

    batch_size = cfg.SOLVER.BATCH_SIZE

    cache_movie_features = {}

    mask = model.mask2d.float()
    grids = torch.nonzero(mask, as_tuple=False)  

    for batch in tqdm(data_loader):
        batches, targets, idxs = batch
        movie = data_loader.dataset.get_vid(idxs[0])

        with torch.inference_mode():
            if timer:
                timer.tic()

            if movie not in cache_movie_features:
                batches = batches.to(device)
                queries = batches.queries
                wordlens = batches.wordlens

                # Compute video windows embeddings
                feats = []
                num_windows = batches.feats.shape[1]
                num_batches = math.ceil(num_windows/batch_size)
                for i in range(num_batches):
                    batch_feat = batches.feats[0, i*batch_size: (i+1)*batch_size]
                    feats.append(model.video_encoder(batch_feat))
                feats = torch.cat(feats)

                # Store computed features
                cache_movie_features[movie] = feats.to(cpu_device)
            else:
                feats = cache_movie_features[movie].to(device)
                queries = batches.queries.to(device)
                wordlens = batches.wordlens.to(device)

            # Compute query embedding
            queries  = model.language_encoder([queries, wordlens]) 
            _, w, d  = queries.shape
            queries  = queries.expand(batch_size, w, d)
            wordlens = wordlens.expand(batch_size)

            # Compute alignment
            num_windows = feats.shape[0]
            num_batches = math.ceil(num_windows/batch_size)
            
            output_cpu = []
            for i in range(num_batches):
                batch_feat = feats[i*batch_size: (i+1)*batch_size]
                B = len(batch_feat)
                scores = mask * model.fuse_and_score(batch_feat, queries[:B], wordlens[:B]).sigmoid_()
                if B == 1:
                    scores = scores[grids[:,0], grids[:,1]].flatten()
                else:
                    scores = scores[:,grids[:,0], grids[:,1]].flatten()
                output_cpu.append(scores)

            output_cpu = torch.cat(output_cpu).to(cpu_device)
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()

        results_dict[idxs[0]] = output_cpu
    return results_dict

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    idxs = list(sorted(predictions.keys()))
    if len(idxs) != idxs[-1] + 1:
        logger = logging.getLogger("vlg.inference")
        logger.warning(
            "Number of samples that were gathered from multiple processes is not "
            "a contiguous set. Some samples might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in idxs]
    return predictions

def inference(
        model,
        data_loader,
        dataset_name,
        nms_thresh,
        name,
        cfg,
        device="cuda",
        logger=None,
        summary_writer=None,
):

    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset (Size: {}).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, cfg, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / inference per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / inference per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    iou_metrics = None
    if 'MAD' in name:
        iou_metrics = (0.1,0.3,0.5)
    else:
        raise ValueError (f'Unknown dataset {name}. ')

    moments_indexes = torch.nonzero(model.mask2d.float(), as_tuple=False).cpu()
    moments_indexes[:, 1] += 1

    return evaluate(
        dataset=dataset, 
        predictions=predictions, 
        nms_thresh=nms_thresh, 
        iou_metrics=iou_metrics, 
        summary_writer=summary_writer,
        cfg=cfg, 
        moments_indexes=moments_indexes,

    )
