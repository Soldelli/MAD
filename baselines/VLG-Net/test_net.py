import argparse
import os

import torch

from lib.config import get_cfg_defaults
from lib.data import make_data_loader
from lib.engine.inference import inference
from lib.modeling import build_model
from lib.utils.comm import synchronize, get_rank
from lib.utils.logger import setup_logger_wandb as setup_logger
from collections import OrderedDict

import os, sys, time, logging, os.path as osp
sys.path.insert(1, f'{os.getcwd()}/utils')
from wandb_utils import Wandb
from config import config, Config
from torch.utils.tensorboard import SummaryWriter

def count_parameters(model):
    train_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        train_params += parameter.numel()
    print(f"Total Trainable Params: {train_params}")

def main():
    parser = argparse.ArgumentParser(description="VLG")
    parser.add_argument(
        "--config-file",
        default="configs/activitynet.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default='val', 
        choices=['val','test']
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.SPLIT = args.split
    cfg.CHECKPOINT = args.ckpt

    dataset_name = cfg.DATASETS.TRAIN[0].split('_')[0]
    model_name = osp.splitext(osp.basename(args.ckpt))[0]
    tags = [dataset_name,
            f'{args.split}',
            f'ModelName:{model_name}',
            f'NegProb_{cfg.MODEL.VLG.NEG_PROB}',
            f'NumAggLayers_{cfg.MODEL.VLG.FEATPOOL.NUM_AGGREGATOR_LAYERS}',
            f'NumNeighborsPool_{cfg.MODEL.VLG.FEATPOOL.NUM_NEIGHBOURS}',
            f'NumNeighborsMatch_{cfg.MODEL.VLG.MATCH.NUM_NEIGHBOURS}',
            f'NumIntLayers_{cfg.MODEL.VLG.INTEGRATOR.NUM_AGGREGATOR_LAYERS}',
            f'NumLSTMLayers_{cfg.MODEL.VLG.INTEGRATOR.LSTM.NUM_LAYERS}',
            f'PredictNumLayers_{cfg.MODEL.VLG.PREDICTOR.NUM_STACK_LAYERS}',
            f'LR_{cfg.SOLVER.LR}',
            f'InputStride_{cfg.INPUT.STRIDE}',
            f'EvalStride_{cfg.TEST.STRIDE}',
            f'NMS-th_{cfg.TEST.NMS_THRESH}'
            ]
    cfg.wandb.tags = tags
    cfg.wandb.name = osp.join(osp.basename(cfg.OUTPUT_DIR), model_name)

    # set up logging
    logger = setup_logger(config, cfg)
    logger.info(cfg)

    # init wandb *FIRST*
    if cfg.wandb.use_wandb:
        assert cfg.wandb.entity is not None
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        logging.info(f"Launch wandb, entity: {cfg.wandb.entity}")
    # then init tensorboard
    summary_writer = SummaryWriter(log_dir=f'{cfg.OUTPUT_DIR}/tensorboard')

    if not torch.cuda.is_available() and cfg.MODEL.DEVICE=='cuda':
        cfg.MODEL.DEVICE = 'cpu'    
    
    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    count_parameters(model)

    output_dir = cfg.OUTPUT_DIR

    #Load checkpoint
    checkpoint = args.ckpt 
    if os.path.isfile(checkpoint):
        state_dict = torch.load(checkpoint, map_location=torch.device(cfg.MODEL.DEVICE))['model']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    data_loader, dataset_split = None, None
    if args.split == 'val':
        dataset_split = cfg.DATASETS.VAL
        data_loader = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)

    elif args.split == 'test':
        dataset_split = cfg.DATASETS.TEST
        data_loader = make_data_loader(cfg, is_train=False, is_distributed=distributed)[0]
        
    else:
        raise ValueError('wrong split')

    with torch.cuda.amp.autocast(enabled=True):
        inference(
            model,
            data_loader,
            dataset_name=dataset_split,
            nms_thresh=cfg.TEST.NMS_THRESH,
            device=cfg.MODEL.DEVICE,
            name=dataset_name,
            logger=logger,
            summary_writer=summary_writer,
            cfg=cfg
        )
        synchronize()
        

if __name__ == "__main__":
    main()
