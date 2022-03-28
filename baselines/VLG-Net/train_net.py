import argparse
import os

import torch
import random
from torch import optim
from torch import multiprocessing
from pathlib import Path
import numpy as np
multiprocessing.set_sharing_strategy('file_system')

from lib.config import get_cfg_defaults, set_hps_cfg
from lib.data import make_data_loader
from lib.engine.inference import inference
from lib.engine.trainer import do_train
from lib.modeling import build_model
from lib.utils.checkpoint import VLGCheckpointer
from lib.utils.comm import synchronize, get_rank, cleanup
from lib.utils.imports import import_file
from lib.utils.logger import setup_logger_wandb as setup_logger
from lib.utils.miscellaneous import mkdir, save_config

import logging

from prettytable import PrettyTable
import os, sys, time, logging, os.path as osp
sys.path.insert(1, f'{os.getcwd()}/utils')
from wandb_utils import Wandb
from config import config, Config
from torch.utils.tensorboard import SummaryWriter

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    train_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        train_params+=param
    print(table)
    print(f"Total Trainable Params: {train_params}")


def load_pretrained_graph_weights(model,cfg,logger):
    #check dimension to load correct model:
    if cfg.MODEL.VLG.FEATPOOL.HIDDEN_SIZE == 256:
        path = './datasets/gcnext_warmup/gtad_best_256.pth.tar'
    elif cfg.MODEL.VLG.FEATPOOL.HIDDEN_SIZE == 512:
        path = './datasets/gcnext_warmup/gtad_best_512.pth.tar'
    
    logger.info('Load pretrained model from {}'.format(path))
    pretrained_dict = torch.load(path)['state_dict']
    pretrained_keep = dict() # manually copy the weight
    if '256' in path:
        layer_name = 'module.x_1d_b'
        for i in range(cfg.MODEL.VLG.FEATPOOL.NUM_AGGREGATOR_LAYERS ):
            pretrained_keep[f'context_aggregator.{i}.tconvs.0.weight'] = pretrained_dict['module.x_1d_b.2.tconvs.0.weight']
            pretrained_keep[f'context_aggregator.{i}.tconvs.0.bias']   = pretrained_dict['module.x_1d_b.2.tconvs.0.bias']
            pretrained_keep[f'context_aggregator.{i}.tconvs.2.weight'] = pretrained_dict['module.x_1d_b.2.tconvs.2.weight']
            pretrained_keep[f'context_aggregator.{i}.tconvs.2.bias']   = pretrained_dict['module.x_1d_b.2.tconvs.2.bias']
            pretrained_keep[f'context_aggregator.{i}.tconvs.4.weight'] = pretrained_dict['module.x_1d_b.2.tconvs.4.weight']
            pretrained_keep[f'context_aggregator.{i}.tconvs.4.bias']   = pretrained_dict['module.x_1d_b.2.tconvs.4.bias']
            pretrained_keep[f'context_aggregator.{i}.sconvs.0.weight'] = pretrained_dict['module.x_1d_b.2.fconvs.0.weight']
            pretrained_keep[f'context_aggregator.{i}.sconvs.0.bias']   = pretrained_dict['module.x_1d_b.2.fconvs.0.bias']
            pretrained_keep[f'context_aggregator.{i}.sconvs.2.weight'] = pretrained_dict['module.x_1d_b.2.fconvs.2.weight']
            pretrained_keep[f'context_aggregator.{i}.sconvs.2.bias']   = pretrained_dict['module.x_1d_b.2.fconvs.2.bias']
            pretrained_keep[f'context_aggregator.{i}.sconvs.4.weight'] = pretrained_dict['module.x_1d_b.2.fconvs.4.weight']
            pretrained_keep[f'context_aggregator.{i}.sconvs.4.bias']   = pretrained_dict['module.x_1d_b.2.fconvs.4.bias']
    elif '512' in path:
        layer_name = 'module.backbone1'
        for i in range(cfg.MODEL.VLG.FEATPOOL.NUM_AGGREGATOR_LAYERS ):
            pretrained_keep[f'context_aggregator.{i}.tconvs.0.weight'] = pretrained_dict['module.backbone1.2.tconvs.0.weight']
            pretrained_keep[f'context_aggregator.{i}.tconvs.0.bias']   = pretrained_dict['module.backbone1.2.tconvs.0.bias']
            pretrained_keep[f'context_aggregator.{i}.tconvs.2.weight'] = pretrained_dict['module.backbone1.2.tconvs.2.weight']
            pretrained_keep[f'context_aggregator.{i}.tconvs.2.bias']   = pretrained_dict['module.backbone1.2.tconvs.2.bias']
            pretrained_keep[f'context_aggregator.{i}.tconvs.4.weight'] = pretrained_dict['module.backbone1.2.tconvs.4.weight']
            pretrained_keep[f'context_aggregator.{i}.tconvs.4.bias']   = pretrained_dict['module.backbone1.2.tconvs.4.bias']
            pretrained_keep[f'context_aggregator.{i}.sconvs.0.weight'] = pretrained_dict['module.backbone1.2.sconvs.0.weight']
            pretrained_keep[f'context_aggregator.{i}.sconvs.0.bias']   = pretrained_dict['module.backbone1.2.sconvs.0.bias']
            pretrained_keep[f'context_aggregator.{i}.sconvs.2.weight'] = pretrained_dict['module.backbone1.2.sconvs.2.weight']
            pretrained_keep[f'context_aggregator.{i}.sconvs.2.bias']   = pretrained_dict['module.backbone1.2.sconvs.2.bias']
            pretrained_keep[f'context_aggregator.{i}.sconvs.4.weight'] = pretrained_dict['module.backbone1.2.sconvs.4.weight']
            pretrained_keep[f'context_aggregator.{i}.sconvs.4.bias']   = pretrained_dict['module.backbone1.2.sconvs.4.bias']
    else:
        raise ValueError ('Specify hidden size in feature file name')

    model.load_state_dict(pretrained_keep,strict=False)
    return model

def train(cfg, writer, local_rank, distributed, logger):
    logger = logging.getLogger("vlg.trainer")
    model = build_model(cfg)
    
    logger = logging.getLogger("vlg.trainer")
    
    ### GTAD pretraining
    if cfg.MODEL.PRETRAINV:
        model = load_pretrained_graph_weights(model,cfg,logger)

    # Move model to GPU
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    count_parameters(model)
    
    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.SOLVER.LR_STEP_SIZE, gamma=cfg.SOLVER.LR_GAMMA) 
    scaler = torch.cuda.amp.GradScaler(enabled=True)  # NEW

    # Deprecated, to be removed.
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    save_to_disk = get_rank() == 0
    checkpointer = VLGCheckpointer(
        cfg, model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(f='', use_latest=False)
    arguments = {"epoch": 1}
    arguments.update(extra_checkpoint_data)
    
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
    )

    data_loader_val = None
    data_loader_test = None 
    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        if len(cfg.DATASETS.VAL) != 0:
            data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)
        else:
            logger.info('Please specify validation dataset in config file for performance evaluation during training')
        data_loader_test = make_data_loader(cfg, is_train=False, is_distributed=distributed)


    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        cfg,
        writer,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scaler,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
        dataset_name = cfg.DATASETS['TRAIN'][0],
        data_loader_test=data_loader_test[0],
        logger=logger,
    )

    return model


def run_test(cfg, model, distributed, logger, summary_writer):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  
    dataset_names = cfg.DATASETS.TEST
    data_loaders_test = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for dataset_name, data_loaders_test in zip(dataset_names, data_loaders_test):
        inference(
            model,
            data_loaders_test,
            dataset_name=dataset_name,
            nms_thresh=cfg.TEST.NMS_THRESH,
            device=cfg.MODEL.DEVICE,
            name=cfg.DATASETS['TEST'][0],
            logger=logger,
            summary_writer=summary_writer
        )
        synchronize()

def main():
    parser = argparse.ArgumentParser(description="VLG")
    parser.add_argument(
        "--config-file",
        default="configs/activitynet.yml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument(
        '--enable-tb', 
        action='store_true',
        help="Enable tensorboard logging",
    )
    parser.add_argument(
        '--hps', 
        type=Path, 
        default=Path('non-existent'),
        help='yml file defining the range of hps to be used in training (randomly sampled)')
    
    parser.add_argument(
        '--change-seed', 
        action='store_true',
        help='Set to false to explore randomness in training. ')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )


    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg = set_hps_cfg(cfg,args.hps)

    dataset_name = cfg.DATASETS.TRAIN[0].split('_')[0]
    tags = [dataset_name,
            f'train',
            f'NegProb_{cfg.MODEL.VLG.NEG_PROB}',
            f'NumAggLayers_{cfg.MODEL.VLG.FEATPOOL.NUM_AGGREGATOR_LAYERS}',
            f'NumNeighborsPool_{cfg.MODEL.VLG.FEATPOOL.NUM_NEIGHBOURS}',
            f'NumNeighborsMatch_{cfg.MODEL.VLG.MATCH.NUM_NEIGHBOURS}',
            f'NumIntLayers_{cfg.MODEL.VLG.INTEGRATOR.NUM_AGGREGATOR_LAYERS}',
            f'NumLSTMLayers_{cfg.MODEL.VLG.INTEGRATOR.LSTM.NUM_LAYERS}',
            f'PredictNumLayers_{cfg.MODEL.VLG.PREDICTOR.NUM_STACK_LAYERS}',
            f'LR_{cfg.SOLVER.LR}',
            f'InputStride_{cfg.INPUT.STRIDE}',
            f'NMS-th_{cfg.TEST.NMS_THRESH}'
            ]
    cfg.wandb.tags = tags
    cfg.wandb.name = osp.basename(cfg.OUTPUT_DIR)

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

    # fix seeds for reproducibility
    if not args.change_seed:
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger.info("Using {} GPUs".format(num_gpus))
    # logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, summary_writer, args.local_rank, args.distributed, logger)

    # if len(cfg.DATASETS.TEST) != 0:
    #     best_checkpoint = f"{cfg.OUTPUT_DIR}/model_best_epoch.pth"
    #     if os.path.isfile(best_checkpoint):
    #         model.load_state_dict(torch.load(best_checkpoint))
    #     run_test(cfg, model, args.distributed, logger, summary_writer)
    #     synchronize()

    if args.distributed:
        cleanup()


if __name__ == "__main__":
    main()
