import logging
import os
import sys
import os.path as osp

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    if distributed_rank > 0:
        logger.setLevel(logging.ERROR)
        return logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def setup_logger_wandb(config, cfg):
    """
    Configure logger on given level. Logging will occur on standard
    output and in a log file saved in model_dir.
    """
    loglevel = config.get('loglevel', 'INFO')  
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(loglevel))

    log_format = logging.Formatter('%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    if not osp.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    file_handler = logging.FileHandler(osp.join(cfg.OUTPUT_DIR,
                                                '{}.log'.format(osp.basename(cfg.OUTPUT_DIR))))
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    file_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logging.root = logger
    logging.info("save log, checkpoint and code to: {}".format(cfg.OUTPUT_DIR))

    return logger