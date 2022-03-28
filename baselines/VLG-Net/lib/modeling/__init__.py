from .model import VLG
ARCHITECTURES = {"VLG": VLG}

def build_model(cfg):
    return ARCHITECTURES[cfg.MODEL.ARCHITECTURE](cfg)
