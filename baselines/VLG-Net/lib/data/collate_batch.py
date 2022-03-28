import torch
from torch.nn.utils.rnn import pad_sequence

from lib.structures import TLGBatch


class BatchCollator(object):
    """
    Collect batch for dataloader
    """

    def __init__(self, ):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # [xxx, xxx, xxx], [xxx, xxx, xxx] ......
        feats, queries, wordlens, ious2d, idxs = transposed_batch
        return TLGBatch(
            feats=torch.stack(feats).float(),
            queries=pad_sequence(queries).transpose(0, 1),
            wordlens=torch.tensor(wordlens),
        ), torch.stack(ious2d), idxs
