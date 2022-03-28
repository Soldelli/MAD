import torch
from torch import nn
import numpy as np

def create_mask(lengths, size, negative=False):
    compare = torch.ge if negative else torch.lt
    combine = torch.logical_or if negative else torch.logical_and
    out = torch.arange(size, device=lengths.device)
    out = compare(out, lengths.unsqueeze(-1))
    out = out.unsqueeze(-2).expand(*out.shape[:-1], size, size)
    return combine(out, out.transpose(-2, -1))

def knn(x, y=None, k=10):
    """
    :param x: BxCxN
    :param y: BxCxM
    :param k: scalar
    :return: BxMxk
    """
    x = x.transpose(-1,-2).contiguous()
    if y is None:
        y = x
    else:
        y = y.transpose(-1,-2).contiguous()
    pairwise_distance = -torch.cdist(x, y)
    _, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
    return idx

def knn_plus_scores(x, y=None, k=10):
    """
    :param x: BxCxN
    :param y: BxCxM
    :param k: scalar
    :return: BxMxk
    """
    x = x.transpose(-1,-2).contiguous()
    if y is None:
        y = x
    else:
        y = y.transpose(-1,-2).contiguous()
    pairwise_distance = -torch.cdist(x, y)
    scores, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
    return scores, idx

def knn_plus_scores_masked(x, wordlens, y=None, k=10):
    """
    :param x: BxCxN
    :param y: BxCxM
    :param k: scalar
    :return: BxMxk
    """
    x = x.transpose(-1,-2).contiguous()
    if y is None:
        y = x
    else:
        y = y.transpose(-1,-2).contiguous()
    
    pairwise_distance = -torch.cdist(x, y) + create_mask(wordlens, wordlens.max(), negative=True) * -1e10

    k = min(k, x.shape[1])
    scores, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
    return scores, idx

# get graph feature
def get_graph_feature(x, prev_x=None, k=20, idx_knn=None, r=-1, style=0):
    """
    :param x:
    :param prev_x:
    :param k:
    :param idx:
    :param r: output downsampling factor (-1 for no downsampling)
    :param style: method to get graph feature
    :return:
    """
    batch_size = x.size(0)
    num_points = x.size(2) if prev_x is None else prev_x.size(2)
    x = x.view(batch_size, -1, num_points)
    if prev_x is None:
        prev_x = x

    if idx_knn is None:
        idx_knn = knn(x=x, y=prev_x, k=k)  # (batch_size, num_points, k)
    else:
        k = idx_knn.shape[-1]
    # print(idx_knn.shape)
    device = x.device  # torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_knn + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = prev_x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if style == 0:  # use offset as feature
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    elif style == 1:  # use feature as feature
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    else: # style == 2:
        feature = feature.permute(0,3,1,2)
    # downsample if needed
    if r != -1:
        select_idx = torch.from_numpy(np.random.choice(feature.size(2), feature.size(2) // r,
                                                       replace=False)).to(device=device)
        feature = feature[:, :, select_idx, :]
    return feature, idx_knn

def get_graph_feature_plus_scores(x, prev_x=None, k=20, idx_knn=None, r=-1, style=0):
    """
    :param x:
    :param prev_x:
    :param k:
    :param idx:
    :param r: output downsampling factor (-1 for no downsampling)
    :param style: method to get graph feature
    :return:
    """
    batch_size = x.size(0)
    num_points = x.size(2) if prev_x is None else prev_x.size(2)
    x = x.view(batch_size, -1, num_points)
    if prev_x is None:
        prev_x = x

    if idx_knn is None:
        scores, idx_knn = knn_plus_scores(x=x, y=prev_x, k=k)  # (batch_size, num_points, k)
    else:
        k = idx_knn.shape[-1]
    # print(idx_knn.shape)
    device = x.device  # torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_knn + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = prev_x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if style == 0:  # use offset as feature
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    elif style == 1:  # use feature as feature
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    else: # style == 2:
        feature = feature.permute(0,3,1,2)
    # downsample if needed
    if r != -1:
        select_idx = torch.from_numpy(np.random.choice(feature.size(2), feature.size(2) // r,
                                                       replace=False)).to(device=device)
        feature = feature[:, :, select_idx, :]
    return feature, idx_knn, scores

def get_graph_feature_plus_scores_masked(x, wordlens, prev_x=None, k=20, idx_knn=None, r=-1, style=0):
    """
    :param x:
    :param prev_x:
    :param k:
    :param idx:
    :param r: output downsampling factor (-1 for no downsampling)
    :param style: method to get graph feature
    :return:
    """
    batch_size = x.size(0)
    num_points = x.size(2) if prev_x is None else prev_x.size(2)
    k = min(k, x.shape[-1])
    x = x.view(batch_size, -1, num_points)
    if prev_x is None:
        prev_x = x

    if idx_knn is None:
        scores, idx_knn = knn_plus_scores_masked(x=x, wordlens=wordlens, y=prev_x, k=k)  # (batch_size, num_points, k)
    else:
        k = idx_knn.shape[-1]

    # print(idx_knn.shape)
    device = x.device  # torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_knn + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = prev_x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if style == 0:  # use offset as feature
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    elif style == 1:  # use feature as feature
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    else: # style == 2:
        feature = feature.permute(0,3,1,2)
    # downsample if needed
    if r != -1:
        select_idx = torch.from_numpy(np.random.choice(feature.size(2), feature.size(2) // r,
                                                       replace=False)).to(device=device)
        feature = feature[:, :, select_idx, :]
    return feature, idx_knn, scores

# basic block
class GCNeXt(nn.Module):
    def __init__(self, channel_in, channel_out, k=3, norm_layer=False, groups=32, width_group=4, idx=None):
        """
        input: (bs, ch, 100)
        output: (bs, ch, 100)
        """
        super(GCNeXt, self).__init__()
        self.k = k
        self.groups = groups

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = width_group * groups
        self.tconvs = nn.Sequential(
            nn.Conv1d(channel_in, width, kernel_size=1), nn.ReLU(True),
            nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1), nn.ReLU(True),
            nn.Conv1d(width, channel_out, kernel_size=1),
        ) # temporal graph

        self.sconvs = nn.Sequential(
            nn.Conv2d(channel_in * 2, width, kernel_size=1), nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=1, groups=groups), nn.ReLU(True),
            nn.Conv2d(width, channel_out, kernel_size=1),
        ) # semantic graph

        self.relu = nn.ReLU(True)
        self.idx_list = idx

    def forward(self, x):
        identity = x  # residual
        tout = self.tconvs(x)  # conv on temporal graph

        x_f, idx = get_graph_feature(x, k=self.k, style=1)  # (bs,ch,100) -> (bs, 2ch, 100, k)
        sout = self.sconvs(x_f)  # conv on semantic graph
        sout = sout.max(dim=-1, keepdim=False)[0]  # (bs, ch, 100, k) -> (bs, ch, 100)

        out = tout + identity + sout  # fusion
        if not self.idx_list is None:
            self.idx_list.append(idx)
        return self.relu(out)
