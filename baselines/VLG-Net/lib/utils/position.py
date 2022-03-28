import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe)
        # print(x.shape) # 16,400,100
        # print(self.pe.shape)
        # exit(0)
        x = x + self.pe
        return self.dropout(x)


class PositionalEncoding2d(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, max_len=100, dropout=0.0, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        
        pe = torch.zeros(num_pos_feats,max_len,max_len)
        x_embed = torch.arange(0, max_len, dtype=torch.float).view(max_len,1).expand(max_len,max_len)
        y_embed = torch.arange(0, max_len, dtype=torch.float).view(1,max_len).expand(max_len,max_len)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)

        pos = pos.unsqueeze(0)
        self.register_buffer('pos2d', pos)
        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # return self.dropout(x + self.pos2d)
        return x + self.pos2d
