import copy
import torch
from torch import nn
from torch.functional import F
from lib.utils.gcnext import *
from lib.utils.position import *
from lib.modeling.loss import TanLoss
from lib.modeling.language_modeling import lstm_encoder
from lib.modeling.graph_matching import Graph_Matching_Module
from lib.modeling.moments_pooling import *

import time
class VLG(nn.Module):
    def __init__(self, cfg):
        super(VLG, self).__init__()

        # Video Preprocessing --------------------------------------------------------------------
        self.set_video_only_operations(cfg)
        
        # Language Preprocessing -----------------------------------------------------------------
        self.set_language_only_operations(cfg)

        # Clips Graph Matching -------------------------------------------------------------------
        self.set_graph_matching_operations(cfg)

        # Anchors Definition ---------------------------------------------------------------------
        self.compute_anchors(cfg)
        
        # Moment Pooling -------------------------------------------------------------------------
        self.set_moment_pooling_operations(cfg)

        # Prediction -----------------------------------------------------------------------------
        self.set_proposals_scoring_operations(cfg)

        # Loss -----------------------------------------------------------------------------------
        self.set_loss(cfg)
        
    def set_1d_pos_encoder(self, cfg):
        # pos 1D
        input_size = cfg.MODEL.VLG.FEATPOOL.INPUT_SIZE
        if cfg.MODEL.VLG.FEATPOOL.POS == 'none':
            self.pos_encoder = nn.Identity()
        elif cfg.MODEL.VLG.FEATPOOL.POS == 'cos':
            self.pos_encoder = PositionalEncoding(input_size, dropout=0.0, max_len=cfg.MODEL.VLG.NUM_CLIPS)
        else:
            raise ValueError('cfg.MODEL.VLG.FEATPOOL.POS is not defined:', cfg.MODEL.VLG.FEATPOOL.POS)

    def set_2d_pos_encoder(self, cfg, N, input_size):
        pos = None
        if cfg.MODEL.VLG.PREDICTOR.POS == 'none':
            pos = nn.Identity()
        elif cfg.MODEL.VLG.PREDICTOR.POS == 'cos':
            pos = PositionalEncoding2d(input_size//2, max_len=N, dropout=0.0)
        else:
            raise ValueError('cfg.MODEL.VLG.PREDICTOR.POS is not defined:', cfg.MODEL.VLG.PREDICTOR.POS)
        return pos

    def set_video_only_operations(self, cfg):
        self.set_1d_pos_encoder(cfg)

        input_size  = cfg.MODEL.VLG.FEATPOOL.INPUT_SIZE
        hidden_size = cfg.MODEL.VLG.FEATPOOL.HIDDEN_SIZE
        kernel_size = cfg.MODEL.VLG.FEATPOOL.KERNEL_SIZE
        stride      = cfg.INPUT.NUM_PRE_CLIPS // cfg.MODEL.VLG.NUM_CLIPS
        dropout     = cfg.MODEL.VLG.FEATPOOL.DROPOUT

        # Setup pooling ops.
        self.feat_pool = nn.Sequential(
                nn.AvgPool1d(kernel_size, stride),
                self.pos_encoder,
                nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
        
        # Setup GCNeXt
        prm_GCNeXt = dict(
            channel_in = hidden_size, 
            channel_out = hidden_size,
            k = cfg.MODEL.VLG.FEATPOOL.NUM_NEIGHBOURS, 
            groups = cfg.MODEL.VLG.FEATPOOL.GROUPS, 
            width_group = cfg.MODEL.VLG.FEATPOOL.WIDTH_GROUP
        )
        layers = [GCNeXt(**prm_GCNeXt)]* cfg.MODEL.VLG.FEATPOOL.NUM_AGGREGATOR_LAYERS  
        self.context_aggregator = nn.Sequential(*layers)
                     
    def set_language_only_operations(self, cfg):
        '''
            Initialize one class per setup, easiest way to wrap different operations.
            Syntactic Dependencies have been removed from this version of VLG-Net.
            See official implementation for more details. 
        '''
        self.language_encoder = lstm_encoder(cfg)
        
    def set_graph_matching_operations(self, cfg):
        self.clip_level_fusion = Graph_Matching_Module(cfg)

    def set_moment_pooling_operations(self, cfg):

        prm = dict(
            num_anchors=self.n_anchor, 
            anchors=self.anchors, 
            num_clips=cfg.MODEL.VLG.NUM_CLIPS, 
            device=cfg.MODEL.DEVICE, 
            hidden_size=cfg.MODEL.VLG.FEATPOOL.HIDDEN_SIZE
        )
        
        if cfg.MODEL.VLG.MOMENT_POOLING.ATTENTION_MODE == 'cross':
            self.masked_attention_pooling = Cross_Attention_Pooling(**prm) 

        elif cfg.MODEL.VLG.MOMENT_POOLING.ATTENTION_MODE == 'cross_learnable':
            self.masked_attention_pooling = Learnable_Cross_Attention_Pooling(**prm) 
                                
        elif cfg.MODEL.VLG.MOMENT_POOLING.ATTENTION_MODE == 'self':
            self.masked_attention_pooling = Self_Attention_Pooling(**prm) 

        else:
            raise ValueError ('Select correct type of attention pooling.')

    def set_loss(self, cfg):
        min_iou = cfg.MODEL.VLG.LOSS.MIN_IOU
        max_iou = cfg.MODEL.VLG.LOSS.MAX_IOU
        self.tanloss = TanLoss(min_iou, max_iou, mask2d=self.mask2d)

    def compute_anchors(self, cfg):
        
        pooling_counts = cfg.MODEL.VLG.FEAT2D.POOLING_COUNTS
        N = cfg.MODEL.VLG.NUM_CLIPS
        B = max(cfg.SOLVER.BATCH_SIZE, cfg.TEST.BATCH_SIZE)
        
        # same anchor as in 2D TAN
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1
        
        stride, offset = 1, 0
        maskij = [(i,i) for i in range(N)]
        for c in pooling_counts:
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij += list(zip(i, j))
            stride *= 2

        # save anchors
        n_anchor = len(maskij)
        anchors = torch.tensor(maskij).repeat((B, 1))
        batch_id = torch.tensor([[k] * n_anchor for k in range(B)]).view(-1, 1)
        self.anchors = torch.cat([batch_id, anchors], dim=-1).int().to(device=cfg.MODEL.DEVICE)
        self.n_anchor = n_anchor

        self.maskij = maskij
        (self.__i, self.__j) = zip(*self.maskij)
        self.mask2d = mask2d.to("cuda")

    def mask2weight(self, mask2d, mask_kernel, padding=0):
        # from the feat2d.py,we can know the mask2d is 4-d
        weight = torch.conv2d(mask2d[None,None,:,:].float(),
                mask_kernel, padding=padding)[0, 0]
        weight[weight > 0] = 1 / weight[weight > 0]
        return weight

    def set_proposals_scoring_operations(self, cfg):
        input_size  = cfg.MODEL.VLG.FEATPOOL.HIDDEN_SIZE
        hidden_size = cfg.MODEL.VLG.PREDICTOR.HIDDEN_SIZE
        num_layers  = cfg.MODEL.VLG.PREDICTOR.NUM_STACK_LAYERS
        kernal_size = cfg.MODEL.VLG.PREDICTOR.KERNEL_SIZE
        dropout     = cfg.MODEL.VLG.PREDICTOR.DROPOUT_CONV

        # Generate weights to remove proposals during training
        mask_kernel = torch.ones(1, 1, kernal_size, kernal_size).to(self.mask2d.device)
        first_padding = (kernal_size - 1) * num_layers // 2
        weights = [self.mask2weight(self.mask2d, mask_kernel, padding=first_padding) ]
        for _ in range(num_layers - 1):
            weights.append(self.mask2weight(weights[-1] > 0, mask_kernel))
        self.weights = weights

        # Instantiate the 2d pos embedding with the right dimension
        self.pos2d_encoder = self.set_2d_pos_encoder(cfg, cfg.MODEL.VLG.NUM_CLIPS, input_size)
        convs_pred = nn.ModuleList(
            ([nn.Sequential(
                nn.Conv2d(input_size, hidden_size, kernal_size, padding=first_padding), 
                nn.Dropout2d(dropout),
                nn.ReLU(True),
            )] +
            [nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size, kernal_size), 
                nn.Dropout2d(dropout),
                nn.ReLU(True)
            )] * (num_layers - 1) 
            
        ))
        self.convs_pred = nn.ModuleList(convs_pred)
        self.iou        = nn.Conv2d(hidden_size, 1, 1)
         
    def reshape2d(self, x, N):
        B, _, d = x.shape
        x2d = x.new_zeros(B, d, N, N)
        x2d[:, :, self.__i, self.__j] = x.transpose(1, 2)  
        return x2d

    def predict_scores(self, x, queries, wordlens):
        # Pass through several conv1D layers
        for conv, weight in zip(self.convs_pred, self.weights):
            x = conv(x) * weight
        
        # Generate score for each proposal
        return self.iou(x).squeeze_()

    def video_encoder(self, x):
        x = self.feat_pool(x.transpose(1, 2))  
        x = self.context_aggregator(x)
        return x

    def fuse_and_score(self, x, queries, wordlens):
        # Graph matching 
        x, queries = self.clip_level_fusion(x, queries, wordlens)

        # Mmoment scoring
        B = queries.shape[0]
        N = x.shape[-1]
        anchor = self.anchors[:self.n_anchor * B, :]

        # Generate moments proposals
        x = self.masked_attention_pooling(x, anchor, queries.transpose(1,2), wordlens)
        
        # Reshape to 2D to use conv layers for scoring
        x = self.reshape2d(x, N)  
        x = self.pos2d_encoder(x)   # pos embedding
        x = self.predict_scores(x, queries, wordlens)  # conv layers + GM 
        return x
 
    def forward(self, feats, queries, wordlens, ious2d=None):
        # Video feat encoder
        x = self.video_encoder(feats)

        # Query feat encoder
        queries = self.language_encoder([queries, wordlens]) 
        
        # Rest of architecture
        x = self.fuse_and_score(x, queries, wordlens)
        
        # Compute loss 
        if not ious2d is None and self.training:
            return self.tanloss(x, ious2d)
        
        return x.sigmoid_() * self.mask2d.float()  


if __name__ == '__main__':
    #TODO: Rewrite unit test
    raise ValueError('Deprecated - Needs rewriting')
