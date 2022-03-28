import torch
from torch import nn
from torch.functional import F

class Syntac_GCN(nn.Module):
    def __init__(self, channel_in, channel_out, dropout=0.0, skip=False):
        super(Syntac_GCN, self).__init__()
        self.w_wc = nn.Sequential(
                    nn.Linear(channel_in*2, channel_out, bias=False),
                    nn.Dropout(dropout),
                    nn.ReLU(True),
                    nn.Linear(channel_out, 1, bias=False),
        )
        self.wd = nn.Sequential(
                    nn.Linear(channel_out, channel_out, bias=False),
                    nn.Dropout(dropout),
        )
        self.skip = skip

    def forward(self, input_):
        queries, wordlens, syntactic_dep = input_
        bs = queries.size(0)
        dim = queries.size(2)
        max_lens = queries.size(1)
        syntactic_dep = syntactic_dep[:,:max_lens,:max_lens]
        output = []
        for b in range(bs):
            b_edge = torch.nonzero(syntactic_dep[b, ...], as_tuple=False)
            i_idx = b_edge[:, 0] 
            j_idx = b_edge[:, 1]
            h_i = queries[b, i_idx, :]
            h_j = queries[b, j_idx, :]
            h = torch.cat((h_i, h_j), 1) # m * 512
            # print(h.shape)
            t = self.w_wc(h) 
            
            # equation 6
            T = torch.ones_like(syntactic_dep[0,...], dtype=torch.float16)*(-100)
            T[i_idx, j_idx]=t.squeeze(-1)
            beta = F.softmax(T, dim=1)

            # equation 7
            H = torch.zeros(max_lens, max_lens, dim, dtype=torch.float16, device=queries.device)
            H[i_idx, j_idx, :] = self.wd(h_j)
            H = H * beta.unsqueeze(-1)
            H = H * (syntactic_dep[b, ...] > 0).unsqueeze(-1)
            b_output = F.relu(queries[b, ...] + torch.sum(H, dim=1)) 
            
            output.append(b_output)
        
        output = torch.stack(output, dim=0)
        if self.skip:
            assert dim == output.shape[2], 'Shape of queris is {}, Shape of output is {}'.format(dim, output.shape)
            output = output + queries
        return [output, wordlens, syntactic_dep]

class lstm_encoder(nn.Module):
    def __init__(self, cfg):
        super(lstm_encoder, self).__init__()

        # Get relevant variables
        hidden_size       = cfg.MODEL.VLG.FEATPOOL.HIDDEN_SIZE
        query_input_size  = cfg.INPUT.PRE_QUERY_SIZE
        query_hidden_size = cfg.MODEL.VLG.INTEGRATOR.QUERY_HIDDEN_SIZE
        num_lstm_layers   = cfg.MODEL.VLG.INTEGRATOR.LSTM.NUM_LAYERS  
        bidirectional     = cfg.MODEL.VLG.INTEGRATOR.LSTM.BIDIRECTIONAL
        dropout_LSTM      = cfg.MODEL.VLG.INTEGRATOR.LSTM.DROPOUT if num_lstm_layers > 1 else 0.0
        dropout_Linear    = cfg.MODEL.VLG.INTEGRATOR.DROPOUT_LINEAR

        # Initialize LSTM
        if bidirectional:
            query_hidden_size //= 2
        self.lstm = nn.LSTM(query_input_size, query_hidden_size, 
                            num_layers=num_lstm_layers, batch_first=True, 
                            dropout=dropout_LSTM, bidirectional=bidirectional)

        #Initialize linear mapping
        self.fc_query = nn.Sequential(
                    nn.Linear(query_hidden_size, hidden_size),
                    nn.Dropout(dropout_Linear),
                    nn.ReLU(True),
        )


    def forward(self, input_):
        queries, wordlens = input_

        self.lstm.flatten_parameters()
        queries = self.lstm(queries)[0]

        return self.fc_query(queries).transpose(1,2)

