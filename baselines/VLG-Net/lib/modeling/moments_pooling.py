import torch
from torch import nn


class Cross_Attention_Pooling(nn.Module):

    def __init__(self, num_anchors, anchors, num_clips, device, hidden_size):
        super(Cross_Attention_Pooling, self).__init__()
        self.M = num_anchors
        self.mask = torch.ones((num_anchors,num_clips),device=device) * 1e10   #Num_proposals x NUm_clips
        for i,anchor in enumerate(anchors.int()[:num_anchors]):
            self.mask[i,anchor[1]:anchor[2]+1] = 0
        self.mask = self.mask.unsqueeze(0)
        self.conv1d_lang =  nn.Conv1d(in_channels=hidden_size, out_channels=1, kernel_size=1)

    def _create_query_mask(self, wordlens):
        q_mask = torch.zeros((len(wordlens),1,max(wordlens)))
        for i,l in enumerate(wordlens): 
            q_mask[i,:,l:] = 1e10
        if torch.cuda.is_available():
            return q_mask.to('cuda')
        else:
            return q_mask

    def _compute_sentence_representation(self, queries, wordlens):
        '''
        Compute attention pooled feature for language as sentence representation
        '''
        q_mask = self._create_query_mask(wordlens)
        similarity = self.conv1d_lang(queries.transpose(1,2))
        scores = torch.softmax(similarity-q_mask, dim=2)
        return torch.bmm(scores,queries)

    def forward(self, x, anchors, queries, wordlens):
        ''' Compute attention pooling on top of clips features
        Args:
            x: BxDxNUM_CLIPS float tensor.
            anchors: (BxNUM_PROPOSALS)x3 int tensor
            queries: BxD
            wordlens: BxMAX_WORDS_IN_BATCHxD
        Returns:
            output: BxNUM_PROPOSALSxD float tensor.
        '''
        B,D,N = x.shape
        queries = self._compute_sentence_representation(queries, wordlens)
        similarity = torch.bmm(queries, x).expand(B, self.M, N) - self.mask.expand(B, self.M, N)
        scores = torch.softmax(similarity, dim=2)
        return torch.bmm(scores,x.transpose(1,2))


class Learnable_Cross_Attention_Pooling(nn.Module):

    def __init__(self, num_anchors, anchors, num_clips, device, hidden_size):
        super(Learnable_Cross_Attention_Pooling, self).__init__()
        self.M = num_anchors
        self.mask = torch.ones((num_anchors,num_clips),device=device) * 1e10   #Num_proposals x NUm_clips
        for i,anchor in enumerate(anchors.int()[:num_anchors]):
            self.mask[i,anchor[1]:anchor[2]+1] = 0
        self.mask = self.mask.unsqueeze(0)
        self.conv1d =  nn.Conv1d(in_channels=2*hidden_size, out_channels=1, kernel_size=1)
        self.conv1d_lang =  nn.Conv1d(in_channels=hidden_size, out_channels=1, kernel_size=1)

    def _create_query_mask(self, wordlens):
        q_mask = torch.zeros((len(wordlens),1,max(wordlens)))
        for i,l in enumerate(wordlens): 
            q_mask[i,:,l:] = 1e10
        if torch.cuda.is_available():
            return q_mask.to('cuda')
        else:
            return q_mask

    def _compute_sentence_representation(self, queries, wordlens):
        '''
        Compute attention pooled feature for language as sentence representation
        '''
        q_mask = self._create_query_mask(wordlens)
        similarity = self.conv1d_lang(queries.transpose(1,2))
        scores = torch.softmax(similarity-q_mask, dim=2)
        return torch.bmm(scores,queries).squeeze(dim=1)

    def forward(self, x, anchors, queries, wordlens):
        ''' Compute attention pooling on top of clips features
        Args:
            x: BxDxNUM_CLIPS float tensor.
            anchors: (BxNUM_PROPOSALS)x3 int tensor
            queries: BxD
            wordlens: BxMAX_WORDS_IN_BATCHxD
        Returns:
            output: BxNUM_PROPOSALSxD float tensor.
        '''
        B,D,N = x.shape
        queries = self._compute_sentence_representation(queries, wordlens).unsqueeze(2).expand(B,D,N)
        similarity = self.conv1d(torch.cat((queries,x),dim=1)).expand(B, self.M, N) - self.mask.expand(B, self.M, N)
        scores = torch.softmax(similarity, dim=2)
        return torch.bmm(scores,x.transpose(1,2))


class Self_Attention_Pooling(nn.Module):

    def __init__(self, num_anchors, anchors, num_clips, device, hidden_size):
        super(Self_Attention_Pooling, self).__init__()
        self.M = num_anchors
        self.mask = torch.ones((num_anchors,num_clips),device=device) * 1e10   #Num_proposals x NUm_clips
        for i,anchor in enumerate(anchors.int()[:num_anchors]):
            self.mask[i,anchor[1]:anchor[2]+1] = 0
        self.mask = self.mask.unsqueeze(0)
        self.conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=1, kernel_size=1)

    def forward(self, x, anchors, queries, wordlens):
        ''' Compute attention pooling on top of clips features
        Args:
            x: BxDxNUM_CLIPS float tensor.
            anchors: (BxNUM_PROPOSALS)x3 int tensor
            queries: Not used (compatibility with cross attention )
            wordlens: Not used (compatibility with cross attention )
        Returns:
            output: BxNUM_PROPOSALSxD float tensor.
        '''
        B,D,N = x.shape
        similarity = self.conv1d(x).expand(B, self.M, N) - self.mask.expand(B, self.M, N)
        scores = torch.softmax(similarity, dim=2)
        return torch.bmm(scores,x.transpose(1,2))
