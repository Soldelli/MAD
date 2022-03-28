import torch
from torch import nn
from torch.functional import F

from lib.utils.gcnext import get_graph_feature_plus_scores, get_graph_feature_plus_scores_masked

class Graph_Matching_Module(nn.Module):
    def __init__(self, cfg):
        super(Graph_Matching_Module, self).__init__()

        # Initialize graph matching layers------------------------------------------
        self.setup_graph_matchin_operations(cfg)

        # Ourput projection 1x1 conv -----------------------------------------------
        hidden_size = cfg.MODEL.VLG.FEATPOOL.HIDDEN_SIZE
        self.conv1x1 = nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
                nn.ReLU(True),
        )

    def setup_graph_matchin_operations(self,cfg):
        prm = dict(
            hidden_size = cfg.MODEL.VLG.FEATPOOL.HIDDEN_SIZE,
            k = cfg.MODEL.VLG.MATCH.NUM_NEIGHBOURS,
            groups = cfg.MODEL.VLG.MATCH.GROUPS, 
            width_group = cfg.MODEL.VLG.MATCH.WIDTH_GROUP,
            enable_ordering = cfg.MODEL.VLG.MATCH.ORDERING_EDGE, 
            enable_semantic = cfg.MODEL.VLG.MATCH.SEMANTIC_EDGE,
            enable_matching = cfg.MODEL.VLG.MATCH.MATCHING_EDGE,
            dropout = cfg.MODEL.VLG.MATCH.DROPOUT_GM,
        )

        gm_layers = [BidirectionalGraphMatching(**prm)]       
        self.graph_matching = nn.Sequential(*gm_layers)


    def forward(self, x, queries, wordlens):
        '''
            INPUTS:
            x : Tensor = [B, Feat_dimension, temporal_resolution] 
            queries: Tensor = [B, Feat_dimension, max_num_words_in_batch]
            wordlens: Tensor = [B]

            OUTPUTS:
            output: Tensor of shape [B, N, Feat_dimension]
        '''
        v, q, _ = self.graph_matching([x, queries, wordlens]) 
        return self.conv1x1(v), q

class BidirectionalGraphMatching(nn.Module):
	'''
		TODO: Write 
	'''
	def __init__(self, hidden_size, k=3, norm_layer=False, groups=32, width_group=4, 
							idx=None, enable_ordering=True, enable_semantic=True,
							enable_matching=True, enable_skip_connection=True, dropout=0.0):

		super().__init__()

		self.k = k
		self.groups = groups

		if norm_layer:
				norm_layer = nn.BatchNorm1d
		width = width_group * groups

		self.tconvs = nn.ModuleDict({
				'video': nn.Sequential(
						nn.Conv1d(hidden_size, width, kernel_size=1), nn.ReLU(True),
						nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1), nn.ReLU(True),
						nn.Conv1d(width, hidden_size, kernel_size=1),
				),
				'query': nn.Sequential(
						nn.Conv1d(hidden_size, width, kernel_size=1), nn.ReLU(True),
						nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1), nn.ReLU(True),
						nn.Conv1d(width, hidden_size, kernel_size=1),
				)             
		}) # video temporal graph

		self.vconvs = nn.ModuleDict({
				'video': nn.Sequential(
						nn.Conv1d(hidden_size*2, width, kernel_size=1), nn.ReLU(True),
						nn.Conv1d(width, width, kernel_size=1, groups=groups), nn.ReLU(True),
						nn.Conv1d(width, hidden_size, kernel_size=1),
				),
				'query': nn.Sequential(
						nn.Conv1d(hidden_size*2, width, kernel_size=1), nn.ReLU(True),
						nn.Conv1d(width, width, kernel_size=1, groups=groups), nn.ReLU(True),
						nn.Conv1d(width, hidden_size, kernel_size=1),
				)                      
		}) # video semantic graph

		self.qconvs = nn.ModuleDict({
				'video': nn.Sequential(
						nn.Conv1d(hidden_size*2, width, kernel_size=1), nn.ReLU(True),
						nn.Conv1d(width, width, kernel_size=1, groups=groups), nn.ReLU(True),
						nn.Conv1d(width, hidden_size, kernel_size=1),
				),
				'query': nn.Sequential(
						nn.Conv1d(hidden_size*2, width, kernel_size=1), nn.ReLU(True),
						nn.Conv1d(width, width, kernel_size=1, groups=groups), nn.ReLU(True),
						nn.Conv1d(width, hidden_size, kernel_size=1),
				)
		}) # query graph

		self.relu = nn.ReLU(True)

		self.ord, self.sem, self.match, self.skip = 1.0, 1.0, 1.0, 1.0
		if not enable_ordering:
			self.ord = 0.0
		if not enable_semantic:
			self.sem = 0.0
		if not enable_matching:
			self.match = 0.0
		if not enable_skip_connection:
			self.skip = 0.0

	def _pairwise_dot_product_similarity(self, x, y):
		"""Compute the dot product similarity between x and y.
		Args:
			x: BxDxN float tensor.
			y: BxDxM float tensor.

		Returns:
			s: BxNxM float tensor, the pairwise dot product similarity.
		"""
		return torch.bmm(x.transpose(2, 1), y)

	def _compute_cross_attention(self, v, q):
		"""Compute cross attention.
		Args:
			x: BxDxN float tensor.
			y: BxDxM float tensor.

		Returns:
			attention_x: BxDxN float tensor.
			attention_y: BxDxM float tensor.
		"""
		a = self._pairwise_dot_product_similarity(v, q)
		a_v = torch.softmax(a, dim=2)  # i->j
		a_q = torch.softmax(a, dim=1)  # j->i
		attention_v = torch.bmm(q, a_v.transpose(1, 2))
		attention_q = torch.bmm(v, a_q)
		return attention_v, attention_q

	def _create_query_mask(self, wordlens):
		mask = wordlens.new_zeros((len(wordlens),1,max(wordlens)))
		for i,l in enumerate(wordlens): 
			mask[i,:,l:] = 1e10
		return mask


	def _create_query_mask2(word_lengths):
		x = F.one_hot(word_lengths)
		x = x.narrow(-1, 0, x.shape[-1] - 1)
		return x.cumsum_(dim=-1).mul_(int(1e10)).unsqueeze_(-2)

	def update_graph(self, feat, att, modality_key, wordlens):
		if wordlens is None:
			neighbours, idx, scores = get_graph_feature_plus_scores(feat, k=self.k, style=2)
		else:
			neighbours, idx, scores = get_graph_feature_plus_scores_masked(feat, wordlens, k=self.k, style=2)
		scores = torch.softmax(scores, dim=2)
		neighbours = (neighbours * scores[:,None,:]).sum(dim=-1)

		# Compute semantic edge aggregation
		sem_featx2 = torch.cat([feat, neighbours],dim=1)
		sem_feat = self.vconvs[modality_key](sem_featx2) # (B, D, T, k) -> (B, D, T)
		
		# Compute matching edge aggregation
		match_featx2 = torch.cat([feat, att], dim=1)
		match_feat = self.qconvs[modality_key](match_featx2)  # (B, D, L, k) -> (B, D, T)

		# Compute ordering edge aggregation (convolution with kernel=3)
		ord_feat = self.tconvs[modality_key](feat)

		# Construct new graph (with possible skip connection)
		out = sem_feat * self.sem   + \
			match_feat * self.match + \
			ord_feat   * self.ord   + \
			feat       * self.skip
		return out

	def forward(self, input_):
		"""
			input_ = [video_feat, query_feat, wordlens]
			video_feat: B,D,T
			query_feat: B,D,L

			T = Number of features for video
			L = Number of features for language
			D = feature size
		"""
		video_feat, query_feat, wordlens = input_

		B,D,T  = video_feat.shape
		B,Dq,L = query_feat.shape
		# assert Dq == D, 'video query embedding should be locate in the same space, meet {} v.s. {}'.format(Dq,D)

		#Preprocess features --------------------------------------------------------------
		video_feat = video_feat.contiguous()  # (B, D, T)
		query_feat = query_feat.contiguous()  # (B, D, L)

		# Compute similarities and cross attention ----------------------------------------
		v_att, q_att = self._compute_cross_attention(video_feat, query_feat)

		# Video graph ---------------------------------------------------------------------
		video_graph = self.update_graph(video_feat, v_att, 'video', None)
	
		# Query graph ---------------------------------------------------------------------
		query_graph = self.update_graph(query_feat, q_att, 'query', wordlens)

		return [video_graph, query_graph, wordlens]

if __name__ == "__main__":

	raise ValueError ('Check the implementation of the unit test.')
	model = GraphMatching(512).cuda()
	video_feat = torch.randn((2,1000,512,8)).cuda()
	query_feat = torch.randn((2,10,512)).cuda()
	out = model(video_feat, query_feat)
