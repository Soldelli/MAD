import os
import json
import h5py
import logging
import random
import numpy as np
import pickle as pk
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from .utils import movie2feats, moment_to_iou2d 

class MADdataset(torch.utils.data.Dataset):

    def __init__(self, ann_file, root, feat_file, tokenizer_folder, num_pre_clips,
                num_clips, pre_query_size, split, fps, test_stride, input_stride, 
                neg_prob, lang_feat_file, lang_feat_type):
        super(MADdataset, self).__init__()
        self.max_words = 0
        self.split = split
        self.num_clips = num_clips
        self.num_pre_clips = num_pre_clips
        self.fps = fps
        self.input_stride = input_stride
        self.test_stride = test_stride
        self.relative_fps = self.fps * self.num_clips / (self.num_pre_clips * input_stride)
        self.relative_stride = int(round(self.test_stride * self.num_clips 
                                                        / self.num_pre_clips))
        self.neg_prob = neg_prob

        #load annotation file
        annos = json.load(open(ann_file,'r'))

        cache = ann_file+'.pickle'
        if os.path.exists(cache):
            # if cached data exist load it.
            self.load_pickle_data(cache)
        else:
            # otherwise compute the annotations information
            self._compute_annotaions(annos, cache)

        # Get correct data for language
        if lang_feat_type == 'clip' and os.path.exists(lang_feat_file):
            self.load_clip_lang_feats(lang_feat_file)
        else:
            raise ValueError('Select a correct type of lang feat - Glove is deprecated.')

        self.movies = {a['movie']:a['movie_duration'] for a in self.annos}
        self.feats = movie2feats(feat_file, self.movies.keys())

        if self.max_words > 50:
            self.max_words = 50
        
        if not split == 'train':
            self._compute_windows_per_movie(test_stride)

    def __getitem__(self, idx):
        anno = self.annos[idx]
        query, wordlen = self._get_language_feature(anno)
        if self.split == 'train':
            feat, iou2d = self._get_video_features_train(anno, anno['movie'])
        else:
            feat, iou2d = self._get_video_features_test(anno, anno['movie'])
        return feat, query, wordlen, iou2d, idx

    def __len__(self):
        return len(self.annos)
    
    def get_duration(self, idx):
        return self.annos[idx]['movie_duration']
    
    def get_relative_stride(self):
        return self.relative_stride

    def get_relative_fps(self):
        return self.relative_fps

    def get_sentence(self, idx):
        return self.annos[idx]['sentence']
    
    def get_moment(self, idx):
        '''
            return moment duration in seconds
        '''
        return self.annos[idx]['moment']
    
    def get_vid(self, idx):
        return self.annos[idx]['movie']

    def get_number_of_windows(self,idx):
        movie = self.annos[idx]['movie']
        return len(self.windows[movie])

    def load_pickle_data(self,cache):
        '''
            The function loads preprocesses annotations and compute the max lenght of the sentences.

            INPUTS:
            cache: path to pickle file from where to load preprocessed annotations

            OUTPUTS:
            None.
        '''
        logger = logging.getLogger("vlg.trainer")
        logger.info("Load cache data, please wait...")
        self.annos = pk.load(open(cache, 'rb'))
        
    def load_clip_lang_feats(self, file):
        with h5py.File(file, 'r') as f:
            for i, anno in enumerate(self.annos):
                lang_feat = f[anno['id']][:]
                self.annos[i]['query'] = torch.from_numpy(lang_feat).float()
                self.annos[i]['wordlen'] = len(lang_feat)

        self.max_words = max([a['wordlen'] for a in self.annos])

    def _compute_annotaions(self, annos, cache):
        '''
            The function processes the annotations computing language tokenizationa and query features.
            Construct the moment annotations for training and the target iou2d map.
            Processed the language to obtain syntactic dependencies.
            Dump everything in the pickle file for speading up following run.

            INPUTS:
            annos: annotations loaded from json files
            cache: path to pickle file where to dump preprocessed annotations

            OUTPUTS:
            None.
        '''
        # compute the annotation data and dump it in a pickle file
        self.annos = []
        logger = logging.getLogger("vlg.trainer")
        logger.info("Preparing data, please wait...")
        for k, anno in tqdm(annos.items()):
            # Unpack Info ----------------------------------------------------------------
            
            movie     = anno['movie']
            duration  = anno['movie_duration']
            timestamp = anno['ext_timestamps']
            sentence  = anno['sentence']

            # Process gt annotations -----------------------------------------------------
            if timestamp[0] < timestamp[1]:
                moment = torch.tensor([max(timestamp[0], 0), min(timestamp[1], duration)])

            start = int(torch.round(moment[0] * self.fps))
            stop = int(torch.round(moment[1] * self.fps))
            frames_idx = [start, stop]
                      
            # Save preprocessed annotations ----------------------------------------------
            dump_dict = {
                    'id'            : k,
                    'movie'         : movie,
                    'moment'        : moment,
                    'frames_idx'    : frames_idx,
                    'sentence'      : sentence,
                    'movie_duration': duration,
                }

            self.annos.append(dump_dict)
        
        # save to file
        pk.dump(self.annos,open(cache,'wb'))

    def _get_language_feature(self, anno):
        '''
            INPUTS:
            anno: annotation data, contains all the preprocessed information

            OUTPUTS:
            query: features of the selected sentence
            wordlen: length of the selected sentence 
        '''
        query = anno['query'][:self.max_words]
        wordlen = min(anno['wordlen'], self.max_words)
        return query, wordlen

    def _get_video_features_train(self, anno, movie):
        '''
            INPUTS:
            anno: annotation data, contains all the preprocessed information
            movie: movie id to select the correct features

            OUTPUTS:
            feat: movie features
            iou2d: target matrix 
        '''

        if random.random() > self.neg_prob:
            moment = anno['moment']
            start_idx, stop_idx = anno['frames_idx']
            num_frames = stop_idx - start_idx
            
            if num_frames < self.num_pre_clips:
                offset = random.sample(range(0, self.num_pre_clips - num_frames, 1),1)[0]
            else:
                center = (start_idx + stop_idx) /2
                offset = int(round(center / 2))

            # Compute features for window
            start_window = max(start_idx - offset, 0)
            stop_window  = start_window + self.num_pre_clips * self.input_stride

            if not stop_window <= anno['movie_duration']*self.fps:
                stop_window = int(anno['movie_duration']*self.fps)
                start_window = stop_window - self.num_pre_clips * self.input_stride

            feats = self.feats[movie][start_window:stop_window: self.input_stride]

            assert feats.shape[0] == self.num_pre_clips

            # Compute moment position withint the windo
            duration = self.num_pre_clips /self.fps
            start_moment = max((start_idx - start_window) /self.fps, 0)
            stop_moment = min((stop_idx - start_window) /self.fps,  duration)
            
            moment = torch.tensor([start_moment, stop_moment])
            # Generate targets for training ----------------------------------------------
            iou2d = moment_to_iou2d(moment, self.num_clips, duration)
        else:
            start_window = random.randint(0, self.feats[movie].shape[0]-self.num_pre_clips * self.input_stride)
            stop_window = start_window + self.num_pre_clips* self.input_stride

            if not stop_window <= anno['movie_duration']*self.fps:
                stop_window = int(anno['movie_duration']*self.fps)
                start_window = stop_window - self.num_pre_clips * self.input_stride

            feats = self.feats[movie][start_window:stop_window: self.input_stride]
            iou2d = torch.zeros(self.num_clips, self.num_clips)

            assert feats.shape[0] == self.num_pre_clips

        return feats, iou2d

    def _get_video_features_test(self, anno, movie):
        '''
            INPUTS:
            anno: annotation data, contains all the preprocessed information
            movie: movie id to select the correct features

            OUTPUTS:
            feat: movie features
            iou2d: target matrix 
        '''

        windows = self.windows[movie]
        windows_feats = torch.stack([self.feats[movie][w[0]:w[1]] for w in windows])
        return windows_feats, torch.empty((1))

    def _compute_windows_per_movie(self, test_stride):
        '''
            INPUTS:
            anno: annotation data, contains all the preprocessed information
            movie: movie id to select the correct features

            OUTPUTS:
            feat: movie features
            iou2d: target matrix 
        '''

        self.windows = {}
        for m in self.movies.keys():
            num_feats = len(self.feats[m])

            starts = torch.arange(0, num_feats - self.num_pre_clips, test_stride, dtype=torch.int)
            stops = starts + self.num_pre_clips
            
            self.windows[m] = torch.stack([starts,stops]).transpose(1,0)

