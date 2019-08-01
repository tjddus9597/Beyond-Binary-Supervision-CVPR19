from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from tqdm import trange
import _pickle as pickle
import math

class EvalDataset(datasets.ImageFolder):

    def __init__(self, dir, batch_size, transform=None, *arg, **kw):
        super(EvalDataset, self).__init__(dir,transform)
        
        self.batch_size = batch_size
        self.eval_pairs = self.generate_evals(self,self.imgs)

    @staticmethod
    def generate_evals(self, imgs):
        def create_indices(_imgs):
            inds = dict()
            for idx, (img_path,label) in enumerate(_imgs):
                if label not in inds:
                    inds[label] = []
                inds[label].append(img_path)
            return inds

        indices = create_indices(imgs)          
        pairs = []
        num_data_val = len(indices[1])      
        for idx_bat in trange(math.ceil(num_data_val / self.batch_size)):            
            pair = []              
            idx_val_stt = idx_bat *  self.batch_size
            idx_val_end = min((idx_bat+1) * self.batch_size, num_data_val)   
            pair = indices[1][idx_val_stt:idx_val_end]                                      
            pairs.append(pair)
        return pairs

    def __getitem__(self, index):
        '''
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        '''
        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            return self.transform(img)

        # Get the index of each image in the triplet
        input = torch.FloatTensor(self.batch_size,3,224,224)
        pair_size = len(self.eval_pairs[index])
        for idx in range(pair_size):
            imgs_path = self.eval_pairs[index][idx]
            # transform images if required        
            input[idx] = transform(imgs_path)
        return input

    def __len__(self):
        return len(self.eval_pairs)