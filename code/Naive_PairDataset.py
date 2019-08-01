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
import random
class Naive_PairDataset(datasets.ImageFolder):
    def __init__(self, dir, is_trn, batch_size, epoch, transform = None, *arg, **kw):
        super(Naive_PairDataset, self).__init__(dir, transform)
        self.batch_size = batch_size
        self.num_nn_pos   = 30
        self.num_pos_max = 5
        self.epoch = epoch
        self.is_trn = is_trn
        self.list_nn, self.dist_nn = self.load_nn(dir)
        self.training_pairs = self.generate_pairs(self,self.is_trn,self.imgs)

    @staticmethod
    def load_nn(dir):
        if os.path.isfile(os.path.join(dir, 'list_nn.pkl')):
            with open(os.path.join(dir,'list_nn.pkl'),'rb') as f:
                list_nn = pickle.load(f)
            list_nn['trn'] = list_nn['trn']     # python base 0
            list_nn['val'] = list_nn['val']     # python base 0
            with open(os.path.join(dir,'dist_nn.pkl'),'rb') as f:
                dist_nn = pickle.load(f)
        return list_nn, dist_nn

    @staticmethod
    def generate_pairs(self, is_trn, imgs):
        def create_indices(_imgs):
            inds = dict()
            for idx, (img_path,label) in enumerate(_imgs):
                if label not in inds:
                    inds[label] = []
                inds[label].append(img_path)
            return inds

        indices = create_indices(imgs)          
        pairs = []
        dist_pairs = []
        if is_trn:
            num_data_trn = len(indices[0])
            
            # sample postivie/negative pairs randomly
            # pos/neg neighbors in the batch
            num_pos_all = self.num_nn_pos
            num_neg_all = num_data_trn - self.num_nn_pos - 1
            num_pos_batch = min(self.num_pos_max, self.num_nn_pos)
            num_neg_batch = self.batch_size - num_pos_batch - 1

            for idx in trange(num_data_trn):
            # random pos/neg pairs
                # random positive images
                rand_smp_pos = np.random.choice(num_pos_all, num_pos_batch, replace=False)
                list_smp_pos = self.list_nn['trn'][idx][rand_smp_pos]
                #  random negative images
                rand_smp_neg = np.random.choice(num_neg_all, num_neg_batch, replace=False) + num_pos_all
                list_smp_neg = self.list_nn['trn'][idx][rand_smp_neg]
                pair = []                  
                pair.append(indices[0][idx])
                for idx_pos in range(num_pos_batch):
                    pair.append(indices[0][list_smp_pos[idx_pos]])
                for idx_neg in range(num_neg_batch):
                    pair.append(indices[0][list_smp_neg[idx_neg]]) 
                pairs.append(pair)                
        else:
            num_data_val = len(indices[1])      
            for idx in trange(num_data_val):
                rand_smp = np.random.choice(num_data_val-self.num_NN-1,self.batch_size-self.num_NN-1,replace=False)+self.num_NN
                rand_smp.sort()
                rand_smp = np.concatenate((np.arange(0,self.num_NN),rand_smp))
                list_smp = self.list_nn['val'][idx][rand_smp]
                dist_smp = self.dist_nn['val'][idx][rand_smp]                
                pair = []                  
                pair.append(indices[1][idx])                                      
                for list_idx in list_smp:
                    pair.append(indices[1][list_idx])
                pairs.append(pair)
                dist_pairs.append(dist_smp)                
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

        def hflip(img):
            """Img : N x C x H x W"""
            return torch.from_numpy(np.flip(img.numpy(),-1).copy())

        # Get the index of each image in the triplet
        input = torch.FloatTensor(self.batch_size,3,224,224).fill_(0)
        for img_idx in range(self.batch_size):
            imgs_path = self.training_pairs[index][img_idx]
            # transform images if required  
            input[img_idx] = transform(imgs_path)
        # Random hflip
        if random.random() > 0.5:
            input = hflip(input)
        return input, index

    def __len__(self):
        return len(self.training_pairs)