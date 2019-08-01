
import math
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio
import scipy.misc
import torch
from torch.autograd import Function, Variable


class Squared_L2dist(Function):
    def __init__(self, p):
        super(Squared_L2dist, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        eps = 1e-4 / x1.size(0)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return out + eps

class L2dist(Function):
    def __init__(self, p):
        super(L2dist, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        eps = 1e-4 / x1.size(0)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)


def eval_retrieval(dir, embed_all, num_data_val):
    # parameters
    num_max_NN = 10
    thres_nn = 50
    num_query = 1919

    fpath_val = os.path.join(dir, 'dist_pose_val.mat')
    dist_file_val = sio.loadmat(fpath_val)
    dist_pose = dist_file_val['dist_pose_val']
    qdist_pose = dist_pose[:num_query, num_query:num_data_val]
    # qdist_pose_sorted = np.sort(qdist_pose, 1)
    NN_pose = np.argsort(qdist_pose, 1)

    # pairwise distances between embedded vectors
    dist_emb = embed_all.pow(2).sum(1) + (-2) * embed_all.mm(embed_all.t())
    dist_emb = embed_all.pow(2).sum(1) + dist_emb.t()

    # distances between query and DB, on the embedding space
    qdist_emb = dist_emb[:num_query, num_query:num_data_val]
    NN_emb = np.argsort(qdist_emb, 1)

    # Test1 : Mean joint distance
    mean_distance = []
    for qidx in range(num_query):
        list_qdist = qdist_pose[qidx, NN_emb[qidx, :num_max_NN]]
        mean_distance.append(np.divide(np.cumsum(list_qdist),np.arange(1, num_max_NN+1)))
    mean_distance = np.mean(mean_distance, 0) * (224 / 288)

    # Test2 : Hit@K absolute based on NN
    # test2_embed = []
    # for qidx in range(num_query):
    #     list_retr_label = np.isin(NN_emb[qidx][:num_max_NN], NN_pose[qidx][:thres_nn])
    #     test2_embed.append(np.cumsum(list_retr_label) > 0)
    # test2_embed = np.mean(test2_embed, 0)

    # Test3 : continuous nDCG
    list_p = np.arange(10, 110, 10)
    nDCG = np.ndarray([num_query, len(list_p)])
    for pidx in range(len(list_p)):
        cur_p = list_p[pidx]
        discounts = 1/(np.log2(np.arange(1, cur_p+1)+1))
        for qidx in range(num_query):
            relevance = 2**(-np.log2(qdist_pose[qidx]+1))
            dcg_ideal = sum(np.multiply(relevance[NN_pose[qidx][:cur_p]], discounts))
            dcg_embed = sum(np.multiply(relevance[NN_emb[qidx][:cur_p]], discounts))
            nDCG[qidx][pidx] = dcg_embed / dcg_ideal
    nDCG = np.mean(nDCG, 0)

    return mean_distance, nDCG
