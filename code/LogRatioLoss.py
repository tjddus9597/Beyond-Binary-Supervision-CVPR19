from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn
from utils import L2dist

class LogRatioLoss(Function):
    """Log ratio loss function. """
    def __init__(self):
        super(LogRatioLoss, self).__init__()
        self.mnd = 10
        self.mxd = 100
        self.pdist = L2dist(2)  # norm 2

    def forward(self, input, gt_dist):
        m = input.size()[0]-1   # #paired
        a = input[0]            # anchor
        p = input[1:]           # paired
        
        # auxiliary variables
        idxs = torch.arange(1, m+1).cuda()
        indc = idxs.repeat(m,1).t() < idxs.repeat(m,1)

        epsilon = 1e-6

        dist = self.pdist.forward(a,p)

        log_dist = torch.log(dist + epsilon)
        log_gt_dist = torch.log(gt_dist + epsilon)
        diff_log_dist = log_dist.repeat(m,1).t()-log_dist.repeat(m, 1)
        diff_log_gt_dist = log_gt_dist.repeat(m,1).t()-log_gt_dist.repeat(m, 1)

        # uniform weight coefficients 
        wgt = indc.clone().float()
        wgt = wgt.div(wgt.sum())

        log_ratio_loss = (diff_log_dist-diff_log_gt_dist).pow(2)

        loss = log_ratio_loss
        loss = loss.mul(wgt).sum()

        return loss