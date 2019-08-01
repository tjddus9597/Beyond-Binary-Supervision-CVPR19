from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils import Squared_L2dist

class Dense_TripletLoss(Function):
    """Log ratio loss function. """
    def __init__(self, mrg=0.03):
        super(Dense_TripletLoss, self).__init__()
        self.mrg = mrg
        self.pdist = Squared_L2dist(2)  # norm 2

    def forward(self, input, gt_dist):
        m = input.size()[0]-1   # #paired
        a = input[0]            # anchor
        p = input[1:]           # paired
        
        #  auxiliary variables
        idxs = torch.arange(1, m+1).cuda()
        indc = idxs.repeat(m,1).t() < idxs.repeat(m,1)

        epsilon = 1e-6

        dist = self.pdist.forward(a,p)

        # uniform weight coefficients 
        wgt = indc.clone().float()
        wgt = wgt.div(wgt.sum())

        loss = dist.repeat(m,1).t() - dist.repeat(m,1) + self.mrg
        loss = loss.clamp(min=1e-12)
    
        loss = loss.mul(wgt).sum()
        return loss