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
import numpy as np 

class Naive_TripletLoss(Function):
    """Log ratio loss function. """
    def __init__(self, mrg=0.2):
        super(Naive_TripletLoss, self).__init__()
        self.mrg = mrg

    def Squared_L2dist(self, x1, x2, norm=2):
        eps = 1e-4 / x1.size(0)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, norm).sum(0)
        return out + eps

    def forward(self, input):
        a = input[0]            # anchor
        p = input[1]            # positive
        n = input[2]            # negative
        N = a.size(0)           # #acnhor

        Li = torch.FloatTensor(N)

        for i in range(N):
            Li[i] = (self.Squared_L2dist(a[i],p[i])-self.Squared_L2dist(a[i],n[i])+self.mrg).clamp(min=1e-12)

        loss = Li.sum().div(N)
        return loss