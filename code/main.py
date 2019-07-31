from __future__ import print_function

import argparse
import datetime
import math
import os
import random

import numpy as np
import scipy.io as sio
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from tensorboardX import SummaryWriter
from torch.autograd import Function, Variable
from torchvision.datasets import ImageFolder

from EvalDataset import EvalDataset

from model import *
from Naive_PairDataset import Naive_PairDataset
from Dense_PairDataset import Dense_PairDataset
# from Dense_PairDataset_wgtsamp import Dense_PairDataset_wgtsamp
from tqdm import tqdm, trange
from utils import *
from transformation import *

from LogRatioLoss import LogRatioLoss
from Naive_TripletLoss import Naive_TripletLoss
from Dense_TripletLoss import Dense_TripletLoss


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Pose Embedding')
# Model options
parser.add_argument('--dataroot', type=str, default='./../data/fullbody',
                    help='path to dataset')
parser.add_argument('--log-dir', default='./../results',
                    help='folder to output model checkpoints')
parser.add_argument('--result-name', default='regularized_boundedhinge',
                    help='name of this project')
parser.add_argument('--resume', type=int, default = 0, metavar='CHECKPOINT',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=30, metavar='E',
                    help='number of epochs to train (default: 10)')

# Training options
parser.add_argument('--model', type=str, default='resnet34',
                    help='Pretrained model name')
parser.add_argument('--embedding-size', type=int, default=128, metavar='ES',
                help='Dimensionality of the embedding')
parser.add_argument('--size-input', type=int, default=224,
                    help='size of model input')
parser.add_argument('--num-NN', type=int, default=5,
                    help='a few nearest neighbors must be included')
parser.add_argument('--batch-size', type=int, default=130, metavar='BS',
                    help='input batch size for training (default: 130)')
parser.add_argument('--test-batch-size', type=int, default= 50, metavar='BST',
                    help='input batch size for testing (default: 130)')
parser.add_argument('--is-norm', default=False,
                    help='Normalization layer for Unit sphere')
parser.add_argument('--sampling', default='dense',
                    help='How to Sampling')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=0, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')                  
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')

# Ignore ratio parameter
parser.add_argument('--ign', type=float, default=0.75,
                    help='Ignore ratio (default : 0.75')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--data-parallel', default=False,
                    help='Model data parallel')
parser.add_argument('--seed', type=int, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=20, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
if not args.data_parallel:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.seed is None:
    args.seed = random.randint(1, 10000)
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if args.cuda:
    cudnn.benchmark = True

if args.is_norm:
    LOG_DIR = args.log_dir + '/{}_pairwise_{}_{}_{}/fullbody_{}_lr{}_bat{}_numNN{}_emb{}'\
    .format(args.model,'norm', args.sampling, args.result_name, args.optimizer, args.lr, 
    args.batch_size, args.num_NN, args.embedding_size)
else: 
    LOG_DIR = args.log_dir + '/{}_pairwise_{}_{}_{}/fullbody_{}_lr{}_bat{}_numNN{}_emb{}'\
    .format(args.model,'unnorm',args.sampling, args.result_name, args.optimizer, args.lr, 
    args.batch_size, args.num_NN, args.embedding_size)

if args.result_name.find('triplet')+1:
    if args.is_norm:
        LOG_DIR = args.log_dir + '/{}_pairwise_{}_{}_{}/fullbody_{}_lr{}_lrd{}_bat{}_numNN{}_emb{}'\
            .format(args.model,'norm', args.sampling, args.result_name, args.optimizer, args.lr, args.lr_decay, 
            args.batch_size, args.num_NN, args.embedding_size)

args.resume = '{}/checkpoint_{}.pth'.format(LOG_DIR, args.resume)

# tensorboard writer
writer = SummaryWriter('{}/{}'.format(LOG_DIR,'run'))

num_data_trn = 12366
num_data_val = 9919

kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
device = torch.device("cuda" if args.cuda else "cpu")

# normalization (BGR sorted)
fullbody_std = 1.0 / 255.0
fullbody_mean = [103.939 / 255.0, 116.779 / 255.0, 123.68 / 255.0]

transform_trn = transforms.Compose([
                        ConvertBGR(),
                        RandTranslation(288, 224),
                        Scale(224),
                        transforms.ToTensor(),                     
                        transforms.Normalize(mean = fullbody_mean, 
                                            std = 3*[fullbody_std])
                     ])

transform_eval = transforms.Compose([
                        ConvertBGR(),
                        CenterTranslation(224),               
                        transforms.ToTensor(),
                        transforms.Normalize(mean = fullbody_mean, 
                                            std = 3*[fullbody_std])
                    ])

eval_dir = EvalDataset(dir=args.dataroot, batch_size = args.test_batch_size, transform=transform_eval)
eval_loader = torch.utils.data.DataLoader(eval_dir, batch_size=1, shuffle = False, **kwargs)

def main():
    os.system('clear')
    # instantiate model and initialize weights
    if args.model == 'resnet34':
        model = PoseModel_Resnet34(embedding_size=args.embedding_size, pretrained=True, is_norm=args.is_norm)
    elif args.model == 'resnet50':
        model = PoseModel_Resnet18(embedding_size=args.embedding_size, pretrained=True, is_norm=args.is_norm)
    elif args.model == 'resnet18':
        model = PoseModel_Resnet18(embedding_size=args.embedding_size, pretrained=True, is_norm=args.is_norm)
    model = model.to(device)

    if args.data_parallel:
        model = nn.DataParallel(model)

    optimizer = create_optimizer(model, args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.param_groups[0]['lr'] = checkpoint('lr')
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    end = start + args.epochs

    for epoch in range(start, end):
        if (args.sampling =='dense'):
            train_dir = Dense_PairDataset(dir=args.dataroot, num_NN = args.num_NN, is_trn = True,
            batch_size = args.batch_size, transform = transform_trn)
        if (args.sampling =='naive'):
            train_dir = Naive_PairDataset(dir=args.dataroot, num_NN = args.num_NN, is_trn = True,
            batch_size = args.batch_size,  epoch = epoch, transform = transform_trn)
        train_loader = torch.utils.data.DataLoader(train_dir, batch_size=1, shuffle=True, **kwargs)                
        if (args.sampling =='dense'):
            train(train_loader, model, optimizer, epoch)
        if (args.sampling =='naive'):
            naive_train(train_loader, model, optimizer, epoch)            
        evaluation(eval_loader, model, epoch)                                     

def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))

    for batch_idx, (input_pairs, gt_dist, img_idx) in pbar:
        num_data_taken = epoch*num_data_trn + batch_idx        
        input_pairs, gt_dist  = torch.squeeze(input_pairs).float(), torch.squeeze(gt_dist).float()
        input_pairs, gt_dist  = input_pairs.to(device), gt_dist.to(device)

        optimizer.zero_grad()
        # compute output
        conv_features, embed = model.forward(input_pairs)

        loss = LogRatioLoss().forward(embed, gt_dist)
        if args.result_name.find('triplet')+1:
            loss = Dense_TripletLoss().forward(embed, gt_dist)

        # compute gradient and update weights
        loss.backward()
        optimizer.step()

        # update the optimizer learning rate
        adjust_learning_rate(optimizer)        
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], num_data_taken)

        # log loss value
        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] img: {} Loss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                img_idx.item(), loss.item()))
        
        if batch_idx % args.log_interval == 0:
            writer.add_scalar('log_ratio_loss', loss.item(), num_data_taken)

    # do checkpointing
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'lr':optimizer.param_groups[0]['lr']},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))

def naive_train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()

    num_nn_pos = 30
    num_nn_neg_min = 1000
    num_nn_neg_red = 3000
    num_pos_max = 5
    num_pos_all = num_nn_pos
    num_neg_all = max(num_data_trn - (num_nn_pos + 1) - ((epoch - 1) * num_nn_neg_red), num_nn_neg_min)
    num_pos_batch = min(num_pos_max, num_nn_pos)
    num_neg_batch = args.batch_size - num_pos_batch - 1

    pbar = tqdm(enumerate(train_loader))

    for batch_idx, (input_pairs, img_idx) in pbar:
        num_data_taken = epoch*num_data_trn + batch_idx        
        input_pairs  = torch.squeeze(input_pairs).float()
        input_pairs = input_pairs.to(device)

        optimizer.zero_grad()
        # compute output
        conv_features, embed = model.forward(input_pairs)
        embed_anc = embed[0]
        embed_pos = embed[1:num_pos_batch+1]
        embed_neg = embed[num_pos_batch+1:args.batch_size]

        # generate triplets
        triplets = torch.FloatTensor(3, num_pos_batch * num_neg_batch, args.embedding_size).fill_(0)
        for idx_pos in range(num_pos_batch):
            for idx_neg in range(num_neg_batch):
                idx_tri = idx_pos * num_neg_batch + idx_neg
                triplets[0,idx_tri] = embed_anc
                triplets[1,idx_tri] = embed_pos[idx_pos]
                triplets[2,idx_tri] = embed_neg[idx_neg]
       
        if args.result_name.find('triplet')+1:
            loss = Naive_TripletLoss(mrg=0.2).forward(triplets)

        # compute gradient and update weights
        loss.backward()
        optimizer.step()

        # update the optimizer learning rate
        adjust_learning_rate(optimizer)        
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], num_data_taken)

        # log loss value
        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] img: {} Loss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                img_idx.item(), loss.item()))
        
        if batch_idx % args.log_interval == 0:
            writer.add_scalar('log_ratio_loss', loss.item(), num_data_taken)

    # do checkpointing
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'lr':optimizer.param_groups[0]['lr']},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))

def evaluation(eval_loader, model, epoch):
    print('== Embedding validation data')
    # switch to test mode
    model.eval()
    embed_all  = torch.FloatTensor(num_data_val, args.embedding_size).fill_(0)
    pbar = tqdm(enumerate(eval_loader))
    for batch_idx, input_pairs in pbar:
        idx_val_stt = batch_idx *  args.test_batch_size
        idx_val_end = min((batch_idx + 1) * args.test_batch_size, num_data_val)    
        input_pairs  = torch.squeeze(input_pairs).to(device)
        with torch.no_grad():
            conv_features, embed = model.forward(input_pairs)
        embed_eval = embed.detach()[:(idx_val_end-idx_val_stt)]
        embed_all[idx_val_stt:idx_val_end] = embed_eval.double()
    
    test1_embed, test2_embed, test3_embed = eval_retrieval(args.dataroot, embed_all, num_data_val) 
   
    writer.add_scalar('test1_embed', test1_embed[-1], epoch)
    writer.add_scalar('test2_embed', test2_embed[-1], epoch)
    writer.add_scalar('test3_embed', test3_embed[-1], epoch)

    print('Test1 : {}, Test2 : {}, Test3 : {}'.format(test1_embed[-1], test2_embed[-1], test3_embed[-1]))
    
    sio.savemat('{}/pose_emb_{}.mat'.format(LOG_DIR,epoch), {'x' : embed_all.numpy()})
    if not os.path.exists('{}/retrieval'.format(LOG_DIR)):
        os.makedirs('{}/retrieval'.format(LOG_DIR))
    sio.savemat('{}/retrieval/retrieval_scores_embed_{}.mat'.format(LOG_DIR,epoch), 
    {'test1_embed' : test1_embed, 'test2_embed' : test2_embed, 'test3_embed' : test3_embed})
    
    # Embedding Projector save file
    if not os.path.exists('{}/embedding'.format(LOG_DIR)):
        os.makedirs('{}/embedding'.format(LOG_DIR))   
    np.savetxt('{}/embedding/embedding_val_{}.tsv'.format(LOG_DIR,epoch), X=embed_all, delimiter = "\t") 

def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = args.lr / (1 + group['step'] * args.lr_decay)


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                                momentum=0.9,
                                weight_decay= 5e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
                                lr = args.lr,                            
                                weight_decay=5e-4)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(),
                                lr = args.lr,      
                                rho = 0.95,
                                eps = 1e-06)
    return optimizer

if __name__ == '__main__':
    main()
