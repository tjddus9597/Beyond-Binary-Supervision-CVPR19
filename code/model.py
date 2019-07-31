import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import vgg16_bn
from torchvision.models import vgg19_bn
from torchvision.models import vgg16

import torch.utils.model_zoo as model_zoo

class PoseModel_Resnet18(nn.Module):
    def __init__(self,embedding_size, pretrained=True, is_norm=False):
        super(PoseModel_Resnet18, self).__init__()

        self.model = resnet18(pretrained)
        self.is_norm = is_norm

        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, self.embedding_size)
        self.not_training = [self.model.conv1]

        self._initialize_weights()

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)            
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        if self.is_norm:
           x = self.l2_norm(x)
        self.features = x

        return self.features

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.fc.weight, mode='fan_out')
        init.constant_(self.model.fc.bias, 0)

class PoseModel_Resnet34(nn.Module):
    def __init__(self,embedding_size, pretrained=True, is_norm=False):
        super(PoseModel_Resnet34, self).__init__()

        self.model = resnet34(pretrained)
        self.is_norm = is_norm

        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, self.embedding_size)
        self.not_training = [self.model.conv1]

        self._initialize_weights()

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)            
        x = x.view(x.size(0), -1)
        self.convfeatures = x
        x = self.model.fc(x)
        if self.is_norm:
           x = self.l2_norm(x)
        self.features = x

        return self.convfeatures, self.features

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.fc.weight, mode='fan_out')
        init.constant_(self.model.fc.bias, 0)

class PoseModel_Resnet50(nn.Module):
    def __init__(self,embedding_size, pretrained=True, is_norm=False):
        super(PoseModel_Resnet50, self).__init__()

        self.model = resnet50(pretrained)
        self.is_norm = is_norm

        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.fc1 = nn.Linear(self.num_ftrs, 512)
        self.model.fc2 = nn.Linear(512, self.embedding_size)
        self.model.fc = nn.Linear(self.num_ftrs, self.embedding_size)
        self.not_training = [self.model.conv1]

        self._initialize_weights()

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)            
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        # x = F.relu(self.model.fc1(x))
        # x = self.model.fc2(x)
        if self.is_norm:
           x = self.l2_norm(x)
        self.features = x

        return self.features

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.fc.weight, mode='fan_out')
        init.constant_(self.model.fc.bias, 0)