"""@package Model for Colorization following Dense-Net
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Net():
    """
    FCL after Dense-Net
    """

    def __init__(self, use_cuda=False):
        self.use_cuda = use_cuda
        self.fcl = nn.Linear(in_features, out_features)

    def forward(self, feature_list):
        out = self.fcl(feature_list)
        return out
